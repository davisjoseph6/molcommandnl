#!/usr/bin/env python3
"""
molcommand/context_history.py

LLM-steered, config-driven context gating + rolling interaction history.

This module is designed to help MolCommand decide **when** to include scene context
and **what** context to include, without hardcoding logic into the pipeline.

Key ideas:
- Maintain a bounded history of user utterances + executed DSL (and optional metadata).
- Track an "update_count" to represent scene complexity growth.
- Gate context usage using:
  (a) the LLM's RequiresContext decision (primary),
  (b) a configurable scene-update threshold (bias),
  (c) lightweight ambiguity heuristics (fallback / safety).

Outputs from this module are plain strings/structures that you can inject into your
LLM prompting layer (system prompt + user prompt + optional context block).

Configuration:
- You should load configuration from YAML/JSON (preferred) and keep project-specific
  policies there. This module includes small defaults as a safe fallback only.

Typical integration (pseudo):
    history = ContextHistory.from_paths(["config/molcommand.yaml", "config/context.yaml"])
    turn_id = history.add_user(utterance)

    # LLM classifier yields: categories, requires_context
    use_ctx = history.should_use_context(utterance, llm_requires_context=requires_context)

    ctx_block = history.build_context_block(utterance) if use_ctx else ""
    prompt = system_prompt + "\n\n" + user_prompt.format(utterance=utterance) + "\n\n" + ctx_block

    dsl = transpile(...)
    history.add_execution(turn_id, dsl=dsl, categories=categories,
                          llm_requires_context=requires_context,
                          did_update_scene=history.infer_scene_update_from_dsl(dsl))
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Optional YAML loader
# ----------------------------
def _try_load_yaml(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse YAML if PyYAML is installed; otherwise return None."""
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


# ----------------------------
# Defaults (fallback only)
# ----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "context": {
        "history": {
            "max_turns": 40,
            "max_chars_per_turn": 600,
            "persist_path": None,  # e.g. "data/context_history.jsonl"
        },
        "gating": {
            # After this many scene updates, we bias toward using context.
            "update_threshold": 8,
            # If LLM says RequiresContext=true => always use context.
            # If LLM says false/unknown => use heuristics + threshold bias.
            "enable_ambiguity_heuristics": True,
            "ambiguity": {
                # Very lightweight textual ambiguity signals
                "pronouns": ["it", "this", "that", "those", "these", "them", "he", "she", "they"],
                "vague_nouns": ["protein", "structure", "model", "chain", "residue", "atom", "molecule"],
                "definite_refs": ["the", "previous", "last", "same", "above", "earlier"],
            },
        },
        "retrieval": {
            # How many history turns to include in context block
            "top_k": 10,
            # Retrieval mode: "recent" or "hybrid"
            "mode": "hybrid",
            # Weight for recency in scoring (0..1). hybrid uses both lexical overlap + recency
            "recency_weight": 0.35,
            # Max characters in the final context block
            "max_context_chars": 3500,
        },
        "update_heuristics": {
            # Prefer config-driven verb lists; these are fallbacks only.
            "dsl_update_verbs": [
                "load", "add", "delete", "remove", "hide", "show", "select",
                "color", "rename", "center", "zoom", "focus", "align",
                "surface", "cartoon", "sticks", "spheres", "label",
                "set", "unset", "traj", "trajectory",
            ],
        },
        "format": {
            # Context block formatting
            "header": "Context (recent & relevant):",
            "turn_prefix_user": "User",
            "turn_prefix_exec": "Exec",
        },
    }
}


# ----------------------------
# Utilities
# ----------------------------
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dictionaries (b overrides a)."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for lexical overlap scoring."""
    text = text.lower()
    return re.findall(r"[a-z0-9_]+", text)


def _clip(s: str, max_chars: int) -> str:
    """Clip string to max_chars with ellipsis."""
    if len(s) <= max_chars:
        return s
    if max_chars <= 3:
        return s[:max_chars]
    return s[: max_chars - 3] + "..."


def _now_ts() -> float:
    return time.time()


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class ContextTurn:
    """One interaction 'turn' with optional execution/metadata."""
    turn_id: int
    ts: float
    utterance: str

    # Optional fields populated after transpilation/execution
    dsl: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    llm_requires_context: Optional[bool] = None
    did_update_scene: Optional[bool] = None

    # Any extra metadata (scene snapshot hash, selected structure id, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Main class
# ----------------------------
class ContextHistory:
    """
    Rolling history store + context gating + retrieval.

    This class is intentionally self-contained and does NOT call an LLM.
    It accepts the LLM classifier's output (RequiresContext) as an input
    and applies additional config-driven policies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self._turns: List[ContextTurn] = []
        self._next_turn_id: int = 1

        # A simple proxy for "scene complexity"
        self._update_count: int = 0

        persist_path = self._persist_path()
        if persist_path:
            self._load_persisted_history(persist_path)

    # -------------
    # Construction
    # -------------
    @classmethod
    def from_paths(cls, paths: Iterable[str]) -> "ContextHistory":
        """
        Load config from one or more YAML/JSON files (later files override earlier ones).
        Missing files are ignored (but logged).
        """
        merged: Dict[str, Any] = {}
        for p in paths:
            data = cls._load_config_file(p)
            if data:
                merged = _deep_merge(merged, data)
        return cls(config=merged)

    @staticmethod
    def _load_config_file(path: str) -> Optional[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            LOGGER.debug("Config file not found (ignored): %s", path)
            return None

        text = p.read_text(encoding="utf-8")
        # Try JSON first (fast + strict), then YAML if available.
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        y = _try_load_yaml(text)
        if y is not None:
            return y

        LOGGER.warning("Could not parse config file as JSON or YAML: %s", path)
        return None

    # ----------------
    # Public API
    # ----------------
    def reset(self) -> None:
        """Clear all history and counters (in-memory only)."""
        self._turns = []
        self._next_turn_id = 1
        self._update_count = 0

    def add_user(self, utterance: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a user utterance to history. Returns the assigned turn_id.

        You can attach optional metadata (e.g., scene snapshot pointer, UI selection state).
        """
        utterance = utterance.strip()
        if not utterance:
            raise ValueError("utterance must be non-empty")

        t = ContextTurn(
            turn_id=self._next_turn_id,
            ts=_now_ts(),
            utterance=self._truncate_utterance(utterance),
            meta=dict(meta or {}),
        )
        self._turns.append(t)
        self._next_turn_id += 1
        self._enforce_bounds()
        self._persist_append_if_enabled(t)
        return t.turn_id

    def add_execution(
        self,
        turn_id: int,
        dsl: str,
        categories: Optional[List[str]] = None,
        llm_requires_context: Optional[bool] = None,
        did_update_scene: Optional[bool] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Attach transpiled DSL + metadata to a previously added user turn.

        If did_update_scene is True, update_count increments (scene complexity grows).
        """
        t = self._find_turn(turn_id)
        t.dsl = _clip(dsl.strip(), self._max_chars_per_turn())
        t.categories = list(categories or [])
        t.llm_requires_context = llm_requires_context
        t.did_update_scene = did_update_scene
        if meta:
            t.meta.update(meta)

        if did_update_scene:
            self._update_count += 1

        # Persist update by rewriting if enabled
        self._persist_rewrite_if_enabled()

    def update_count(self) -> int:
        """Return the current number of recorded scene updates."""
        return self._update_count

    # ----------------
    # Context gating
    # ----------------
    def should_use_context(
        self,
        utterance: str,
        llm_requires_context: Optional[bool],
        *,
        update_count_override: Optional[int] = None,
    ) -> bool:
        """
        Decide whether to include scene context for this utterance.

        Priority:
        1) If LLM says RequiresContext=True -> True
        2) If LLM says RequiresContext=False -> maybe True if ambiguity heuristics trigger strongly
           AND/OR update_count exceeds threshold (bias)
        3) If LLM is unknown (None) -> use heuristics + threshold bias
        """
        if llm_requires_context is True:
            return True

        gating = self._cfg("context.gating", {})
        threshold = int(gating.get("update_threshold", 0))
        enable_heur = bool(gating.get("enable_ambiguity_heuristics", True))

        upd = int(update_count_override) if update_count_override is not None else self._update_count
        threshold_reached = (threshold > 0 and upd >= threshold)

        if not enable_heur:
            # If heuristics disabled, only threshold biases unknown/false toward context.
            return threshold_reached and (llm_requires_context is None)

        ambiguity_score = self._ambiguity_score(utterance)

        # When scene is complex, we lower the bar for context usage.
        if threshold_reached:
            # If LLM is unsure OR says false but ambiguity is non-trivial, use context.
            if llm_requires_context is None:
                return True
            return ambiguity_score >= 0.25

        # Before threshold, be more conservative:
        if llm_requires_context is None:
            return ambiguity_score >= 0.45
        # LLM says false:
        return ambiguity_score >= 0.65

    # ----------------
    # Retrieval / formatting
    # ----------------
    def build_context_block(
        self,
        utterance: str,
        *,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Build a context block from stored history (recent + relevant), suitable for prompt injection.
        """
        retrieval = self._cfg("context.retrieval", {})
        fmt = self._cfg("context.format", {})

        k = int(top_k) if top_k is not None else int(retrieval.get("top_k", 10))
        max_chars = int(max_chars) if max_chars is not None else int(retrieval.get("max_context_chars", 3500))

        selected = self.retrieve_relevant(utterance, top_k=k)
        if not selected:
            return ""

        lines: List[str] = []
        header = str(fmt.get("header", "Context:")).strip()
        if header:
            lines.append(header)

        user_prefix = str(fmt.get("turn_prefix_user", "User")).strip()
        exec_prefix = str(fmt.get("turn_prefix_exec", "Exec")).strip()

        for t in selected:
            lines.append(f"{user_prefix} (t{t.turn_id}): {t.utterance}")
            if t.dsl:
                lines.append(f"{exec_prefix} (t{t.turn_id}): {t.dsl}")

        block = "\n".join(lines).strip()
        return _clip(block, max_chars)

    def retrieve_relevant(self, utterance: str, *, top_k: int) -> List[ContextTurn]:
        """
        Retrieve relevant turns from history.

        Modes:
        - recent: last K turns (most recent first)
        - hybrid: lexical overlap + recency bias
        """
        retrieval = self._cfg("context.retrieval", {})
        mode = str(retrieval.get("mode", "hybrid")).lower()
        recency_weight = float(retrieval.get("recency_weight", 0.35))

        if not self._turns:
            return []

        if mode == "recent":
            return list(reversed(self._turns[-top_k:]))

        # hybrid scoring: overlap + recency
        query_tokens = set(_tokenize(utterance))
        if not query_tokens:
            return list(reversed(self._turns[-top_k:]))

        scored: List[Tuple[float, ContextTurn]] = []
        n = len(self._turns)
        for idx, t in enumerate(self._turns):
            t_tokens = set(_tokenize(t.utterance + " " + (t.dsl or "")))
            if not t_tokens:
                overlap = 0.0
            else:
                inter = len(query_tokens.intersection(t_tokens))
                union = len(query_tokens.union(t_tokens))
                overlap = float(inter) / float(union) if union else 0.0

            # recency: newest turn gets score 1.0, oldest ~ 0.0
            rec = 1.0 if n <= 1 else float(idx) / float(n - 1)
            score = (1.0 - recency_weight) * overlap + recency_weight * rec
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [t for _, t in scored[:top_k]]
        # Present most recent first for readability
        best.sort(key=lambda t: t.turn_id, reverse=True)
        return best

    # ----------------
    # Update inference helper
    # ----------------
    def infer_scene_update_from_dsl(self, dsl: str) -> bool:
        """
        Best-effort inference: decide if a DSL string likely updates the scene.

        Prefer using explicit signals from your pipeline. This is only a fallback.
        """
        dsl = dsl.strip().lower()
        if not dsl:
            return False

        heur = self._cfg("context.update_heuristics", {})
        verbs = heur.get("dsl_update_verbs", [])
        if not isinstance(verbs, list):
            verbs = []

        # Extremely lightweight: does any configured verb appear as a word boundary?
        for v in verbs:
            v = str(v).strip().lower()
            if not v:
                continue
            if re.search(rf"\b{re.escape(v)}\b", dsl):
                return True

        return False

    # ----------------------------
    # Internals
    # ----------------------------
    def _ambiguity_score(self, utterance: str) -> float:
        """
        Return a score in [0,1] indicating how likely the utterance needs context
        due to ambiguous references.
        """
        text = utterance.strip().lower()
        if not text:
            return 0.0

        gating = self._cfg("context.gating", {})
        amb = gating.get("ambiguity", {}) if isinstance(gating, dict) else {}
        pronouns = set(map(str.lower, amb.get("pronouns", []))) if isinstance(amb, dict) else set()
        vague_nouns = set(map(str.lower, amb.get("vague_nouns", []))) if isinstance(amb, dict) else set()
        definite_refs = set(map(str.lower, amb.get("definite_refs", []))) if isinstance(amb, dict) else set()

        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        # Signals
        has_pronoun = any(t in pronouns for t in tokens)
        has_vague_noun = any(t in vague_nouns for t in tokens)
        has_definite = any(t in definite_refs for t in tokens)

        # Also detect patterns like "the chain", "the residue", "that structure"
        pattern_ref = bool(re.search(r"\b(the|this|that|those|these)\s+(protein|structure|model|chain|residue|atom)\b", text))

        # Basic scoring heuristic
        score = 0.0
        if has_pronoun:
            score += 0.55
        if pattern_ref:
            score += 0.35
        if has_definite and has_vague_noun:
            score += 0.25
        elif has_definite:
            score += 0.10
        elif has_vague_noun:
            score += 0.10

        # Normalize
        if score > 1.0:
            score = 1.0
        return score

    def _find_turn(self, turn_id: int) -> ContextTurn:
        for t in self._turns:
            if t.turn_id == turn_id:
                return t
        raise KeyError(f"turn_id not found: {turn_id}")

    def _enforce_bounds(self) -> None:
        max_turns = int(self._cfg("context.history.max_turns", 40))
        if max_turns <= 0:
            return
        if len(self._turns) > max_turns:
            # Drop oldest turns
            self._turns = self._turns[-max_turns:]

            # NOTE: We do NOT decrement update_count because it reflects cumulative scene changes.
            # If you want update_count to reflect only active window, store a per-turn counter and recompute.

    def _truncate_utterance(self, utterance: str) -> str:
        return _clip(utterance, self._max_chars_per_turn())

    def _max_chars_per_turn(self) -> int:
        return int(self._cfg("context.history.max_chars_per_turn", 600))

    def _persist_path(self) -> Optional[Path]:
        p = self._cfg("context.history.persist_path", None)
        if not p:
            return None
        try:
            return Path(str(p)).expanduser().resolve()
        except Exception:
            return None

    def _persist_append_if_enabled(self, turn: ContextTurn) -> None:
        p = self._persist_path()
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(turn)
        record["_kind"] = "turn"
        record["_update_count"] = self._update_count
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_rewrite_if_enabled(self) -> None:
        """
        Rewrite the whole persisted file to reflect updates (when we attach DSL to a prior turn).
        JSONL is append-friendly, but rewriting keeps it simple and consistent for debugging.
        """
        p = self._persist_path()
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for t in self._turns:
                record = asdict(t)
                record["_kind"] = "turn"
                record["_update_count"] = self._update_count
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_persisted_history(self, path: Path) -> None:
        """
        Load existing JSONL history if present. This is best-effort:
        - If file is malformed, we ignore offending lines.
        - We restore turns + update_count.
        """
        if not path.exists():
            return

        turns: List[ContextTurn] = []
        update_count = 0
        max_turn_id = 0

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("_kind") != "turn":
                continue

            try:
                t = ContextTurn(
                    turn_id=int(rec.get("turn_id", 0)),
                    ts=float(rec.get("ts", _now_ts())),
                    utterance=str(rec.get("utterance", "")),
                    dsl=rec.get("dsl", None),
                    categories=list(rec.get("categories", []) or []),
                    llm_requires_context=rec.get("llm_requires_context", None),
                    did_update_scene=rec.get("did_update_scene", None),
                    meta=dict(rec.get("meta", {}) or {}),
                )
            except Exception:
                continue

            if t.turn_id <= 0 or not t.utterance:
                continue

            turns.append(t)
            max_turn_id = max(max_turn_id, t.turn_id)
            try:
                update_count = max(update_count, int(rec.get("_update_count", 0)))
            except Exception:
                pass

        # Load into memory
        if turns:
            turns.sort(key=lambda x: x.turn_id)
            self._turns = turns
            self._next_turn_id = max_turn_id + 1
            self._update_count = update_count
            self._enforce_bounds()

    def _cfg(self, dotted_key: str, default: Any) -> Any:
        """
        Read config key using dotted path, e.g. "context.history.max_turns".
        """
        cur: Any = self.config
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur
