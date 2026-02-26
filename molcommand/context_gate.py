#!/usr/bin/env python3
"""
molcommand.context_gate

LLM-steered context gate.

Decides whether an utterance should be processed with scene context, using:
1) An LLM classifier (entity categories + RequiresContext)
2) A config-driven threshold policy (update_count / complexity score)

This module is intentionally policy-driven and contains minimal hardcoding.

Patch notes:
- Supports multiple LLM client method names/signatures (complete/chat/generate/invoke/...)
- Gracefully extracts text from common response shapes (str/dict/object)
- Tolerates fenced YAML/JSON output and parse failures more robustly
- FAIL-OPEN behavior: if LLM call fails, we still apply policy bias (threshold/complexity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import re

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ContextDecision:
    categories: List[str]
    requires_context: bool
    raw: str


class ContextGate:
    """LLM-based context gating with config-driven thresholds."""

    def __init__(self, policy: Dict[str, Any], llm_client: Any) -> None:
        self.policy = policy or {}
        self.llm_client = llm_client

        gate_cfg = self.policy.get("context_gate", {}) if isinstance(self.policy, dict) else {}
        llm_cfg = gate_cfg.get("llm", {}) if isinstance(gate_cfg, dict) else {}

        self.update_threshold = int(gate_cfg.get("update_threshold", 0))
        self.complexity_cfg = gate_cfg.get("complexity", {"enabled": False})

        self.output_format = str(llm_cfg.get("output_format", "yaml")).lower()
        self.max_retries = int(llm_cfg.get("max_retries", 1))

        system_prompt_path = llm_cfg.get("system_prompt_path")
        user_prompt_path = llm_cfg.get("user_prompt_path")
        self.system_prompt = self._read_text(system_prompt_path) if system_prompt_path else ""
        self.user_prompt_tmpl = self._read_text(user_prompt_path) if user_prompt_path else "{utterance}"

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _complexity_score(self, scene_stats: Dict[str, Any]) -> float:
        if not self.complexity_cfg.get("enabled", False):
            return 0.0
        weights = self.complexity_cfg.get("weights", {})
        return (
            float(weights.get("updates", 0.0)) * float(scene_stats.get("updates", 0))
            + float(weights.get("structures", 0.0)) * float(scene_stats.get("structures", 0))
            + float(weights.get("selections", 0.0)) * float(scene_stats.get("selections", 0))
        )

    def decide(self, utterance: str, scene_stats: Dict[str, Any]) -> ContextDecision:
        """
        Return categories + requires_context, applying threshold/complexity bias.

        FAIL-OPEN behavior:
        - If the LLM call fails, we return empty categories + requires_context=False initially,
          but we still apply policy bias (updates threshold / complexity threshold).
        """
        user_prompt = self.user_prompt_tmpl.format(utterance=utterance)

        raw = ""
        categories: List[str] = []
        requires = False

        # Try to call classifier. If it fails, we still proceed with policy bias.
        last_err: Optional[Exception] = None
        for _ in range(max(1, self.max_retries)):
            try:
                raw = self._call_llm_classifier(system=self.system_prompt, user=user_prompt)
                categories, requires = self._parse_output(raw)
                last_err = None
                break
            except Exception as e:
                last_err = e
                raw = raw or ""
                categories = []
                requires = False

        if last_err is not None and not raw:
            # keep raw empty; caller can trace decide_error if they catch exceptions around decide()
            pass

        # Policy bias: if scene is complex enough, force context (config-driven).
        updates = int(scene_stats.get("updates", 0))
        if updates >= self.update_threshold:
            requires = True

        score = self._complexity_score(scene_stats)
        threshold = float(self.complexity_cfg.get("threshold", 0.0))
        if self.complexity_cfg.get("enabled", False) and score >= threshold:
            requires = True

        return ContextDecision(categories=categories, requires_context=requires, raw=raw)

    # ------------------------------------------------------------------
    # LLM adapter logic (works with multiple client interfaces)
    # ------------------------------------------------------------------
    def _call_llm_classifier(self, system: str, user: str) -> str:
        """
        Call the injected LLM client using best-effort adapters for common method names
        and signatures.

        Tries methods in this order:
          complete, complete_text, completion, chat, chat_completion, generate, invoke, ask,
          call, run, predict, __call__
        """
        client = self.llm_client
        if client is None:
            raise RuntimeError("ContextGate.llm_client is None")

        messages = [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user or ""},
        ]
        combined_prompt = f"{system}\n\n{user}".strip()

        candidate_methods: List[str] = [
            "complete",
            "complete_text",
            "completion",
            "chat",
            "chat_completion",
            "generate",
            "invoke",
            "ask",
            "call",
            "run",
            "predict",
        ]
        if callable(client):
            candidate_methods.append("__call__")

        last_exc: Optional[Exception] = None

        for method_name in candidate_methods:
            fn = client if method_name == "__call__" else getattr(client, method_name, None)
            if not callable(fn):
                continue

            # Try several common signatures for each method.
            attempts = [
                # explicit system/user
                {"kwargs": {"system": system, "user": user}},
                {"kwargs": {"system_prompt": system, "user_prompt": user}},
                {"kwargs": {"system": system, "prompt": user}},
                # messages style
                {"kwargs": {"messages": messages}},
                {"args": [messages]},
                # prompt style
                {"kwargs": {"prompt": combined_prompt}},
                {"kwargs": {"input": combined_prompt}},
                {"kwargs": {"query": combined_prompt}},
                {"kwargs": {"text": combined_prompt}},
                {"args": [combined_prompt]},
                # minimal
                {"args": [user]},
                {"kwargs": {"prompt": user}},
                {"kwargs": {"input": user}},
                # (system, user)
                {"args": [system, user]},
            ]

            for attempt in attempts:
                try:
                    args = attempt.get("args", [])
                    kwargs = attempt.get("kwargs", {})
                    resp = fn(*args, **kwargs)
                    text = self._extract_text(resp)
                    if text is None:
                        continue
                    return text
                except TypeError as e:
                    last_exc = e
                    continue
                except Exception as e:
                    last_exc = e
                    continue

        if last_exc is not None:
            raise last_exc

        raise AttributeError(
            "LLM client has no compatible method/signature for ContextGate "
            "(tried complete/complete_text/completion/chat/chat_completion/generate/"
            "invoke/ask/call/run/predict/__call__)."
        )

    def _extract_text(self, resp: Any) -> Optional[str]:
        """Extract text from common response shapes returned by LLM clients."""
        if resp is None:
            return None

        if isinstance(resp, str):
            return resp

        # Common dict-based shapes
        if isinstance(resp, dict):
            # direct text-like keys
            for key in ("text", "content", "response", "output", "raw", "result", "completion", "output_text"):
                val = resp.get(key)
                if isinstance(val, str):
                    return val

            # OpenAI chat-like: {"choices": [{"message": {"content": "..."} }]}
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                if isinstance(ch0, dict):
                    msg = ch0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(ch0.get("text"), str):
                        return ch0["text"]

            # Anthropic-like: {"content": [{"text": "..."}]}
            content = resp.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                    return c0["text"]

            # OpenAI Responses-like: {"output": [{"content":[{"text":"..."}]}]}
            out = resp.get("output")
            if isinstance(out, list) and out:
                o0 = out[0]
                if isinstance(o0, dict):
                    oc = o0.get("content")
                    if isinstance(oc, list) and oc:
                        oc0 = oc[0]
                        if isinstance(oc0, dict) and isinstance(oc0.get("text"), str):
                            return oc0["text"]

            # messages-style
            msg = resp.get("message")
            if isinstance(msg, dict):
                content2 = msg.get("content")
                if isinstance(content2, str):
                    return content2

        # Object-based shapes
        for attr in ("text", "content", "response", "output", "raw", "result", "completion", "output_text"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, str):
                    return val

        # OpenAI-ish object shapes
        if hasattr(resp, "choices"):
            try:
                choices = getattr(resp, "choices")
                if choices:
                    ch0 = choices[0]
                    msg = getattr(ch0, "message", None)
                    if msg is not None:
                        content = getattr(msg, "content", None)
                        if isinstance(content, str):
                            return content
                    txt = getattr(ch0, "text", None)
                    if isinstance(txt, str):
                        return txt
            except Exception:
                pass

        # Some SDKs return list of blocks/chunks with .text
        if isinstance(resp, list):
            chunks: List[str] = []
            for item in resp:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                    chunks.append(getattr(item, "text"))
            if chunks:
                return "\n".join(chunks)

        return None

    @staticmethod
    def _strip_code_fences(raw: str) -> str:
        """Strip ```yaml / ```json fences if the model returns fenced output."""
        txt = (raw or "").strip()
        if not txt.startswith("```"):
            return txt

        lines = txt.splitlines()
        if not lines:
            return txt

        # Drop first fence line
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Drop last fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_output(self, raw: str) -> Tuple[List[str], bool]:
        txt = self._strip_code_fences((raw or "").strip())
        if not txt:
            return [], False

        # Preferred format first
        if self.output_format == "json":
            parsed = self._try_parse_json(txt)
            if parsed is not None:
                return parsed
            parsed = self._try_parse_yaml(txt)
            if parsed is not None:
                return parsed
        else:
            parsed = self._try_parse_yaml(txt)
            if parsed is not None:
                return parsed
            parsed = self._try_parse_json(txt)
            if parsed is not None:
                return parsed

        # Fallback regex parsing
        cats = self._regex_parse_categories(txt)
        req = self._regex_parse_requires_context(txt)
        return cats, req

    def _try_parse_json(self, txt: str) -> Optional[Tuple[List[str], bool]]:
        try:
            obj = json.loads(txt)
        except Exception:
            return None
        return self._coerce_output_object(obj)

    def _try_parse_yaml(self, txt: str) -> Optional[Tuple[List[str], bool]]:
        if yaml is None:
            return None
        try:
            obj = yaml.safe_load(txt)  # type: ignore[attr-defined]
        except Exception:
            return None
        return self._coerce_output_object(obj)

    def _coerce_output_object(self, obj: Any) -> Optional[Tuple[List[str], bool]]:
        if not isinstance(obj, dict):
            return None

        cats = (
            obj.get("Categories")
            or obj.get("categories")
            or obj.get("EntityCategories")
            or obj.get("entity_categories")
            or []
        )

        if isinstance(cats, str):
            cats_list = [cats]
        elif isinstance(cats, list):
            cats_list = [str(x) for x in cats if str(x).strip()]
        else:
            cats_list = []

        req = obj.get("RequiresContext", None)
        if req is None:
            req = obj.get("requires_context", None)
        if req is None:
            req = obj.get("requiresContext", False)

        if isinstance(req, str):
            req_bool = req.strip().lower() in {"1", "true", "yes", "on"}
        else:
            req_bool = bool(req)

        return cats_list, req_bool

    @staticmethod
    def _regex_parse_categories(txt: str) -> List[str]:
        # Inline: Categories: [structure, selection]
        m_inline = re.search(r"(?im)^\s*categories\s*:\s*\[(.*?)\]\s*$", txt)
        if m_inline:
            raw_items = m_inline.group(1)
            return [x.strip().strip("'\"") for x in raw_items.split(",") if x.strip()]

        out: List[str] = []
        lines = txt.splitlines()
        in_cats = False
        for ln in lines:
            if re.match(r"(?i)^\s*categories\s*:\s*$", ln):
                in_cats = True
                continue
            if in_cats:
                m = re.match(r"^\s*-\s*(.+?)\s*$", ln)
                if m:
                    out.append(m.group(1).strip().strip("'\""))
                    continue
                # Stop on next top-level key
                if re.match(r"^\s*[A-Za-z_]+\s*:\s*", ln):
                    break
        return out

    @staticmethod
    def _regex_parse_requires_context(txt: str) -> bool:
        m = re.search(r"(?im)^\s*requirescontext\s*:\s*(.+?)\s*$", txt)
        if not m:
            m = re.search(r"(?im)^\s*requires_context\s*:\s*(.+?)\s*$", txt)
        if not m:
            return False
        v = m.group(1).strip().strip("'\"").lower()
        return v in {"1", "true", "yes", "on"}
