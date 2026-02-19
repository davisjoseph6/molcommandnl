#!/usr/bin/env python3
"""
molcommandnl.script.mol_repl

Interactive REPL: NL -> (optional LLM context gate + retrieval) -> DSL -> normalize -> validate -> execute.

What’s new (context gate + history, policy-driven):
- Loads config/context_policy.yaml (or --policy) to decide:
  1) When to include scene context (LLM classifier + threshold/complexity bias)
  2) How to retrieve context (recent turns + top-k relevance + optional vector store)
  3) Where to persist history (JSONL)

Core idea:
- No hardcoding “apply context” logic in the transpiler: the LLM decides RequiresContext,
  and the policy biases that decision as the scene grows.

Option A behavior (DSL path):
- Use unitymol_copilot.dsl_normalizer.normalize_dsl() in DEV mode to fix common
  "LLM-ish" DSL formatting issues centrally (parenthesized form, space-args form,
  missing commas, double quotes, alias verbs, etc).
- Keep only REPL-specific stateful rewrite:
    sel="all" -> sel="<last_all_sel>"
  because the normalizer intentionally defaults to sel="all" when inferring targets,
  and only the REPL knows what the last focused structure is.

DEV-LOOSE behavior (MOLCOMMANDNL_DEV_LOOSE=1):
- Never hard-stop a turn due to validator errors.
- Salvage by validating each line independently, dropping invalid ones.
- Avoid "No selection named ..." by:
    * auto-prepending add_structure(PDBID="<code>") when the user *explicitly*
      mentioned <code> in the NL query and we reference all_<code> in DSL.
    * dropping hide/show/color lines that refer to selections we don't know exist.

Option 2 (DEMO RAW):
- If --demo-raw is set, bypass retrieval + DSL normalization/grammar/validation entirely.
- Directly execute UnityMolX scripting commands over ZMQ.

Notes on Chroma logging:
- We must set telemetry env vars and logger levels BEFORE importing anything that
  might import/initialize chromadb.
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
import argparse
import re
import logging
import importlib.util
from pathlib import Path
from importlib import import_module
from typing import Any, Dict, List, Optional, Set


# -----------------------------------------------------------------------------
# Telemetry + logging silencing (must be BEFORE chromadb import/initialization)
# -----------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma expects "False"/"True"
os.environ.setdefault("CHROMADB_TELEMETRY", "0")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

for name in (
    "chromadb",
    "chromadb.api.segment",
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry",
):
    lg = logging.getLogger(name)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(HERE)   # semantic_interpreter.py
sys.path.append(ROOT)   # repo root

# --- interpreter + LLM shim ---
from semantic_interpreter import SemanticInterpreter, LLMClient  # noqa: E402

# --- unitymol_copilot tools (validate/execute + DEV normalizer) ---
UNITYMOL_COPILOT_PATH = (
    os.environ.get("UNITYMOL_COPILOT_PATH")
    or os.path.expanduser("~/unitymol_copilot")
)
if UNITYMOL_COPILOT_PATH not in sys.path:
    sys.path.append(UNITYMOL_COPILOT_PATH)

from mcp_server import validate_dsl, execute_dsl  # noqa: E402

try:
    # Available in unitymol_copilot/
    from dsl_normalizer import normalize_dsl  # noqa: E402
except Exception:
    normalize_dsl = None

# --- context gate + history (policy-driven) ---
try:
    from molcommand.context_gate import ContextGate  # noqa: E402
    from molcommand.context_history import ContextHistory  # noqa: E402
except Exception:
    ContextGate = None
    ContextHistory = None


# =============================================================================
# Policy loading utilities
# =============================================================================
def _try_load_yaml(path: str) -> Optional[Dict[str, Any]]:
    """Load YAML if possible; return None on failure."""
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_policy(policy_path: str) -> Dict[str, Any]:
    """Load context policy YAML; return {} if missing/unreadable."""
    data = _try_load_yaml(policy_path)
    return data or {}


def _policy_get(d: Dict[str, Any], keys: str, default: Any) -> Any:
    """Dotted-path read from a dict."""
    cur: Any = d
    for part in keys.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# =============================================================================
# Optional vector store for history retrieval (policy-enabled)
# =============================================================================
class HistoryVectorStore:
    """
    Optional Chroma-backed vector store for scene history retrieval.

    This is best-effort and will gracefully disable itself if:
    - chromadb is not installed
    - an embedding function is not available
    - directory/collection cannot be created

    It stores "turn documents" as:
        User: <utterance>\nExec: <dsl>
    """

    def __init__(self, chroma_dir: str, collection: str) -> None:
        self.enabled = False
        self._client = None
        self._collection = None

        try:
            import chromadb  # type: ignore
        except Exception:
            return

        embed_fn = self._try_get_embedding_function()
        if embed_fn is None:
            return

        try:
            self._client = chromadb.PersistentClient(path=chroma_dir)
            try:
                self._collection = self._client.get_or_create_collection(
                    name=collection,
                    embedding_function=embed_fn,
                )
            except TypeError:
                self._collection = self._client.get_or_create_collection(name=collection)
                self._collection._embedding_function = embed_fn  # type: ignore[attr-defined]
            self.enabled = True
        except Exception:
            self.enabled = False

    @staticmethod
    def _try_get_embedding_function():
        """
        Try to locate an embedding function in the repo, without hardcoding dependencies.

        Priority:
        - script/get_embedding_function.py (used elsewhere in your repo)
        """
        candidates = [
            os.path.join(ROOT, "script", "get_embedding_function.py"),
            os.path.join(ROOT, "get_embedding_function.py"),
        ]

        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                spec = importlib.util.spec_from_file_location("get_embedding_function", p)
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                fn = getattr(mod, "get_embedding_function", None)
                if callable(fn):
                    return fn()
            except Exception:
                continue

        return None

    def upsert_turn(self, turn_id: int, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._collection is None:
            return
        try:
            self._collection.upsert(
                ids=[f"t{turn_id}"],
                documents=[text],
                metadatas=[meta or {}],
            )
        except Exception:
            return

    def query_turn_ids(self, query: str, top_k: int) -> List[int]:
        if not self.enabled or self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[query], n_results=top_k)
            ids = (res.get("ids") or [[]])[0]
            out: List[int] = []
            for rid in ids:
                m = re.match(r"^t(\d+)$", str(rid))
                if m:
                    out.append(int(m.group(1)))
            return out
        except Exception:
            return []


# =============================================================================
# DEMO RAW MODE (Option 2): UnityMol direct execution bypassing all DSL parsing
# =============================================================================

DEMO_STATE = {
    "known_selections": set(),
    "pdb_to_sel": {},
    "last_sel": "all",
    "last_pdb": None,
}


def _demo_load_unitymol():
    """Load UnityMolZMQ bridge from the user's UnityMol-ScriptCollection path."""
    p = Path.home() / "UnityMol-ScriptCollection/zmq/unitymol_zmq.py"
    if not p.exists():
        raise FileNotFoundError(
            f"UnityMol bridge not found at: {p}\n"
            "Expected UnityMol-ScriptCollection installed in your home directory."
        )

    spec = importlib.util.spec_from_file_location("unitymol_zmq", p)
    um = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(um)

    u = um.UnityMolZMQ(host="localhost", port=5555)
    ok = u.connect()
    if not ok:
        raise RuntimeError("Could not connect to UnityMolZMQ at tcp://localhost:5555")
    return u


def _demo_send_script(unitymol, script: str):
    """Send a raw UnityMol scripting command string to the ZMQ bridge."""
    candidates = (
        "sendCommand",
        "send_command",
        "send",
        "command",
        "exec",
        "execute",
        "run",
        "eval",
        "sendPython",
        "send_python",
    )
    for name in candidates:
        fn = getattr(unitymol, name, None)
        if callable(fn):
            return fn(script)

    raise AttributeError(
        "UnityMolZMQ wrapper has no compatible send method "
        f"(tried: {', '.join(candidates)})."
    )


def _demo_bool(v: bool) -> str:
    """UnityMol scripting uses capitalized True/False (Python-style)."""
    return "True" if v else "False"


def _demo_fetch(unitymol, pdbid: str, use_mmcif: bool = True):
    """Fetch a PDB from the internet via UnityMol."""
    for name in ("fetch", "fetchPDB", "fetch_pdb", "Fetch"):
        fn = getattr(unitymol, name, None)
        if callable(fn):
            try:
                return fn(pdbid, use_mmcif)
            except TypeError:
                return fn(pdbid)

    return _demo_send_script(unitymol, f'fetch("{pdbid}", {_demo_bool(use_mmcif)})')


def _demo_select(unitymol, query: str, alias: str):
    """Create/alias a selection in UnityMol."""
    fn = getattr(unitymol, "select", None)
    if callable(fn):
        try:
            return fn(query, alias, True, False, True)
        except TypeError:
            return fn(query, alias)

    return _demo_send_script(unitymol, f'select("{query}", "{alias}", True, False, True)')


def _demo_rep_variants(rep: str) -> List[str]:
    rep = (rep or "").lower()
    if rep == "c":
        return ["c", "cartoon"]
    if rep == "s":
        return ["s", "surface"]
    if rep == "l":
        return ["l", "lines"]
    if rep == "b":
        return ["b", "spheres", "balls"]
    return [rep]


def _demo_color_variants(color: str) -> List[str]:
    m = {
        "red": "#ff0000",
        "green": "#00ff00",
        "blue": "#0000ff",
        "yellow": "#ffff00",
        "orange": "#ff8800",
        "purple": "#8000ff",
        "cyan": "#00ffff",
        "magenta": "#ff00ff",
        "white": "#ffffff",
        "black": "#000000",
        "gray": "#808080",
    }
    c = (color or "").lower()
    out = [c] if c else []
    if c in m:
        out.append(m[c])
    return out


def _demo_show(unitymol, sel: str, rep: str):
    fn = getattr(unitymol, "showSelection", None)
    last_err = None
    for r in _demo_rep_variants(rep):
        try:
            if callable(fn):
                return fn(sel, r)
            return _demo_send_script(unitymol, f'showSelection("{sel}", "{r}")')
        except Exception as e:
            last_err = e
    return {"success": False, "stdout": f"showSelection failed ({last_err})"}


def _demo_hide(unitymol, sel: str, rep: Optional[str] = None):
    fn = getattr(unitymol, "hideSelection", None)

    try:
        if callable(fn):
            return fn(sel)
        return _demo_send_script(unitymol, f'hideSelection("{sel}")')
    except Exception:
        pass

    if rep is not None:
        last_err = None
        for r in _demo_rep_variants(rep):
            try:
                if callable(fn):
                    return fn(sel, r)
                return _demo_send_script(unitymol, f'hideSelection("{sel}", "{r}")')
            except Exception as e:
                last_err = e
        return {"success": False, "stdout": f"hideSelection(rep) failed ({last_err})"}

    return {"success": False, "stdout": "hideSelection failed"}


def _demo_color(unitymol, sel: str, rep: str, color: str):
    fn = getattr(unitymol, "colorSelection", None)
    last_err = None

    for r in _demo_rep_variants(rep):
        for c in _demo_color_variants(color):
            try:
                if callable(fn):
                    return fn(sel, r, c)
                return _demo_send_script(unitymol, f'colorSelection("{sel}", "{r}", "{c}")')
            except Exception as e:
                last_err = e

    return {"success": False, "stdout": f"colorSelection failed ({last_err})"}


def _demo_get_selections(unitymol) -> List[str]:
    resp = None
    fn = getattr(unitymol, "getSelectionListString", None)
    if callable(fn):
        try:
            resp = fn()
        except Exception:
            resp = None

    if resp is None:
        try:
            resp = _demo_send_script(unitymol, "getSelectionListString()")
        except Exception:
            return []

    if isinstance(resp, dict):
        s = f"{resp.get('result', '')} {resp.get('stdout', '')}"
    else:
        s = str(resp)

    s = re.sub(r"\[Log\]\s*", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    items = re.split(r"[\s,;]+", (s or "").strip())
    items = [x for x in items if x and re.match(r"^[A-Za-z0-9_]+$", x)]
    return items


def _demo_find_pdbid(text: str) -> Optional[str]:
    m = re.search(r"\b([0-9][a-zA-Z0-9]{3})\b", text or "")
    return m.group(1).lower() if m else None


def _demo_rep_code(text: str) -> str:
    t = (text or "").lower()
    if "cartoon" in t or "ribbon" in t:
        return "c"
    if "surface" in t or "surfaces" in t:
        return "s"
    if "lines" in t or "wire" in t or "wireframe" in t:
        return "l"
    if "sphere" in t or "spheres" in t or "ball" in t or "balls" in t:
        return "b"
    return "c"


def _demo_color_name(text: str) -> Optional[str]:
    colors = [
        "red", "blue", "green", "yellow", "orange",
        "purple", "cyan", "magenta", "white", "black",
        "grey", "gray",
    ]
    t = (text or "").lower()
    for c in colors:
        if re.search(rf"\b{c}\b", t):
            return "gray" if c == "grey" else c
    return None


def _demo_target_selection(text: str, pdbid: Optional[str] = None) -> Optional[str]:
    m = re.search(r"\b(all_[a-zA-Z0-9_]+)\b", text or "")
    if m:
        return m.group(1)
    if pdbid:
        return f"all_{pdbid}"
    return None


def _demo_resolve_sel(pdbid: Optional[str], sel: Optional[str]) -> str:
    if sel:
        DEMO_STATE["last_sel"] = sel
        return sel

    if pdbid and pdbid in DEMO_STATE["pdb_to_sel"]:
        DEMO_STATE["last_sel"] = DEMO_STATE["pdb_to_sel"][pdbid]
        return DEMO_STATE["last_sel"]

    last_sel = DEMO_STATE.get("last_sel") or "all"
    return last_sel


def _demo_register_loaded(pdbid: str, alias: str) -> None:
    DEMO_STATE["known_selections"].add(alias)
    DEMO_STATE["pdb_to_sel"][pdbid] = alias
    DEMO_STATE["last_sel"] = alias
    DEMO_STATE["last_pdb"] = pdbid


def _demo_should_skip_fetch(pdbid: str, alias: str) -> bool:
    if alias in DEMO_STATE["known_selections"]:
        return True
    if DEMO_STATE["pdb_to_sel"].get(pdbid) == alias:
        return True
    return False


def _demo_exec(unitymol, user_line: str) -> None:
    """
    Execute a single demo command by directly calling UnityMolZMQ methods.
    """
    line = (user_line or "").strip()
    low = line.lower()

    if not line:
        return

    if low in {"help", "?"}:
        print("DEMO-RAW examples:")
        print("  load 1crn")
        print("  show 1crn cartoon")
        print("  color 1crn cartoon red")
        print("  hide 1crn")
        print("Also accepts explicit selections like: show all_1crn cartoon")
        print("Tip: 'state' prints local registry; 'selections' tries UnityMol listing.")
        return

    if low in {"state", "st"}:
        print("DEMO-RAW state:")
        print("  last_sel:", DEMO_STATE.get("last_sel"))
        print("  last_pdb:", DEMO_STATE.get("last_pdb"))
        print("  pdb_to_sel:", DEMO_STATE.get("pdb_to_sel"))
        print("  known_selections:", sorted(DEMO_STATE.get("known_selections", set())))
        return

    if low in {"selections", "ls"}:
        sels = _demo_get_selections(unitymol)
        print("DEMO-RAW selections (UnityMol-reported, may be empty):", sels)
        return

    if ";" in line:
        for part in [x.strip() for x in line.split(";") if x.strip()]:
            _demo_exec(unitymol, part)
        return

    pdbid = _demo_find_pdbid(line)
    rep = _demo_rep_code(line)
    color = _demo_color_name(line)
    sel = _demo_target_selection(line, pdbid=pdbid)

    if "load" in low or "fetch" in low:
        if not pdbid:
            print("DEMO-RAW: could not find a PDB id (expected like 1crn).")
            return

        alias = f"all_{pdbid}"

        if _demo_should_skip_fetch(pdbid, alias):
            print(f"DEMO-RAW: already loaded in this session (alias {alias}); skipping fetch.")
            DEMO_STATE["last_sel"] = alias
            DEMO_STATE["last_pdb"] = pdbid
            return

        print(f"DEMO-RAW EXEC: fetch('{pdbid}', use_mmcif=True)")
        print(_demo_fetch(unitymol, pdbid, True))

        print(f"DEMO-RAW EXEC: select('all', '{alias}', True, False, True)")
        print(_demo_select(unitymol, "all", alias))

        _demo_register_loaded(pdbid, alias)
        return

    target = _demo_resolve_sel(pdbid, sel)

    if pdbid and pdbid not in DEMO_STATE["pdb_to_sel"] and (
        ("show" in low) or ("hide" in low) or ("color" in low)
    ):
        print(f"DEMO-RAW: '{pdbid}' not loaded in this session. Run: load {pdbid}")
        return

    if "show" in low:
        print(f"DEMO-RAW EXEC: showSelection('{target}', '{rep}')")
        print(_demo_show(unitymol, target, rep))
        return

    if "color" in low or "paint" in low or "make" in low:
        if not color:
            print("DEMO-RAW: no color found (try 'red', 'blue', etc.).")
            return
        print(f"DEMO-RAW EXEC: colorSelection('{target}', '{rep}', '{color}')")
        print(_demo_color(unitymol, target, rep, color))
        return

    if "hide" in low:
        print(f"DEMO-RAW EXEC: hideSelection('{target}')")
        print(_demo_hide(unitymol, target, rep))
        return

    print("DEMO-RAW: command not recognized. Type 'help'.")


# =============================================================================
# Normal DSL-based REPL (Option A)
# =============================================================================
def load_mol_dsl():
    """Load the molcommand DSL object/factory from molcommand/dsl_definition.py."""
    mod = import_module("molcommand.dsl_definition")
    for name in (
        "build_dsl",
        "make_dsl",
        "create_dsl",
        "load_dsl",
        "MolCommandDSL",
        "DSL",
        "dsl",
    ):
        if hasattr(mod, name):
            obj = getattr(mod, name)
            return obj() if callable(obj) else obj
    raise RuntimeError("No DSL factory found in molcommand/dsl_definition.py")


def cleaned(s: str) -> str:
    """Trim whitespace safely."""
    return (s or "").strip()


def _state_file(chroma_dir: str) -> Path:
    """Determine where to store REPL state."""
    override = os.environ.get("MOLCOMMANDNL_REPL_STATE_FILE")
    if override:
        return Path(override).expanduser()
    return Path(chroma_dir).expanduser() / "repl_state.json"


def load_repl_state(chroma_dir: str) -> dict:
    """Load persisted REPL state (best-effort)."""
    path = _state_file(chroma_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_repl_state(chroma_dir: str, state: dict) -> None:
    """Persist REPL state (best-effort, never hard-fails)."""
    path = _state_file(chroma_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def nl_pre_rewrite(q: str) -> str:
    """Pre-rewrite a few common shorthand NL patterns into explicit phrasing."""
    t = cleaned(q)
    if not t:
        return t
    m = re.match(r"^(?:please\s+)?(?:load|open)\s+([0-9A-Za-z]{4})\s*$", t, flags=re.I)
    if m:
        return f"Load PDB ID {m.group(1)}"
    return t


def repair_sel_all_to_last(dsl_text: str, last_all_sel: str) -> str:
    """
    Rewrite sel="all" -> sel="<last_all_sel>" once we have a focused structure selection.
    """
    if not last_all_sel or last_all_sel == "all":
        return dsl_text

    dsl_text = re.sub(
        r'show\(\s*sel\s*=\s*"all"\s*,',
        f'show(sel="{last_all_sel}",',
        dsl_text,
    )
    dsl_text = re.sub(
        r'hide\(\s*sel\s*=\s*"all"\s*\)',
        f'hide(sel="{last_all_sel}")',
        dsl_text,
    )
    # FIX: do NOT inject an extra "target=" (it corrupts the DSL)
    dsl_text = re.sub(
        r'color_by_chain\(\s*sel\s*=\s*"all"\s*,',
        f'color_by_chain(sel="{last_all_sel}",',
        dsl_text,
    )
    return dsl_text


def repair_missing_sel_args(dsl_text: str, last_all_sel: str) -> str:
    """If show(rep=...) or color_by_chain(target=...) lacks sel=, inject sel="..."."""
    sel = last_all_sel if last_all_sel else "all"

    dsl_text = re.sub(
        r"\bshow\(\s*rep\s*=",
        f'show(sel="{sel}", rep=',
        dsl_text,
    )

    dsl_text = re.sub(
        r"\bcolor_by_chain\(\s*target\s*=",
        f'color_by_chain(sel="{sel}", target=',
        dsl_text,
    )

    return dsl_text


def repair_unknown_all_sel_to_last(
    nl_query: str,
    dsl_text: str,
    last_all_sel: str,
    known_sels: Set[str],
) -> str:
    """
    Real REPL fix:
    If the model copied an example selection like sel="all_3eam" that does not exist
    in this session, and the user did NOT explicitly mention that PDB code,
    rewrite it to the current focus (last_all_sel) to avoid:
        [Warning] No selection named 'all_3eam'
    """
    if (
        not last_all_sel
        or not isinstance(last_all_sel, str)
        or not last_all_sel.startswith("all_")
    ):
        return dsl_text

    mentioned = _mentioned_pdb_codes(nl_query)

    def _repl(m: re.Match) -> str:
        sel = m.group(1)  # e.g. all_3eam
        code = sel.split("_", 1)[1].lower()

        # Only rewrite if it's unknown AND user didn't mention it.
        if sel not in known_sels and code not in mentioned:
            return f'sel="{last_all_sel}"'
        return m.group(0)

    return re.sub(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"', _repl, dsl_text)


def maybe_focus_last_all_sel_from_program(dsl_prog: str) -> Optional[str]:
    """If a command explicitly references sel="all_XXXX", treat that as the new focus."""
    m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"\s*', dsl_prog)
    if m:
        return m.group(1)
    return None


def repair_common_arg_mistakes(dsl_text: str) -> str:
    """Fix common arg-name mistakes (safe no-op if not present)."""
    dsl_text = re.sub(r"\bshow\(\s*add\s*=", "show(sel=", dsl_text)
    dsl_text = re.sub(r"\bhide\(\s*add\s*=", "hide(sel=", dsl_text)
    dsl_text = re.sub(r"\bcolor_by_chain\(\s*add\s*=", "color_by_chain(sel=", dsl_text)
    return dsl_text


def _split_lines(dsl_text: str) -> List[str]:
    """Split a DSL program into non-empty, trimmed lines."""
    return [cleaned(x) for x in (dsl_text or "").splitlines() if cleaned(x)]


def _dedupe_preserve_order(lines: List[str]) -> List[str]:
    """Deduplicate lines while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for ln in lines:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return out


def _mentioned_pdb_codes(nl_query: str) -> Set[str]:
    """Return 4-char PDB-like tokens explicitly mentioned by the user in the NL query."""
    return {m.group(0).lower() for m in re.finditer(r"\b[0-9][A-Za-z0-9]{3}\b", nl_query or "")}


def _planned_add_structure_codes(lines: List[str]) -> Set[str]:
    """Return PDB codes already planned to be loaded in this DSL program."""
    out: Set[str] = set()
    for ln in lines:
        m = re.search(r'add_structure\(\s*PDBID\s*=\s*"([0-9A-Za-z]{4})"\s*\)', ln)
        if m:
            out.add(m.group(1).lower())
    return out


def _extract_all_sel_code(line: str) -> Optional[str]:
    """If a line references sel="all_<code>", return <code> (lowercase)."""
    m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"', line)
    if not m:
        return None
    sel = m.group(1)
    return sel.split("_", 1)[1].lower()


def ensure_user_mentioned_structures_loaded(
    nl_query: str,
    dsl_text: str,
    known_sels: Set[str],
) -> str:
    """
    If the DSL references all_<pdb> but that selection doesn't exist yet,
    prepend add_structure(PDBID="<pdb>") ONLY if the PDB code appeared in the NL query.
    """
    lines = _split_lines(dsl_text)
    mentioned = _mentioned_pdb_codes(nl_query)
    if not mentioned:
        return dsl_text

    planned = _planned_add_structure_codes(lines)

    to_add: List[str] = []
    for ln in lines:
        code = _extract_all_sel_code(ln)
        if not code:
            continue

        sel = f"all_{code}"
        if code in mentioned and sel not in known_sels and code not in planned:
            to_add.append(f'add_structure(PDBID="{code}")')
            planned.add(code)

    if not to_add:
        return dsl_text

    return "\n".join(to_add + lines)


def drop_unknown_selections_in_dev_loose(dsl_text: str, known_sels: Set[str]) -> str:
    """
    DEV-LOOSE safety: drop lines referencing selections we don't know exist
    (unless they will be created by add_structure in the same DSL program).
    """
    lines = _split_lines(dsl_text)
    planned = _planned_add_structure_codes(lines)

    kept: List[str] = []
    for ln in lines:
        m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"', ln)
        if not m:
            kept.append(ln)
            continue

        sel = m.group(1)
        code = sel.split("_", 1)[1].lower()
        if sel in known_sels or code in planned:
            kept.append(ln)
        else:
            print("DEV-LOOSE drop unknown selection:", sel)

    kept = _dedupe_preserve_order(kept)
    return "\n".join(kept)


def _scene_stats(update_count: int, known_sels: Set[str], id_map: Dict[str, str]) -> Dict[str, Any]:
    """Compute simple, policy-friendly scene stats for context gating."""
    structures = 0
    if isinstance(id_map, dict):
        structures = len(id_map)
    if structures == 0:
        structures = len([s for s in known_sels if isinstance(s, str) and s.startswith("all_")])

    selections = len(known_sels)
    return {
        "updates": int(update_count),
        "structures": int(structures),
        "selections": int(selections),
    }


def _format_context_turns(turns: List[Any], header: str, max_chars: int) -> str:
    """Format turns into a single context block with clipping."""
    if not turns:
        return ""

    lines: List[str] = []
    if header:
        lines.append(header)

    for t in turns:
        tid = getattr(t, "turn_id", None)
        utt = getattr(t, "utterance", "") or ""
        dsl = getattr(t, "dsl", None)

        prefix_u = f"User (t{tid})" if tid is not None else "User"
        lines.append(f"{prefix_u}: {utt}")
        if dsl:
            prefix_e = f"Exec (t{tid})" if tid is not None else "Exec"
            lines.append(f"{prefix_e}: {dsl}")

    block = "\n".join(lines).strip()
    if len(block) <= max_chars:
        return block
    if max_chars <= 3:
        return block[:max_chars]
    return block[: max_chars - 3] + "..."


def _select_context_turns(
    history: Any,
    query: str,
    recent_turns: int,
    top_k: int,
    vector_store: Optional[HistoryVectorStore],
) -> List[Any]:
    """
    Choose context turns as union of:
      - most recent N turns
      - top-k relevant (lexical/hybrid via ContextHistory.retrieve_relevant)
      - top-k via vector store (optional)
    Deduplicate by turn_id, return most recent first.
    """
    chosen: Dict[int, Any] = {}

    turns = getattr(history, "_turns", [])
    if isinstance(turns, list) and recent_turns > 0:
        for t in turns[-recent_turns:]:
            tid = getattr(t, "turn_id", None)
            if isinstance(tid, int):
                chosen[tid] = t

    try:
        rel = history.retrieve_relevant(query, top_k=top_k)
        for t in rel:
            tid = getattr(t, "turn_id", None)
            if isinstance(tid, int):
                chosen[tid] = t
    except Exception:
        pass

    if vector_store and vector_store.enabled:
        ids = vector_store.query_turn_ids(query, top_k=top_k)
        for tid in ids:
            try:
                t = history._find_turn(tid)  # type: ignore[attr-defined]
                chosen[tid] = t
            except Exception:
                continue

    out = list(chosen.values())
    out.sort(key=lambda t: getattr(t, "turn_id", 0), reverse=True)
    return out


async def run_repl(entity_hint=None, with_context=False, demo_raw=False, policy_path: Optional[str] = None):
    # --- Option 2: DEMO RAW path ---
    if demo_raw:
        unitymol = _demo_load_unitymol()
        print("molREPL (DEMO-RAW) — bypassing DSL/validator; type 'help' for examples; 'quit' to exit.")
        while True:
            try:
                q = input("DEMO> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if q.lower() in ("quit", "exit"):
                break
            if not q:
                continue

            _demo_exec(unitymol, q)
        return

    # --- Normal DSL-based path ---
    chroma_dir = (
        os.environ.get("MOLCOMMANDNL_CHROMA_DIR")
        or os.environ.get("CHROMA_PATH")
        or os.path.expanduser("~/.cache/molcommandnl/chroma")
    )

    policy_path = policy_path or os.path.join(ROOT, "config", "context_policy.yaml")
    policy = load_policy(policy_path)

    llm = LLMClient()

    gate = None
    if policy and ContextGate is not None:
        try:
            gate = ContextGate(policy, llm)
        except Exception:
            gate = None

    history = None
    if ContextHistory is not None:
        try:
            jsonl_path = _policy_get(policy, "context_retrieval.history_store.jsonl_path", None)
            hist_cfg = {
                "context": {
                    "history": {"persist_path": jsonl_path},
                    "gating": {
                        "update_threshold": int(_policy_get(policy, "context_gate.update_threshold", 0)),
                    },
                }
            }
            history = ContextHistory(config=hist_cfg)
        except Exception:
            history = None

    vec = None
    vec_enabled = bool(_policy_get(policy, "context_retrieval.vector_store.enabled", False))
    if vec_enabled:
        chroma_hist_dir = str(_policy_get(policy, "context_retrieval.vector_store.chroma_dir", "data/chroma_history"))
        collection = str(_policy_get(policy, "context_retrieval.vector_store.collection", "scene_history"))
        if not os.path.isabs(chroma_hist_dir):
            chroma_hist_dir = os.path.join(ROOT, chroma_hist_dir)
        vec = HistoryVectorStore(chroma_hist_dir, collection)

    retrieval_enabled = bool(_policy_get(policy, "context_retrieval.enabled", True))
    recent_turns = int(_policy_get(policy, "context_retrieval.recent_turns", 6))
    top_k = int(_policy_get(policy, "context_retrieval.top_k", 4))
    max_ctx_chars = int(_policy_get(policy, "context_retrieval.max_chars", 3500))

    print("molREPL — type natural language; 'quit' to exit.")
    print(f"[chroma] {chroma_dir}")
    if policy:
        print(f"[policy] {policy_path}")
        if ContextGate is None or gate is None:
            print("[policy] context_gate unavailable (missing deps/import error); continuing without gate.")
        if ContextHistory is None or history is None:
            print("[policy] context_history unavailable (missing deps/import error); continuing without story.")
        if vec_enabled:
            print(f"[policy] vector_store={'on' if (vec and vec.enabled) else 'off'}")

    if entity_hint:
        print(f"[hint] default entity: {entity_hint}")
    if with_context:
        print("[hint] using example context object (demo)")

    dsl_spec = load_mol_dsl()
    si = SemanticInterpreter(dsl_spec, llm)

    base_context = {"id": "scene1", "_type": "unitymol_scene"} if with_context else None

    id_map: Dict[str, str] = {}

    state = load_repl_state(chroma_dir)
    last_all_sel = state.get("last_all_sel", "all")
    known_sels = set(state.get("known_sels", []))
    if isinstance(last_all_sel, str) and last_all_sel.startswith("all_"):
        known_sels.add(last_all_sel)

    def prefer_object_id(dsl_text: str) -> str:
        def _r_id_unquoted(m):
            code = m.group(1).lower()
            obj = id_map.get(code)
            return f"select(id:{obj})" if obj else m.group(0)

        def _r_id_quoted(m):
            code = m.group(1).lower()
            obj = id_map.get(code)
            return f"select(id:{obj})" if obj else m.group(0)

        dsl_text = re.sub(r"select\(name:([0-9A-Za-z]{4})\)", _r_id_unquoted, dsl_text)
        dsl_text = re.sub(r'select\("name:([0-9A-Za-z]{4})"\)', _r_id_quoted, dsl_text)
        return dsl_text

    while True:
        try:
            q = input("NL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        q = nl_pre_rewrite(q)

        update_count = int(history.update_count()) if history is not None else 0
        stats = _scene_stats(update_count, known_sels, id_map)

        categories: List[str] = []
        requires_ctx = False

        if gate is not None:
            try:
                dec = gate.decide(q, stats)
                if os.environ.get("MOLCOMMANDNL_TRACE_CONTEXT") == "1":
                    print(f"[gate] requires_ctx={requires_ctx} categories={categories} stats={stats}")
                categories = list(dec.categories or [])
                requires_ctx = bool(dec.requires_context)
            except Exception:
                categories = []
                requires_ctx = False

        context_payload = dict(base_context or {})
        turn_context_block = ""

        if retrieval_enabled and requires_ctx and history is not None:
            chosen_turns = _select_context_turns(
                history=history,
                query=q,
                recent_turns=recent_turns,
                top_k=top_k,
                vector_store=vec,
            )
            header = "Context (recent & relevant):"
            turn_context_block = _format_context_turns(chosen_turns, header=header, max_chars=max_ctx_chars)

            context_payload.update({
                "scene_stats": stats,
                "known_selections": sorted(list(known_sels))[:200],
                "last_all_sel": last_all_sel,
                "history": turn_context_block,
            })

        context_for_interp = context_payload if context_payload else None

        turn_id = None
        if history is not None:
            try:
                turn_id = history.add_user(
                    q,
                    meta={
                        "scene_stats": stats,
                        "known_sels": sorted(list(known_sels))[:200],
                        "last_all_sel": last_all_sel,
                        "requires_ctx": requires_ctx,
                        "categories": categories,
                    },
                )
            except Exception:
                turn_id = None

        interp = si.interpret(q, entity_hint=entity_hint, context=context_for_interp)
        if os.environ.get("MOLCOMMANDNL_TRACE_CONTEXT") == "1":
            if context_for_interp is None:
                print("[ctx] NONE")
            else:
                print("[ctx] keys:", sorted(context_for_interp.keys()))
                h = context_for_interp.get("history")
                if isinstance(h, str) and h.strip():
                    print("[ctx] history:\n" + h)
                    print("[ctx] last_all_sel:", context_for_interp.get("last_all_sel"))

        dsl_prog = cleaned(
            getattr(interp, "cleaned_program", "")
            or interp.get("cleaned_dsl", "")
            or interp.get("dsl", "")
        )

        if not dsl_prog and interp.get("ast"):
            try:
                dsl_prog = cleaned(si.parser.unparse(interp["ast"]))
            except Exception:
                dsl_prog = ""

        if not dsl_prog:
            print("No DSL produced.")
            if history is not None and turn_id is not None:
                try:
                    history.add_execution(
                        turn_id,
                        dsl="",
                        categories=categories,
                        llm_requires_context=requires_ctx,
                        did_update_scene=False,
                        meta={"note": "no_dsl"},
                    )
                except Exception:
                    pass
            continue

        if normalize_dsl is not None:
            normed, info = normalize_dsl(dsl_prog)
            if info.get("changed") and os.environ.get("MOLCOMMANDNL_SHOW_NORM") == "1":
                print("[DEV-NORM]", json.dumps(info, indent=2))
            dsl_prog = normed

        dsl_prog = repair_common_arg_mistakes(dsl_prog)
        dsl_prog = repair_sel_all_to_last(dsl_prog, last_all_sel)
        dsl_prog = repair_missing_sel_args(dsl_prog, last_all_sel)

        # REAL FIX: if model outputs sel="all_3eam" etc, and it's unknown and user didn't say "3eam",
        # rewrite it to the last focused selection (so "show surface" acts on the last loaded object).
        dsl_prog = repair_unknown_all_sel_to_last(q, dsl_prog, last_all_sel, known_sels)

        if os.environ.get("MCL_PREFER_OBJECT_ID") == "1":
            dsl_prog = prefer_object_id(dsl_prog)

        dev_loose = os.environ.get("MOLCOMMANDNL_DEV_LOOSE") == "1"

        if dev_loose:
            dsl_prog = ensure_user_mentioned_structures_loaded(q, dsl_prog, known_sels)
            dsl_prog = drop_unknown_selections_in_dev_loose(dsl_prog, known_sels)

        print("DSL>", dsl_prog)

        v = await validate_dsl(dsl_prog)

        if not v.get("ok") and dev_loose:
            lines = _split_lines(dsl_prog)
            ok_lines: List[str] = []
            for ln in lines:
                vv = await validate_dsl(ln)
                if vv.get("ok"):
                    ok_lines.append(ln)
                else:
                    print("DEV-LOOSE drop:", ln)
                    print("  reason:", vv.get("errors"))

            ok_lines = _dedupe_preserve_order(ok_lines)
            dsl_prog = "\n".join(ok_lines)

            if not dsl_prog:
                print("DEV-LOOSE: nothing executable after salvage.")
                if history is not None and turn_id is not None:
                    try:
                        history.add_execution(
                            turn_id,
                            dsl="",
                            categories=categories,
                            llm_requires_context=requires_ctx,
                            did_update_scene=False,
                            meta={"note": "salvage_empty"},
                        )
                    except Exception:
                        pass
                continue

            print("DSL> (salvaged)\n" + dsl_prog)
            v = await validate_dsl(dsl_prog)

        if not v.get("ok"):
            print("Validator errors:", v.get("errors"))
            if v.get("dev"):
                print("[DEV]", json.dumps(v["dev"], indent=2))

            if history is not None and turn_id is not None:
                try:
                    history.add_execution(
                        turn_id,
                        dsl=dsl_prog,
                        categories=categories,
                        llm_requires_context=requires_ctx,
                        did_update_scene=False,
                        meta={"note": "validator_error", "errors": v.get("errors")},
                    )
                except Exception:
                    pass
            continue

        out = await execute_dsl(dsl_prog)
        print(json.dumps(out, indent=2))

        did_update_scene = False
        if history is not None:
            try:
                okish = False
                if isinstance(out, dict):
                    if out.get("ok") is True or out.get("success") is True:
                        okish = True
                    if out.get("error"):
                        okish = False
                did_update_scene = bool(okish) and bool(history.infer_scene_update_from_dsl(dsl_prog))
            except Exception:
                did_update_scene = False

        if history is not None and turn_id is not None:
            try:
                history.add_execution(
                    turn_id,
                    dsl=dsl_prog,
                    categories=categories,
                    llm_requires_context=requires_ctx,
                    did_update_scene=did_update_scene,
                    meta={"scene_stats": stats},
                )
            except Exception:
                pass

            if vec is not None and vec.enabled:
                try:
                    doc = f"User: {q}\nExec: {dsl_prog}"
                    vec.upsert_turn(
                        turn_id=turn_id,
                        text=doc,
                        meta={"turn_id": turn_id, "updates": int(stats.get("updates", 0))},
                    )
                except Exception:
                    pass

        m_pdb = re.search(
            r'add_structure\(\s*PDBID\s*=\s*"([0-9A-Za-z]{4})"\s*\)',
            dsl_prog,
        )
        if m_pdb and isinstance(out, dict):
            obj = out.get("result")
            if isinstance(obj, str):
                id_map[m_pdb.group(1).lower()] = obj
                if obj.startswith("all_"):
                    last_all_sel = obj
                    known_sels.add(obj)
                    save_repl_state(
                        chroma_dir,
                        {"last_all_sel": last_all_sel, "known_sels": sorted(known_sels)},
                    )

        focused = maybe_focus_last_all_sel_from_program(dsl_prog)
        if focused and focused != last_all_sel:
            last_all_sel = focused
            known_sels.add(focused)
            save_repl_state(
                chroma_dir,
                {"last_all_sel": last_all_sel, "known_sels": sorted(known_sels)},
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--entity",
        default=None,
        help="Optional entity hint (e.g. 'structure', 'selection', 'representation').",
    )
    ap.add_argument(
        "--with-context",
        action="store_true",
        help="Use a tiny demo context object (legacy demo).",
    )
    ap.add_argument(
        "--dev",
        action="store_true",
        help="Enable DEV-mode DSL normalization (sets MOLCOMMANDNL_DEV=1).",
    )
    ap.add_argument(
        "--dev-loose",
        action="store_true",
        help="Enable loose DEV normalization (sets MOLCOMMANDNL_DEV_LOOSE=1; implies --dev).",
    )
    ap.add_argument(
        "--show-norm",
        action="store_true",
        help="Print DEV normalizer steps when it changes the DSL (sets MOLCOMMANDNL_SHOW_NORM=1).",
    )
    ap.add_argument(
        "--demo-raw",
        action="store_true",
        help="DEMO MODE: bypass retrieval + DSL normalization/grammar/validation; execute UnityMolX directly over ZMQ.",
    )
    ap.add_argument(
        "--policy",
        default=None,
        help="Path to context_policy.yaml (default: config/context_policy.yaml).",
    )

    args = ap.parse_args()

    if args.dev or args.dev_loose:
        os.environ["MOLCOMMANDNL_DEV"] = "1"
    if args.dev_loose:
        os.environ["MOLCOMMANDNL_DEV_LOOSE"] = "1"
    if args.show_norm:
        os.environ["MOLCOMMANDNL_SHOW_NORM"] = "1"

    asyncio.run(
        run_repl(
            entity_hint=args.entity,
            with_context=args.with_context,
            demo_raw=args.demo_raw,
            policy_path=args.policy,
        )
    )


if __name__ == "__main__":
    main()
