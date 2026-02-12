#!/usr/bin/env python3
"""
molcommandnl.script.mol_repl

Interactive REPL: NL -> DSL -> validate -> execute via unitymol_copilot.

Option A behavior:
- Use unitymol_copilot.dsl_normalizer.normalize_dsl() in DEV mode to fix common
  "LLM-ish" DSL formatting issues centrally (parenthesized form, space-args form,
  missing commas, double quotes, alias verbs like add_coloring/colorByChain, etc).
- Keep only the REPL-specific stateful rewrite:
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

Extra DEV robustness:
- If the LLM outputs show(rep=...) or color_by_chain(target=...) without sel=...,
  auto-inject sel="<last_all_sel>" (or "all" if unknown) BEFORE validation.

Other:
- Persist last_all_sel and known selections across REPL runs (repl_state.json).
- Pre-rewrite common shorthand NL like "load 1crn" -> "Load PDB ID 1crn"

Option 2 (DEMO RAW):
- If --demo-raw is set, bypass retrieval + DSL normalization/grammar/validation entirely.
- Directly execute UnityMolX scripting commands over ZMQ (fetch/select/show/color/hide).
- This mode is intended for demos when DSL rules/validators are getting in the way.

IMPORTANT DEMO-RAW FIX:
- Do NOT depend on getSelectionListString() (it often returns empty on some builds).
- Maintain a dynamic in-session registry of selections/objects we created (DEMO_STATE).
- Resolve targets using:
    explicit sel (all_XXXX) > pdbid mapping > last_sel > "all"
  so the demo stays dynamic without hardcoding a fixed vocabulary.

Notes on Chroma logging:
- We must set telemetry env vars and logger levels BEFORE importing anything that
  might import/initialize chromadb (e.g., semantic_interpreter / unitymol_copilot).
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


# =============================================================================
# DEMO RAW MODE (Option 2): UnityMol direct execution bypassing all DSL parsing
# =============================================================================

# Dynamic, in-session registry of what we created / focused.
# This is NOT a fixed vocabulary. It is runtime state.
DEMO_STATE = {
    "known_selections": set(),   # selections we created or saw
    "pdb_to_sel": {},            # "1crn" -> "all_1crn"
    "last_sel": "all",           # current focus
    "last_pdb": None,            # last pdbid mentioned/loaded
}


def _demo_load_unitymol():
    """
    Load UnityMolZMQ bridge from the user's UnityMol-ScriptCollection path.

    This keeps demo mode independent of unitymol_copilot execution/validation
    and avoids any DSL/grammar/validator rules.
    """
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
    """
    Send a raw UnityMol scripting command string to the ZMQ bridge.

    Different UnityMolZMQ wrappers expose different method names. We try a set of
    common ones.
    """
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
    """
    Fetch a PDB from the internet via UnityMol.

    If the wrapper does not expose .fetch(), fall back to sending:
        fetch("<pdbid>", True/False)
    """
    for name in ("fetch", "fetchPDB", "fetch_pdb", "Fetch"):
        fn = getattr(unitymol, name, None)
        if callable(fn):
            try:
                return fn(pdbid, use_mmcif)
            except TypeError:
                return fn(pdbid)

    return _demo_send_script(unitymol, f'fetch("{pdbid}", {_demo_bool(use_mmcif)})')


def _demo_select(unitymol, query: str, alias: str):
    """
    Create/alias a selection in UnityMol.

    Preferred wrapper call:
        select(query, alias, True, False, True)

    Fallback scripting string:
        select("<query>", "<alias>", True, False, True)
    """
    fn = getattr(unitymol, "select", None)
    if callable(fn):
        try:
            return fn(query, alias, True, False, True)
        except TypeError:
            return fn(query, alias)

    return _demo_send_script(unitymol, f'select("{query}", "{alias}", True, False, True)')


def _demo_rep_variants(rep: str) -> list[str]:
    """
    Try multiple representation identifiers. UnityMol builds differ:
    some accept "c", some accept "cartoon", etc.
    """
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


def _demo_color_variants(color: str) -> list[str]:
    """
    Try name + hex fallback (some UnityMol builds accept hex colors).
    """
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
    """
    Show selection with representation. Tries rep variants for robustness.
    """
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


def _demo_hide(unitymol, sel: str, rep: str | None = None):
    """
    Hide selection. Some builds support hideSelection(sel, rep).
    Try both forms.
    """
    fn = getattr(unitymol, "hideSelection", None)

    # 1) Try hideSelection(sel)
    try:
        if callable(fn):
            return fn(sel)
        return _demo_send_script(unitymol, f'hideSelection("{sel}")')
    except Exception:
        pass

    # 2) Try hideSelection(sel, rep) variants
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
    """
    Color selection for a representation. Tries rep variants and color variants.
    """
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


def _demo_get_selections(unitymol) -> list[str]:
    """
    Best-effort selection listing.

    WARNING: On some UnityMol builds/wrappers, getSelectionListString()
    returns empty or log-only output. We keep this for debugging, but DEMO-RAW
    functionality must NOT depend on it.
    """
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


def _demo_find_pdbid(text: str) -> str | None:
    """Find a PDB id like 1crn / 1kx2 in free text."""
    m = re.search(r"\b([0-9][a-zA-Z0-9]{3})\b", text or "")
    return m.group(1).lower() if m else None


def _demo_rep_code(text: str) -> str:
    """
    Map rep keywords to UnityMol rep codes used by showSelection/colorSelection.

    Common UnityMolX scripting codes used in your logs:
      c = cartoon
      s = surface
      l = lines
      b = spheres/balls (if supported)
    """
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


def _demo_color_name(text: str) -> str | None:
    """Extract a basic color name from free text."""
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


def _demo_target_selection(text: str, pdbid: str | None = None) -> str | None:
    """
    Determine selection name to use:
    - If user types all_1crn explicitly, use that.
    - Else if pdbid is known, use all_<pdbid>.
    """
    m = re.search(r"\b(all_[a-zA-Z0-9_]+)\b", text or "")
    if m:
        return m.group(1)
    if pdbid:
        return f"all_{pdbid}"
    return None


def _demo_resolve_sel(pdbid: str | None, sel: str | None) -> str:
    """
    Resolve the best target selection dynamically without relying on UnityMol selection listing.

    Priority:
      1) explicit sel (e.g. all_1crn)
      2) pdbid -> known mapping
      3) last focused selection
      4) fallback "all"
    """
    if sel:
        DEMO_STATE["last_sel"] = sel
        return sel

    if pdbid and pdbid in DEMO_STATE["pdb_to_sel"]:
        DEMO_STATE["last_sel"] = DEMO_STATE["pdb_to_sel"][pdbid]
        return DEMO_STATE["last_sel"]

    last_sel = DEMO_STATE.get("last_sel") or "all"
    return last_sel


def _demo_register_loaded(pdbid: str, alias: str) -> None:
    """Update dynamic demo registry after a successful load/select."""
    DEMO_STATE["known_selections"].add(alias)
    DEMO_STATE["pdb_to_sel"][pdbid] = alias
    DEMO_STATE["last_sel"] = alias
    DEMO_STATE["last_pdb"] = pdbid


def _demo_should_skip_fetch(pdbid: str, alias: str) -> bool:
    """
    Decide if we should skip fetch for this pdbid.

    Since UnityMol selection listing may be unreliable, we prefer our local registry:
    - If we've already registered alias in this session, skip.
    """
    if alias in DEMO_STATE["known_selections"]:
        return True
    if DEMO_STATE["pdb_to_sel"].get(pdbid) == alias:
        return True
    return False


def _demo_exec(unitymol, user_line: str) -> None:
    """
    Execute a single demo command by directly calling UnityMolZMQ methods.

    Supported intents (very lightweight heuristics):
      - load/fetch <pdbid>
      - show <pdbid> <rep>
      - color <pdbid> <rep> <color>
      - hide <pdbid>  (or hide all_<pdbid>)

    IMPORTANT:
    - We do NOT auto-load on show/color/hide (avoid 1crn_2, 1crn_3, ... confusion).
    - We DO resolve targets dynamically using DEMO_STATE even if selection listing is broken.
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
        print("Tip: 'state' prints the local demo registry; 'selections' tries UnityMol listing.")
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

    # Allow multiple commands separated by ';'
    if ";" in line:
        for part in [x.strip() for x in line.split(";") if x.strip()]:
            _demo_exec(unitymol, part)
        return

    pdbid = _demo_find_pdbid(line)
    rep = _demo_rep_code(line)
    color = _demo_color_name(line)
    sel = _demo_target_selection(line, pdbid=pdbid)

    # --- LOAD ---
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

    # Resolve target for non-load commands
    target = _demo_resolve_sel(pdbid, sel)

    # If user referenced a pdbid but never loaded it, be explicit
    if pdbid and pdbid not in DEMO_STATE["pdb_to_sel"] and (("show" in low) or ("hide" in low) or ("color" in low)):
        print(f"DEMO-RAW: '{pdbid}' not loaded in this session. Run: load {pdbid}")
        return

    # --- SHOW ---
    if "show" in low:
        print(f"DEMO-RAW EXEC: showSelection('{target}', '{rep}')")
        print(_demo_show(unitymol, target, rep))
        return

    # --- COLOR ---
    if "color" in low or "paint" in low or "make" in low:
        if not color:
            print("DEMO-RAW: no color found (try 'red', 'blue', etc.).")
            return
        print(f"DEMO-RAW EXEC: colorSelection('{target}', '{rep}', '{color}')")
        print(_demo_color(unitymol, target, rep, color))
        return

    # --- HIDE ---
    if "hide" in low:
        print(f"DEMO-RAW EXEC: hideSelection('{target}')")
        print(_demo_hide(unitymol, target, rep))
        return

    print("DEMO-RAW: command not recognized. Type 'help'.")


# =============================================================================
# Normal DSL-based REPL (existing behavior)
# =============================================================================
def load_mol_dsl():
    """
    Load the molcommand DSL object/factory from molcommand/dsl_definition.py.
    Tries several common names so we’re robust to refactors.
    """
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
    """
    Determine where to store REPL state.

    Priority:
      1) MOLCOMMANDNL_REPL_STATE_FILE (explicit)
      2) <chroma_dir>/repl_state.json
    """
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
    """
    Pre-rewrite a few common shorthand NL patterns into explicit, sample-friendly phrasing.
    """
    t = cleaned(q)
    if not t:
        return t

    # "load 1crn" / "open 3eam" -> "Load PDB ID 1crn"
    m = re.match(r"^(?:please\s+)?(?:load|open)\s+([0-9A-Za-z]{4})\s*$", t, flags=re.I)
    if m:
        return f"Load PDB ID {m.group(1)}"

    return t


def repair_sel_all_to_last(dsl_text: str, last_all_sel: str) -> str:
    """
    Rewrite sel="all" -> sel="<last_all_sel>" once we have a focused structure selection.

    We intentionally keep this REPL-local because only the REPL knows the current focus.
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
    dsl_text = re.sub(
        r'color_by_chain\(\s*sel\s*=\s*"all"\s*,',
        f'color_by_chain(sel="{last_all_sel}",',
        dsl_text,
    )
    return dsl_text


def repair_missing_sel_args(dsl_text: str, last_all_sel: str) -> str:
    """
    If the LLM outputs show(rep=...) or color_by_chain(target=...) without sel=,
    inject sel="<last_all_sel>" (or "all" if unknown) so the validator can accept it.
    """
    sel = last_all_sel if last_all_sel else "all"

    # show(rep="surface") -> show(sel="...", rep="surface")
    dsl_text = re.sub(
        r"\bshow\(\s*rep\s*=",
        f'show(sel="{sel}", rep=',
        dsl_text,
    )

    # color_by_chain(target="surface") -> color_by_chain(sel="...", target="surface")
    dsl_text = re.sub(
        r"\bcolor_by_chain\(\s*target\s*=",
        f'color_by_chain(sel="{sel}", target=',
        dsl_text,
    )

    return dsl_text


def maybe_focus_last_all_sel_from_program(dsl_prog: str) -> str | None:
    """
    If a command explicitly references sel="all_XXXX", treat that as the new focus.
    This makes follow-ups act on the last-mentioned structure, not just the last-loaded one.
    """
    m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"\s*', dsl_prog)
    if m:
        return m.group(1)
    return None


def repair_common_arg_mistakes(dsl_text: str) -> str:
    """
    Fix common DEV-normalizer argument name mistakes (e.g. show(add=...) should be show(sel=...)).
    Safe no-op if not present.
    """
    dsl_text = re.sub(r"\bshow\(\s*add\s*=", "show(sel=", dsl_text)
    dsl_text = re.sub(r"\bhide\(\s*add\s*=", "hide(sel=", dsl_text)
    dsl_text = re.sub(r"\bcolor_by_chain\(\s*add\s*=", "color_by_chain(sel=", dsl_text)
    return dsl_text


def _split_lines(dsl_text: str) -> list[str]:
    """Split a DSL program into non-empty, trimmed lines."""
    return [cleaned(x) for x in (dsl_text or "").splitlines() if cleaned(x)]


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    """Deduplicate lines while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return out


def _mentioned_pdb_codes(nl_query: str) -> set[str]:
    """
    Return 4-char PDB-like tokens explicitly mentioned by the user in the NL query.
    Require first char to be a digit (PDB ID shape).
    """
    return {m.group(0).lower() for m in re.finditer(r"\b[0-9][A-Za-z0-9]{3}\b", nl_query or "")}


def _planned_add_structure_codes(lines: list[str]) -> set[str]:
    """Return PDB codes already planned to be loaded in this DSL program."""
    out: set[str] = set()
    for ln in lines:
        m = re.search(r'add_structure\(\s*PDBID\s*=\s*"([0-9A-Za-z]{4})"\s*\)', ln)
        if m:
            out.add(m.group(1).lower())
    return out


def _extract_all_sel_code(line: str) -> str | None:
    """If a line references sel="all_<code>", return <code> (lowercase)."""
    m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"', line)
    if not m:
        return None
    sel = m.group(1)
    return sel.split("_", 1)[1].lower()


def ensure_user_mentioned_structures_loaded(
    nl_query: str,
    dsl_text: str,
    known_sels: set[str],
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

    to_add: list[str] = []
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


def drop_unknown_selections_in_dev_loose(dsl_text: str, known_sels: set[str]) -> str:
    """
    DEV-LOOSE safety: drop lines referencing selections we don't know exist
    (unless they will be created by add_structure in the same DSL program).
    """
    lines = _split_lines(dsl_text)
    planned = _planned_add_structure_codes(lines)

    kept: list[str] = []
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


async def run_repl(entity_hint=None, with_context=False, demo_raw=False):
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

    print("molREPL — type natural language; 'quit' to exit.")
    print(f"[chroma] {chroma_dir}")
    if entity_hint:
        print(f"[hint] default entity: {entity_hint}")
    if with_context:
        print("[hint] using example context")

    dsl_spec = load_mol_dsl()
    si = SemanticInterpreter(dsl_spec, LLMClient())
    context = {"id": "scene1", "_type": "unitymol_scene"} if with_context else None

    id_map: dict[str, str] = {}

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

        interp = si.interpret(q, entity_hint=entity_hint, context=context)

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
            continue

        # --- Option A: normalize DEV "LLM-ish" DSL centrally ---
        if normalize_dsl is not None:
            normed, info = normalize_dsl(dsl_prog)
            if info.get("changed") and os.environ.get("MOLCOMMANDNL_SHOW_NORM") == "1":
                print("[DEV-NORM]", json.dumps(info, indent=2))
            dsl_prog = normed

        dsl_prog = repair_common_arg_mistakes(dsl_prog)
        dsl_prog = repair_sel_all_to_last(dsl_prog, last_all_sel)
        dsl_prog = repair_missing_sel_args(dsl_prog, last_all_sel)

        if os.environ.get("MCL_PREFER_OBJECT_ID") == "1":
            dsl_prog = prefer_object_id(dsl_prog)

        dev_loose = os.environ.get("MOLCOMMANDNL_DEV_LOOSE") == "1"

        if dev_loose:
            dsl_prog = ensure_user_mentioned_structures_loaded(q, dsl_prog, known_sels)
            dsl_prog = drop_unknown_selections_in_dev_loose(dsl_prog, known_sels)

        print("DSL>", dsl_prog)

        # Validate whole program first
        v = await validate_dsl(dsl_prog)

        # DEV-LOOSE: salvage by validating each line independently
        if not v.get("ok") and dev_loose:
            lines = _split_lines(dsl_prog)
            ok_lines: list[str] = []
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
                continue

            print("DSL> (salvaged)\n" + dsl_prog)
            v = await validate_dsl(dsl_prog)

        if not v.get("ok"):
            print("Validator errors:", v.get("errors"))
            if v.get("dev"):
                print("[DEV]", json.dumps(v["dev"], indent=2))
            continue

        out = await execute_dsl(dsl_prog)
        print(json.dumps(out, indent=2))

        # Update id_map and last_all_sel on add_structure
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

        # ALSO update focus when a command explicitly mentions sel="all_XXXX"
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
        help="Use a tiny demo context object.",
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
        )
    )


if __name__ == "__main__":
    main()

