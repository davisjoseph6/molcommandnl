#!/usr/bin/env python3
"""
molcommandnl.script.mol_repl

Interactive REPL: NL -> DSL -> validate -> execute via unitymol_copilot.
Includes small repairs for common LLM formatting slips.

Dev QoL:
- Repair common LLM formatting mistakes ((show ...) or space-arg forms).
- Repair color_by_chain(sel="all_cartoon") missing target.
- Rewrite generic sel="all" into the last loaded structure selection (all_<code>).
- Persist last_all_sel across REPL runs (repl_state.json) so structure -> representation
  works even when run in separate REPL processes.
"""

from pathlib import Path

import os
import sys
import asyncio
import json
import argparse
import re
from importlib import import_module

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(HERE)   # semantic_interpreter.py
sys.path.append(ROOT)   # repo root

# --- interpreter + LLM shim ---
from semantic_interpreter import SemanticInterpreter, LLMClient  # noqa: E402

# --- unitymol_copilot tools (validate/execute) ---
sys.path.append(os.path.expanduser("~/unitymol_copilot"))
from mcp_server import validate_dsl, execute_dsl  # noqa: E402


def load_mol_dsl():
    """
    Load your molcommand DSL object/factory from molcommand/dsl_definition.py.
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


def repair_llm_dsl(s: str) -> str:
    """
    Repair minor formatting mistakes in LLM output so it matches the strict DSL grammar.

    Fixes:
      1) (show sel="...", rep="...")  -> show(sel="...", rep="...")
      2) show sel="..." rep="..."     -> show(sel="...", rep="...")
    """
    s = cleaned(s)
    if not s:
        return s

    # Case 1: (show sel="...", rep="...")  -> show(sel="...", rep="...")
    m = re.match(r"^\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.*)\)\s*$", s, flags=re.S)
    if m:
        fn, inner = m.group(1), m.group(2).strip()
        # Insert commas between key/value pairs when the LLM uses spaces.
        inner = re.sub(r'"\s+([A-Za-z_][A-Za-z0-9_]*\s*=)', r'", \1', inner)
        return f"{fn}({inner})"

    # Case 2: show sel="..." rep="..." -> show(sel="...", rep="...")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", s, flags=re.S)
    if m and "(" not in s and "=" in s:
        fn, inner = m.group(1), m.group(2).strip()
        inner = re.sub(r'"\s+([A-Za-z_][A-Za-z0-9_]*\s*=)', r'", \1', inner)
        return f"{fn}({inner})"

    return s


def repair_color_by_chain_defaults(dsl_text: str, last_all_sel: str) -> str:
    """
    Repair common LLM slip where it emits only a sel, and encodes the rep in the sel:

      color_by_chain(sel="all_cartoon") -> color_by_chain(sel="<last_all_sel>", target="cartoon")
      color_by_chain(sel="cartoon")     -> color_by_chain(sel="<last_all_sel>", target="cartoon")
      color_by_chain(sel="all_lines")   -> ... target="lines"

    If we can't confidently infer the target, we leave it unchanged.
    """
    dsl_text = cleaned(dsl_text)
    m = re.fullmatch(r'color_by_chain\(\s*sel\s*=\s*"([^"]+)"\s*\)', dsl_text)
    if not m:
        return dsl_text

    raw_sel = m.group(1).strip().lower()
    reps = ("cartoon", "lines", "spheres", "surface")

    target = None
    if raw_sel in reps:
        target = raw_sel
    elif raw_sel.startswith("all_") and raw_sel[4:] in reps:
        target = raw_sel[4:]

    if not target:
        return dsl_text

    sel = last_all_sel or "all"
    return f'color_by_chain(sel="{sel}", target="{target}")'


def repair_sel_all_to_last(dsl_text: str, last_all_sel: str) -> str:
    """
    If DSL uses sel="all" (common in generic samples), rewrite it to the most
    recently created selection like all_1crn.
    """
    if not last_all_sel or last_all_sel == "all":
        return dsl_text

    # show(sel="all", rep="...")
    dsl_text = re.sub(
        r'show\(\s*sel\s*=\s*"all"\s*,',
        f'show(sel="{last_all_sel}",',
        dsl_text,
    )

    # hide(sel="all")
    dsl_text = re.sub(
        r'hide\(\s*sel\s*=\s*"all"\s*\)',
        f'hide(sel="{last_all_sel}")',
        dsl_text,
    )

    # color_by_chain(sel="all", target="...")
    dsl_text = re.sub(
        r'color_by_chain\(\s*sel\s*=\s*"all"\s*,',
        f'color_by_chain(sel="{last_all_sel}",',
        dsl_text,
    )

    return dsl_text


async def run_repl(entity_hint=None, with_context=False):
    # Only for operator visibility; SemanticInterpreter picks the actual path.
    chroma_dir = (
        os.environ.get("MOLCOMMANDNL_CHROMA_DIR")
        or os.environ.get("CHROMA_PATH")
        or os.path.expanduser("~/.cache/molcommandnl/chroma")
    )
    os.environ.setdefault("CHROMADB_TELEMETRY", "0")

    print("molREPL — type natural language; 'quit' to exit.")
    print(f"[chroma] {chroma_dir}")
    if entity_hint:
        print(f"[hint] default entity: {entity_hint}")
    if with_context:
        print("[hint] using example context")

    dsl_spec = load_mol_dsl()
    si = SemanticInterpreter(dsl_spec, LLMClient())
    context = {"id": "scene1", "_type": "unitymol_scene"} if with_context else None

    # Prefer selecting by concrete object id once we know it (optional via env)
    id_map: dict[str, str] = {}

    # Track "most recent structure selection" for generic ops.
    # Falls back to "all" (UnityMol keyword), until we load a structure and get all_<code>.
    state = load_repl_state(chroma_dir)
    last_all_sel = state.get("last_all_sel", "all")

    def prefer_object_id(dsl_text: str) -> str:
        # replace select(name:XXXX) or select("name:XXXX") with select(id:OBJID) if known
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

        # NL → DSL
        interp = si.interpret(q, entity_hint=entity_hint, context=context)

        # Prefer CLEANED DSL, fall back if needed
        dsl_prog = cleaned(
            getattr(interp, "cleaned_program", "")
            or interp.get("cleaned_dsl", "")
            or interp.get("dsl", "")
        )

        # Last resort: rebuild from AST
        if not dsl_prog and interp.get("ast"):
            try:
                dsl_prog = cleaned(si.parser.unparse(interp["ast"]))
            except Exception:
                dsl_prog = ""

        if not dsl_prog:
            print("No DSL produced.")
            continue

        # Repair common LLM formatting mistakes (parenthesized / space-arg forms)
        dsl_prog = repair_llm_dsl(dsl_prog)

        # Rewrite sel="all" -> sel="all_XXXX" once we have a loaded structure
        dsl_prog = repair_sel_all_to_last(dsl_prog, last_all_sel)

        # Repair color_by_chain missing target (e.g., sel="all_cartoon")
        dsl_prog = repair_color_by_chain_defaults(dsl_prog, last_all_sel)

        # OPTIONAL: prefer selecting by concrete object id when allowed
        if os.environ.get("MCL_PREFER_OBJECT_ID") == "1":
            dsl_prog = prefer_object_id(dsl_prog)

        print("DSL>", dsl_prog)

        v = await validate_dsl(dsl_prog)
        if not v.get("ok"):
            print("Validator errors:", v.get("errors"))
            continue

        out = await execute_dsl(dsl_prog)
        print(json.dumps(out, indent=2))

        # Memoize last created object id for the PDB code used in add_structure
        m_pdb = re.search(r'add_structure\(\s*PDBID\s*=\s*"([0-9A-Za-z]{4})"\s*\)', dsl_prog)
        if m_pdb and isinstance(out, dict):
            obj = out.get("result")
            if isinstance(obj, str):
                id_map[m_pdb.group(1).lower()] = obj
                # If executor returns an all_<code> selection, use it for generic followups.
                if obj.startswith("all_"):
                    last_all_sel = obj
                    save_repl_state(chroma_dir, {"last_all_sel": last_all_sel})

        # Also handle filePath case when executor returns all_<stem>
        if dsl_prog.startswith("add_structure") and isinstance(out, dict):
            obj = out.get("result")
            if isinstance(obj, str) and obj.startswith("all_"):
                last_all_sel = obj
                save_repl_state(chroma_dir, {"last_all_sel": last_all_sel})


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
    args = ap.parse_args()
    asyncio.run(run_repl(entity_hint=args.entity, with_context=args.with_context))


if __name__ == "__main__":
    main()

