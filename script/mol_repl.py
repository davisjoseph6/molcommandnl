#!/usr/bin/env python3
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
sys.path.append(os.path.expanduser('~/unitymol_copilot'))
from mcp_server import validate_dsl, execute_dsl  # noqa: E402


def load_mol_dsl():
    """
    Load your molcommand DSL object/factory from molcommand/dsl_definition.py.
    Tries several common names so we’re robust to refactors.
    """
    mod = import_module('molcommand.dsl_definition')
    for name in ('build_dsl', 'make_dsl', 'create_dsl', 'load_dsl',
                 'MolCommandDSL', 'DSL', 'dsl'):
        if hasattr(mod, name):
            obj = getattr(mod, name)
            return obj() if callable(obj) else obj
    raise RuntimeError("No DSL factory found in molcommand/dsl_definition.py")


def cleaned(s: str) -> str:
    """Trim whitespace safely."""
    return (s or "").strip()


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
    m = re.match(r'^\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.*)\)\s*$', s, flags=re.S)
    if m:
        fn, inner = m.group(1), m.group(2).strip()
        # Insert commas between key/value pairs when the LLM uses spaces.
        inner = re.sub(r'"\s+([A-Za-z_][A-Za-z0-9_]*\s*=)', r'", \1', inner)
        return f"{fn}({inner})"

    # Case 2: show sel="..." rep="..." -> show(sel="...", rep="...")
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$', s, flags=re.S)
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
    # Falls back to "all" (executor ensures it exists after add_structure).
    last_all_sel = "all"

    def prefer_object_id(dsl_text: str) -> str:
        # replace select(name:XXXX) or select("name:XXXX") with select(id:OBJID) if known
        def _r_id_unquoted(m):
            code = m.group(1).lower()
            obj = id_map.get(code)
            return f'select(id:{obj})' if obj else m.group(0)

        def _r_id_quoted(m):
            code = m.group(1).lower()
            obj = id_map.get(code)
            return f'select(id:{obj})' if obj else m.group(0)

        dsl_text = re.sub(r'select\(name:([0-9A-Za-z]{4})\)', _r_id_unquoted, dsl_text)
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

        # Also handle filePath case when executor returns all_<stem>
        if dsl_prog.startswith("add_structure") and isinstance(out, dict):
            obj = out.get("result")
            if isinstance(obj, str) and obj.startswith("all_"):
                last_all_sel = obj


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

