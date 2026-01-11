#!/usr/bin/env python3
"""
molcommandnl.script.mol_repl

Interactive REPL: NL -> DSL -> validate -> execute via unitymol_copilot.

Option A behavior:
- Use unitymol_copilot.dsl_normalizer.normalize_dsl() in DEV mode to fix the common
  "LLM-ish" DSL formatting issues centrally (parenthesized form, space-args form,
  missing commas, double quotes, alias verbs like add_coloring/colorByChain, etc).
- Keep only the REPL-specific stateful rewrite:
    sel="all" -> sel="<last_all_sel>"
  because the normalizer intentionally defaults to sel="all" when inferring targets,
  and only the REPL knows what the last focused structure is.

Other:
- Persist last_all_sel across REPL runs (repl_state.json).
- Pre-rewrite common shorthand NL like "load 1crn" -> "Load PDB ID 1crn"
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

# --- unitymol_copilot tools (validate/execute + DEV normalizer) ---
sys.path.append(os.path.expanduser("~/unitymol_copilot"))
from mcp_server import validate_dsl, execute_dsl  # noqa: E402

try:
    # Available in unitymol_copilot/
    from dsl_normalizer import normalize_dsl  # noqa: E402
except Exception:
    normalize_dsl = None


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

    # show(sel="all", ...) -> show(sel="<last_all_sel>", ...)
    dsl_text = re.sub(
        r'show\(\s*sel\s*=\s*"all"\s*,',
        f'show(sel="{last_all_sel}",',
        dsl_text,
    )
    # hide(sel="all") -> hide(sel="<last_all_sel>")
    dsl_text = re.sub(
        r'hide\(\s*sel\s*=\s*"all"\s*\)',
        f'hide(sel="{last_all_sel}")',
        dsl_text,
    )
    # color_by_chain(sel="all", ...) -> color_by_chain(sel="<last_all_sel>", ...)
    dsl_text = re.sub(
        r'color_by_chain\(\s*sel\s*=\s*"all"\s*,',
        f'color_by_chain(sel="{last_all_sel}",',
        dsl_text,
    )
    return dsl_text


def maybe_focus_last_all_sel_from_program(dsl_prog: str) -> str | None:
    """
    If a command explicitly references sel="all_XXXX", treat that as the new focus.
    This makes follow-ups ("Color ... by chain") act on the last-mentioned structure,
    not just the last-loaded one.
    """
    m = re.search(r'sel\s*=\s*"(all_[0-9A-Za-z]{4})"\s*', dsl_prog)
    if m:
        return m.group(1)
    return None


async def run_repl(entity_hint=None, with_context=False):
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

    id_map: dict[str, str] = {}

    state = load_repl_state(chroma_dir)
    last_all_sel = state.get("last_all_sel", "all")

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

        # REPL-only: rewrite sel="all" -> sel="<last_all_sel>"
        dsl_prog = repair_sel_all_to_last(dsl_prog, last_all_sel)

        if os.environ.get("MCL_PREFER_OBJECT_ID") == "1":
            dsl_prog = prefer_object_id(dsl_prog)

        print("DSL>", dsl_prog)

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
                    save_repl_state(chroma_dir, {"last_all_sel": last_all_sel})

        if dsl_prog.startswith("add_structure") and isinstance(out, dict):
            obj = out.get("result")
            if isinstance(obj, str) and obj.startswith("all_"):
                last_all_sel = obj
                save_repl_state(chroma_dir, {"last_all_sel": last_all_sel})

        # ALSO update focus when a command explicitly mentions sel="all_XXXX"
        focused = maybe_focus_last_all_sel_from_program(dsl_prog)
        if focused and focused != last_all_sel:
            last_all_sel = focused
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

    # QoL: enable DEV normalizer without exporting env vars
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

    args = ap.parse_args()

    if args.dev or args.dev_loose:
        os.environ["MOLCOMMANDNL_DEV"] = "1"
    if args.dev_loose:
        os.environ["MOLCOMMANDNL_DEV_LOOSE"] = "1"
    if args.show_norm:
        os.environ["MOLCOMMANDNL_SHOW_NORM"] = "1"

    asyncio.run(run_repl(entity_hint=args.entity, with_context=args.with_context))


if __name__ == "__main__":
    main()

