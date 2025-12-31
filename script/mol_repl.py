#!/usr/bin/env python3
"""
molcommandnl.script.mol_repl

Interactive REPL: NL -> DSL -> validate -> execute via unitymol_copilot.
Includes small repairs for common LLM formatting slips.

Dev QoL:
- Repair common LLM formatting mistakes ((show ...) or space-arg forms).
- Repair color_by_chain single-arg slips (sel encodes target).
- Repair color_by_chain selection slips (sel="all_lines"/"all_surface"/"all_tube").
- Rewrite generic sel="all" into the last loaded structure selection (all_<code>).
- Persist last_all_sel across REPL runs (repl_state.json) so structure -> representation
  works even when run in separate REPL processes.

Stability:
- Pre-rewrite common shorthand NL like "load 1crn" -> "Load PDB ID 1crn"
  to avoid retrieval falling into non-DSL artifacts (e.g. select_structure(sim=...)).
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


def nl_pre_rewrite(q: str) -> str:
    """
    Pre-rewrite a few common shorthand NL patterns into explicit, sample-friendly phrasing.

    This prevents retrieval/LLM from producing non-DSL artifacts like select_structure(sim=...).
    """
    t = cleaned(q)
    if not t:
        return t

    # "load 1crn" / "open 3eam" -> "Load PDB ID 1crn"
    m = re.match(r"^(?:please\s+)?(?:load|open)\s+([0-9A-Za-z]{4})\s*$", t, flags=re.I)
    if m:
        return f"Load PDB ID {m.group(1)}"

    return t


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
        inner = re.sub(r'"\s+([A-Za-z_][A-Za-z0-9_]*\s*=)', r'", \1', inner)
        return f"{fn}({inner})"

    # Case 2: show sel="..." rep="..." -> show(sel="...", rep="...")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", s, flags=re.S)
    if m and "(" not in s and "=" in s:
        fn, inner = m.group(1), m.group(2).strip()
        inner = re.sub(r'"\s+([A-Za-z_][A-Za-z0-9_]*\s*=)', r'", \1', inner)
        return f"{fn}({inner})"

    return s


def repair_double_quotes_in_kwargs(dsl_text: str) -> str:
    """
    Fix occasional LLM quote slip:
      target=""surface" -> target="surface"
      target="surface"" -> target="surface"
    Conservative: only touches sel= / target=.
    """
    s = cleaned(dsl_text)
    if not s:
        return s

    s2 = re.sub(r'(target|sel)\s*=\s*""([^"]+)"', r'\1="\2"', s)
    s2 = re.sub(r'(target|sel)\s*=\s*"([^"]+)""', r'\1="\2"', s2)
    return s2


_COLOR_TARGET_ALIASES = {
    "atoms": "atom",
    "bonds": "bond",
    "lines": "line",
    "points": "point",
    "tubes": "tube",
    "surfaces": "surface",
    "cartoons": "cartoon",
}

_COLOR_TARGETS = {"atom", "bond", "cartoon", "line", "point", "surface", "tube"}


def repair_color_by_chain_target_aliases(dsl_text: str) -> str:
    """
    Normalize plural/alias target values for strict validation:
      atoms->atom, lines->line, tubes->tube, ...
    """
    s = cleaned(dsl_text)
    if not s.lower().startswith("color_by_chain("):
        return s

    m = re.search(r'target\s*=\s*"([^"]+)"', s)
    if not m:
        return s

    tgt = m.group(1).strip().lower()
    tgt2 = _COLOR_TARGET_ALIASES.get(tgt, tgt)
    if tgt2 == tgt:
        return s

    return re.sub(r'target\s*=\s*"([^"]+)"', f'target="{tgt2}"', s)


def repair_color_by_chain_defaults(dsl_text: str, last_all_sel: str) -> str:
    """
    Repair common LLM slip where it emits only a sel, and encodes the target in the sel:

      color_by_chain(sel="all_cartoon") -> color_by_chain(sel="<last_all_sel>", target="cartoon")
      color_by_chain(sel="cartoon")     -> color_by_chain(sel="<last_all_sel>", target="cartoon")
      color_by_chain(sel="all_lines")   -> ... target="line"
      color_by_chain(sel="all_tube")    -> ... target="tube"
      color_by_chain(sel="atoms")       -> ... target="atom"

    If we can't confidently infer the target, we leave it unchanged.
    """
    s = cleaned(dsl_text)
    m = re.fullmatch(r'color_by_chain\(\s*sel\s*=\s*"([^"]+)"\s*\)', s)
    if not m:
        return s

    raw = m.group(1).strip().lower()

    # allow "all_<target>" and "<target>" (plus plurals)
    if raw.startswith("all_"):
        candidate = raw[4:]
    else:
        candidate = raw

    candidate = _COLOR_TARGET_ALIASES.get(candidate, candidate)
    if candidate not in _COLOR_TARGETS:
        return s

    sel = last_all_sel or "all"
    return f'color_by_chain(sel="{sel}", target="{candidate}")'


def repair_color_by_chain_sel_repname(dsl_text: str, last_all_sel: str) -> str:
    """
    If model produces sel="all_lines"/"all_surface"/"all_tube"/etc, rewrite sel to last_all_sel.

    Example:
      color_by_chain(sel="all_lines", target="line")
        -> color_by_chain(sel="all_1crn", target="line")
    """
    s = cleaned(dsl_text)
    m = re.fullmatch(
        r'color_by_chain\(\s*sel\s*=\s*"([^"]+)"\s*,\s*target\s*=\s*"([^"]+)"\s*\)',
        s,
    )
    if not m:
        return s

    sel_raw = m.group(1).strip()
    tgt_raw = m.group(2).strip().lower()
    tgt = _COLOR_TARGET_ALIASES.get(tgt_raw, tgt_raw)

    if tgt not in _COLOR_TARGETS:
        return s

    sel_l = sel_raw.lower()
    bad = {
        "all_lines",
        "all_line",
        "all_surface",
        "all_tube",
        "all_cartoon",
        "all_atoms",
        "all_atom",
        "all_bonds",
        "all_bond",
        "all_points",
        "all_point",
    }

    # Also treat "all_<target>" as suspicious (selection name, not structure selection).
    suspicious = sel_l in bad
    if sel_l.startswith("all_"):
        c = sel_l[4:]
        c = _COLOR_TARGET_ALIASES.get(c, c)
        if c in _COLOR_TARGETS:
            suspicious = True

    if not suspicious:
        return s

    sel2 = last_all_sel or "all"
    return f'color_by_chain(sel="{sel2}", target="{tgt}")'


def repair_sel_all_to_last(dsl_text: str, last_all_sel: str) -> str:
    """
    If DSL uses sel="all" (common in generic samples), rewrite it to the most
    recently created selection like all_1crn.
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

        # NL pre-rewrite (stability)
        q = nl_pre_rewrite(q)

        # NL → DSL
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

        # Repairs (REPL-side)
        dsl_prog = repair_llm_dsl(dsl_prog)
        dsl_prog = repair_double_quotes_in_kwargs(dsl_prog)

        # Rewrite sel="all" -> sel="<last_all_sel>"
        dsl_prog = repair_sel_all_to_last(dsl_prog, last_all_sel)

        # color_by_chain: normalize target aliases
        dsl_prog = repair_color_by_chain_target_aliases(dsl_prog)

        # color_by_chain: infer missing target when only sel is provided
        dsl_prog = repair_color_by_chain_defaults(dsl_prog, last_all_sel)

        # color_by_chain: fix sel="all_lines"/"all_surface"/"all_tube" slips
        dsl_prog = repair_color_by_chain_sel_repname(dsl_prog, last_all_sel)

        if os.environ.get("MCL_PREFER_OBJECT_ID") == "1":
            dsl_prog = prefer_object_id(dsl_prog)

        print("DSL>", dsl_prog)

        v = await validate_dsl(dsl_prog)
        if not v.get("ok"):
            print("Validator errors:", v.get("errors"))
            # If DEV mode is on and server attached debug info, show it (optional).
            if v.get("dev"):
                print("[DEV]", json.dumps(v["dev"], indent=2))
            continue

        out = await execute_dsl(dsl_prog)
        print(json.dumps(out, indent=2))

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

