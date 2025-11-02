#!/usr/bin/env python3
import re
from lark import Lark, Transformer, v_args

GRAMMAR = r"""
start: stmt+

?stmt: add_structure
     | select_stmt
     | show_stmt
     | hide_stmt
     | color_by_chain_stmt

# ---- add_structure ----
add_structure : "add_structure" "(" add_args ")"
add_args      : pdbid_arg | file_arg
pdbid_arg     : "PDBID" "=" ESCAPED_STRING      -> arg_pdbid
file_arg      : "filePath" "=" ESCAPED_STRING   -> arg_file

# ---- select ----
select_stmt   : "select" "(" "query" "=" ESCAPED_STRING ("," "name" "=" ESCAPED_STRING)? ")"

# ---- show/hide ----
show_stmt     : "show" "(" "sel" "=" ESCAPED_STRING "," "rep" "=" ESCAPED_STRING ")"
hide_stmt     : "hide" "(" "sel" "=" ESCAPED_STRING ")"

# ---- color_by_chain ----
color_by_chain_stmt : "color_by_chain" "(" "sel" "=" ESCAPED_STRING "," "target" "=" ESCAPED_STRING ")"

%import common.ESCAPED_STRING
%import common.WS_INLINE
%import common.NEWLINE
SEMI: ";"

%ignore WS_INLINE
%ignore NEWLINE
%ignore SEMI
"""

_parser = Lark(GRAMMAR, start="start", maybe_placeholders=False)

def _unq(tok):
    s = str(tok)     # ESCAPED_STRING like '"abc"'
    return s[1:-1]

@v_args(inline=True)
class BuildAST(Transformer):
    def start(self, *stmts):
        return list(stmts)

    # --- add_structure ---
    def arg_pdbid(self, s):
        return ("PDBID", _unq(s))

    def arg_file(self, s):
        return ("filePath", _unq(s))

    def add_args(self, *pairs):
        return {k: v for (k, v) in pairs}

    def add_structure(self, args):
        return {"stmt": "add_structure", "args": args}

    # --- select ---
    def select_stmt(self, q, *maybe_name):
        name = _unq(maybe_name[0]) if maybe_name else "selection"
        return {"stmt": "select", "args": {"query": _unq(q), "name": name}}

    # --- show/hide ---
    def show_stmt(self, sel, rep):
        return {"stmt": "show", "args": {"sel": _unq(sel), "rep": _unq(rep)}}

    def hide_stmt(self, sel):
        return {"stmt": "hide", "args": {"sel": _unq(sel)}}

    # --- color_by_chain ---
    def color_by_chain_stmt(self, sel, target):
        return {"stmt": "color_by_chain", "args": {"sel": _unq(sel), "target": _unq(target)}}

def parse_and_validate_molcommand(program: str):
    """
    Parse a molcommand script and return (ok: bool, ast: list[dict] | None, errors: list[str]).
    """
    try:
        tree = _parser.parse(program)
        ast = BuildAST().transform(tree)   # list[dict]

        errors = []
        allowed_color_targets = {"atom", "bond", "cartoon", "surface", "point", "tube", "line"}

        for node in ast:
            stmt = node.get("stmt")
            args = node.get("args", {})

            if stmt == "add_structure":
                has_pdb  = "PDBID" in args
                has_file = "filePath" in args
                if has_pdb == has_file:
                    errors.append("add_structure requires exactly one of PDBID or filePath")
                if has_pdb:
                    pdb = args["PDBID"].strip()
                    if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb):
                        errors.append(f"Invalid PDB ID: {pdb}")
                if has_file and not args["filePath"].strip():
                    errors.append("filePath must be non-empty")

            elif stmt == "select":
                q    = args.get("query", "").strip()
                name = args.get("name", "").strip()
                if not q:
                    errors.append("select.query must be non-empty")
                if not name:
                    errors.append("select.name must be non-empty")
                if any(ch in name for ch in ['"', "\n", "\r", "\t"]):
                    errors.append(f'select.name contains invalid characters: {name!r}')

            elif stmt == "show":
                if not args.get("sel", "").strip():
                    errors.append("show.sel must be non-empty")
                if not args.get("rep", "").strip():
                    errors.append("show.rep must be non-empty")

            elif stmt == "hide":
                if not args.get("sel", "").strip():
                    errors.append("hide.sel must be non-empty")

            elif stmt == "color_by_chain":
                if not args.get("sel", "").strip():
                    errors.append("color_by_chain.sel must be non-empty")
                tgt = args.get("target", "").strip().lower()
                if tgt not in allowed_color_targets:
                    errors.append(
                        f"color_by_chain.target must be one of {sorted(allowed_color_targets)}, got {args.get('target')!r}"
                    )

        if errors:
            return False, None, errors
        return True, ast, []
    except Exception as e:
        return False, None, [str(e)]

