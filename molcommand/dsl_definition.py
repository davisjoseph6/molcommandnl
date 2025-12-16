# -*- coding: utf-8 -*-
import re
import os
import io
import string
from typing import List, Dict, Any, Optional

from dsl_interface import DSLInterface
from utils import load_config, load_yaml_file
from llm_client import load_prompt
import json
import inspect

# ---- Allow-list for select(...) keys (MolCommandNL side)
ALLOWED_SELECT_KEYS = {"name", "query", "PDBID", "sel", "scope"}


class DSL(DSLInterface):
    """Implementation of DSL interface for MolCommandNL automation DSL"""

    def __init__(self):
        """Initialize the MolCommandNL DSL with configs and rules"""
        self.debug_enabled = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config = load_config(script_dir + '/../config/env')

        # Effective Chroma dir
        env_override = os.environ.get("MOLCOMMANDNL_CHROMA_DIR")
        if env_override:
            self.CHROMA_PATH = env_override
        else:
            cfg_base = (config.get('CHROMAPATH') or os.path.expanduser("~/.cache"))
            self.CHROMA_PATH = os.path.join(cfg_base, "molcommand")

        # Load syntax, rules, prompts, hierarchy
        syntax_path = script_dir + "/syntax.yaml"
        rules_path = script_dir + "/rules.yaml"
        hierarchy_path = script_dir + "/hierarchy.yaml"
        entity_ctx_path = script_dir

        self._syntax = load_yaml_file(syntax_path)
        self._rules = load_yaml_file(rules_path)
        self._hierarchy_schema = load_yaml_file(hierarchy_path)
        self._sysprompt_entity_context, self._usrprompt_entity_context = load_prompt(
            "entity_context", entity_ctx_path
        )

    # --------------------------
    # Prompt scaffolding helpers
    # --------------------------
    def extract_key_terms_as_string(self, query: str) -> str:
        stop_words = {
            "the", "a", "an", "it", "with", "by", "for", "this", "there",
            "is", "are", "can", "how", "what", "when", "where", "which", "who", "why"
        }
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        return " ".join(keywords)

    def get_syntax(self, entities: List[str]) -> str:
        parts = []
        parts.append("### Global Syntax")
        for entry in self._syntax.get("global_syntax", []):
            if isinstance(entry, dict):
                parts.append("- " + entry["text"])

        parts.append("\n### Select Statements")
        for entry in self._syntax.get("select_statements", []):
            if isinstance(entry, dict) and any(e in entry["entities"] for e in entities):
                parts.append(entry["text"])

        parts.append("\n### Add Statements")
        for entry in self._syntax.get("add_statements", []):
            if isinstance(entry, dict) and any(e in entry["entities"] for e in entities):
                parts.append(entry["text"])

        parts.append("\n### Update Statements")
        for entry in self._syntax.get("update_statements", []):
            if isinstance(entry, dict) and any(e in entry["entities"] for e in entities):
                parts.append(entry["text"])

        parts.append("\n### Delete Statements")
        for entry in self._syntax.get("delete_statements", []):
            if isinstance(entry, dict) and any(e in entry["entities"] for e in entities):
                parts.append(entry["text"])

        parts.append("\n")
        return "\n".join(parts)

    def get_rules(self, entities: List[str]) -> str:
        parts = []
        for entry in self._rules.get("global_rules", []):
            if isinstance(entry, dict):
                parts.append("- " + entry["text"])
        for entry in self._rules.get("entity_rules", []):
            if isinstance(entry, dict) and any(e in entry["entities"] for e in entities):
                parts.append("- " + entry["text"])
        return "\n".join(parts)

    def get_sysprompt_entity_context(self) -> str:
        return self._sysprompt_entity_context

    def get_usrprompt_entity_context(self) -> str:
        return self._usrprompt_entity_context

    def normalize_text(self, query_text: str) -> str:
        if self.debug_enabled:
            print(f"[SUB-DGB] Normalizing : {query_text}")
        text = query_text.lower()

        replacements = {
            "how do i": "", "how to": "", "i want to": "",
            "can you tell me how to": "", "can you": "",
            "please": "", "show me how to": "",
            "what is the command for": "", "what is the syntax for": "",
            "command to": "", "code for": "",
        }
        for p, r in replacements.items():
            text = text.replace(p, r)

        text = self.extract_key_terms_as_string(text)
        text = " ".join(text.split())

        keyword_mappings = {
            "color": ["colorby", "coloring"],
            "select": ["selection"],
            "hide": ["visibility", "display"],
            "show": ["visibility", "display"],
            "load": ["loading", "structure"],
            "center": ["view", "focus"],
            "rotate": ["rotation", "movement"],
            "annotate": ["annotation", "measure"],
        }
        for kw, rel in keyword_mappings.items():
            if kw in text:
                for t in rel:
                    if t not in text:
                        text += f" {t}"

        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\b(create|insert|add|generate|load|fetch)\b", "add", text)
        text = re.sub(r"\b(red|blue|green|teal|olive)\b", "color", text)
        text = re.sub(r"\b(picture|photo|image)\b", "image", text)
        text = re.sub(r"\btitles?\b", "heading", text)

        if self.debug_enabled:
            print(f"[SUB-DGB]               {text}")
        return text.strip()

    # --------------------------
    # Canonicalization helpers
    # --------------------------
    def get_statement_aliases(self) -> Dict[str, str]:
        """Maps alias statement names to canonical DSL function names."""
        return {
            # Office aliases (existing)
            "add_picture": "add_images",
            "delete_image": "delete_shapes",
            # MolCommandNL friendly forms → canonical validator verbs
            "select_structure": "select",
            "select_atoms": "select",
            "select_all": "select",
            "select_cartoons_by_chain": "select",
            "load_pdb": "add_structure",
            "insert_protein": "add_structure",
            "hide_all": "hide",
            "show_as_cartoon": "update_representation",
            "show_cartoon": "update_representation",
            "color_chains": "update_coloring",
            "colour_chains": "update_coloring",
        }

    def get_valid_enums(self) -> Dict[str, List[str]]:
        return {
            "underline": ["None", "Single", "Double", "Wavy"],
            "horizontalAlignment": ["Left", "Center", "Right"],
        }

    # --------------------------
    # Context pruning (Office/ODSL side kept)
    # --------------------------
    def prune_context_tree(self, tree: dict, entity_types: list[str]) -> dict:
        if self.debug_enabled:
            print(f"[SUB-DGB] in function {inspect.currentframe().f_code.co_name}")
            print(f"[SUB-DBG] Original Context is: {tree}")

        if not isinstance(tree, dict):
            if self.debug_enabled:
                print(f"[SUB-DGB] Attention: context is not an instance of tree!")
            return None

        entity_types = self._expand_entity_types_for_context(entity_types)

        if tree.get("_type") in entity_types:
            if self.debug_enabled:
                print(f"[SUB-DGB] We matched entity type {tree.get('_type')}!")
            return tree

        if "slides" in tree:
            pruned_slides = []
            for slide in tree["slides"]:
                pruned_slide = self.prune_context_tree(slide, entity_types)
                if pruned_slide:
                    pruned_slides.append(pruned_slide)
            if pruned_slides:
                return {"slides": pruned_slides}
            return None

        if "shapes" in tree:
            pruned_shapes = []
            for shape in tree["shapes"]:
                pruned_shape = self.prune_context_tree(shape, entity_types)
                if pruned_shape:
                    pruned_shapes.append(pruned_shape)
            if pruned_shapes:
                return {"shapes": pruned_shapes}
            return None

        pruned = {}
        for key, value in tree.items():
            if isinstance(value, dict):
                pruned_value = self.prune_context_tree(value, entity_types)
                if pruned_value:
                    pruned[key] = pruned_value
            elif isinstance(value, list):
                pruned_list = [self.prune_context_tree(item, entity_types) for item in value]
                pruned_list = [item for item in pruned_list if item]
                if pruned_list:
                    pruned[key] = pruned_list
            else:
                if any(
                    self._node_contains_entity(child, entity_types)
                    for child in tree.values()
                    if isinstance(child, (dict, list))
                ):
                    pruned[key] = value

        return pruned if pruned else None

    def _expand_entity_types_for_context(self, entity_types):
        expanded_types = set(entity_types)
        for entity in entity_types:
            children = self._hierarchy_schema["valid_scopes"].get(entity, [])
            expanded_types.update(children)
        return list(expanded_types)

    def _node_contains_entity(self, node, entity_types):
        if isinstance(node, dict):
            if node.get("_type") in entity_types:
                return True
            return any(self._node_contains_entity(v, entity_types) for v in node.values())
        elif isinstance(node, list):
            return any(self._node_contains_entity(item, entity_types) for item in node)
        return False

    # --------------------------
    # LLM output → DSL extraction
    # --------------------------
    def extract_dsl_lines(self, raw: str) -> List[str]:
        """
        Extract valid DSL instruction lines from raw LLM output.
        Capture both canonical verbs and friendly shorthands.
        """
        lines, buffer, inside = [], "", False

        for line in io.StringIO(raw or ""):
            stripped = line.strip()
            if not stripped:
                continue

            # Accept select / select_* / add / update / delete / hide / show / color_*
            if re.match(
                r"^\s*(\w+\s*=)?\s*"
                r"((?:select|add|update|delete|hide|show)(?:_\w+)?|color_\w+)\s*\(",
                stripped,
            ):
                buffer = stripped
                inside = True
                if stripped.endswith(")"):
                    lines.append(buffer)
                    buffer = ""
                    inside = False
            elif inside:
                buffer += " " + stripped
                if stripped.endswith(")"):
                    lines.append(buffer)
                    buffer = ""
                    inside = False
            else:
                if self.debug_enabled:
                    print(f"[INFO] ⚠️ Ignoring non-DSL line: {stripped}")

        # ---- Filter out invalid select(...) lines (e.g., select(score=...))
        def _is_valid(line: str) -> bool:
            m = re.match(r"^\s*(?:\w+\s*=)?\s*(\w+)\s*\((.*)\)\s*$", line)
            if not m:
                return False
            func, argstr = m.group(1), m.group(2)
            if func != "select":
                return True
            if "score=" in argstr:
                return False
            # If key=value pairs are present, all keys must be allowed
            if "=" in argstr:
                keys = [k.strip() for k in re.findall(r"(\w+)\s*=", argstr)]
                return all(k in ALLOWED_SELECT_KEYS for k in keys)
            # Bare positional token is fine (e.g., select(all_1kx2))
            return True

        lines = [l for l in lines if _is_valid(l)]

        if buffer:
            print(f"[INFO] ⚠️ Incomplete DSL line left open: {buffer}")
        return lines

    # --------------------------
    # Scope validation (Office side kept)
    # --------------------------
    def validate_scope(self, node: Dict[str, Any]) -> Dict[str, Any]:
        if not node or not isinstance(node, dict):
            return node

        """Perform scope validation for select_* statements (Office DSL)."""
        if not node["stmt"].startswith("select_"):
            return node

        target_entity = node["stmt"].split("_")[1]
        args = node["args"]

        if "scope" not in args and any(
            k in self._hierarchy_schema["contextual_scopes"].get(target_entity, []) for k in args
        ):
            return node

        scope_entity = args.get("scope")
        setDefaultFlag = False

        if scope_entity == "document":
            scope_entity = "Scene"
            node["args"]["scope"] = "Scene"

        if scope_entity is None:
            setDefaultFlag = True
        else:
            try:
                if not isinstance(scope_entity, str):
                    raise ValueError(
                        f"Scope must be a string or special keyword. Got: {scope_entity} "
                        f"(type: {type(scope_entity)})"
                    )
            except ValueError as e:
                print(f"[ERROR] : {e} -> will use default scope!")
                setDefaultFlag = True

        if setDefaultFlag:
            default_scope = self._hierarchy_schema["scope_defaults"].get(
                node["stmt"], "Selection"
            )
            print(f"[INFO] : normalized scope for '{node['stmt']}' to default {default_scope}")
            node["args"]["scope"] = default_scope
            scope_entity = default_scope

        if scope_entity in self._hierarchy_schema["special_scopes"]:
            return node

        normalized_scope = scope_entity.rstrip("s") if scope_entity.endswith("s") else scope_entity
        valid_parents = self._hierarchy_schema["valid_scopes"][target_entity]

        try:
            if scope_entity not in valid_parents and normalized_scope not in valid_parents:
                raise ValueError(
                    f"Cannot select '{target_entity}' from '{scope_entity}'. "
                    f"Valid scopes: {valid_parents + self._hierarchy_schema['special_scopes']}"
                )
        except ValueError as e:
            print(f"[ERROR] : {e} -> skipping invalid command!")
            return None

        return node

    # --------------------------
    # Post-parse corrections
    # --------------------------
    def additional_corrections(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        MolCommand-specific corrections to AST nodes.
        - Normalize friendly verbs to canonical args.
        - Argument synonym cleanup.
        - Heuristic mapping for PDB IDs.
        - Clamp numeric [0,1] where relevant.
        """
        stmt = node.get("stmt", "")
        args = node.setdefault("args", {})

        # Common arg synonyms
        if "selection" in args and "sel" not in args:
            args["sel"] = args.pop("selection")
        if "path" in args and "filePath" not in args:
            args["filePath"] = args.pop("path")
        if "file" in args and "filePath" not in args:
            args["filePath"] = args.pop("file")

        # --- SELECT normalization ---
        if stmt == "select":
            # Drop any select with unknown keys early
            invalid = [k for k in list(args.keys()) if k not in ALLOWED_SELECT_KEYS]
            if invalid:
                print(f"[INFO] Dropping select(...) with invalid keys: {invalid}")
                return None

            pdb = (
                args.pop("structureName", None)
                or args.pop("pdb", None)
                or args.pop("PDB", None)
            )
            nm = args.get("name")

            # If user passed a 4-char code, normalize to all_<code>
            code = None
            if isinstance(pdb, str) and re.fullmatch(r"[0-9a-zA-Z]{4}", pdb.strip()):
                code = pdb.strip().lower()
            elif isinstance(nm, str) and re.fullmatch(r"[0-9a-zA-Z]{4}", nm.strip()):
                code = nm.strip().lower()

            if code:
                args["query"] = args.get("query") or "all"
                args["name"] = args.get("name") or code
            else:
                if isinstance(nm, str) and "query" not in args:
                    args["query"] = "all"

            args.pop("id", None)

        # --- HIDE normalization ---
        elif stmt == "hide":
            if not args:
                args["scope"] = "all"

        # --- Representation & coloring defaults ---
        elif stmt == "update_representation":
            args.setdefault("type", "cartoon")

        elif stmt == "update_coloring":
            args.setdefault("method", "chain")

        # --- Representation name synonyms ---
        if stmt == "show":
            rep = args.get("rep")
            if isinstance(rep, str):
                rep_syn = {
                    "licorice": "lines",
                    "stick": "lines",
                    "ball+stick": "lines",
                    "ball_and_stick": "lines",
                    "spacefill": "spheres",
                    "vdw": "spheres",
                    "cpk": "spheres",
                    "ribbon": "cartoon",
                    "tube": "cartoon",
                }
                norm = rep_syn.get(rep.lower())
                if norm:
                    args["rep"] = norm
                allowed = {"cartoon", "lines", "spheres", "surface"}
                if isinstance(args.get("rep"), str) and args["rep"] not in allowed:
                    args["rep"] = "lines"

        # --- SHOW/HIDE/COLOR_BY_CHAIN normalization ---
        if stmt in ("show", "hide", "color_by_chain"):

            def _sel_token(v):
                if not isinstance(v, str):
                    return v
                vv = v.strip()
                if not vv:
                    return v
                # already a concrete selection name
                if vv.lower().startswith("all_"):
                    return vv.lower()
                # only treat real PDB IDs as structure codes (1crn, 3eam, ...)
                if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", vv):
                    return "all_" + vv.lower()
                return vv

            if "sel" in args:
                args["sel"] = _sel_token(args["sel"])


        # --- ADD_STRUCTURE normalization ---
        if stmt == "add_structure":
            # Map common keys to PDBID when they look like 4-char codes
            for k in ("name", "pdb", "pdbid", "PDB", "id"):
                if k in args and "PDBID" not in args:
                    val = str(args[k]).strip()
                    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", val):
                        args["PDBID"] = args.pop(k)
                        break

            # Normalize filePath while preserving case and try to fix
            # case-only mismatches when the filesystem is case-sensitive.
            fp = args.get("filePath")
            if isinstance(fp, str):
                path = fp.strip()
                # Expand ~ and environment variables, but DO NOT change case.
                path = os.path.expanduser(os.path.expandvars(path))

                try:
                    if not os.path.exists(path):
                        # Best-effort case-insensitive correction on the basename.
                        dirname = os.path.dirname(path) or "."
                        basename = os.path.basename(path)
                        if basename and os.path.isdir(dirname):
                            try:
                                entries = os.listdir(dirname)
                            except OSError:
                                entries = []
                            matches = [
                                entry
                                for entry in entries
                                if entry.lower() == basename.lower()
                            ]
                            if len(matches) == 1:
                                # Use the actual on-disk name
                                path = os.path.join(dirname, matches[0])
                except Exception:
                    # Any error → keep original path; this is only a convenience layer.
                    pass

                args["filePath"] = path

        # Clamp Office numeric fields
        for key in ("fillTransparency", "lineTransparency"):
            if key in args and isinstance(args[key], (int, float)):
                args[key] = max(0.0, min(1.0, args[key]))

        node["args"] = args
        return node        

    # --------------------------
    # Chroma collection name
    # --------------------------
    def get_collection_name(self) -> str:
        return "molcommand_samples"

    def set_debug(self, dbg: bool) -> None:
        print(">>> DEBUGGING FOR OFFICE DSL TOGGLED <<<")
        self.debug_enabled = dbg

    # ---------- Final text-level adapter for the validator grammar ----------
    def postprocess_for_validator(
        self,
        dsl_text: str,
        utterance: Optional[str] = None,
    ) -> str:
        """
        Final rewrite to match UnityMol validator grammar where safe.

        IMPORTANT:
        - By default we DO NOT rewrite select(...) into colon tokens (name: / id:).
          For MolCommandNL we normalize selections to select(TOKEN) where TOKEN
          can be 'last', 'all_xxxx', etc., and ensure that add_structure(PDBID="CODE")
          has been issued for any referenced PDB code.
        """
        text = dsl_text or ""

        # --- Session cache for loaded PDB codes (persisted in CHROMA_PATH) ---
        _loaded_codes_path = os.path.join(self.CHROMA_PATH, "loaded_codes.json")

        def _read_loaded_codes():
            try:
                with open(_loaded_codes_path, "r") as f:
                    arr = json.load(f)
                if isinstance(arr, list):
                    return [
                        c for c in arr
                        if isinstance(c, str)
                        and re.fullmatch(r"[0-9A-Za-z]{4}", c)
                    ]
            except Exception:
                pass
            return []

        def _write_loaded_codes(codes):
            try:
                os.makedirs(os.path.dirname(_loaded_codes_path), exist_ok=True)
                with open(_loaded_codes_path, "w") as f:
                    json.dump(codes, f)
            except Exception:
                pass

        def _update_loaded_codes_from_text(t):
            prev = _read_loaded_codes()
            seen = set(c.lower() for c in prev)
            new = []
            # add_structure(PDBID="CODE")
            for m in re.findall(
                r'add_structure\(\s*PDBID\s*=\s*["\']([0-9A-Za-z]{4})["\']\s*\)',
                t,
                re.IGNORECASE,
            ):
                c = m.lower()
                if c not in seen:
                    new.append(c)
                    seen.add(c)
            # sel="all_CODE"
            for m in re.findall(
                r'sel\s*=\s*["\']all_([0-9A-Za-z]{4})["\']',
                t,
                re.IGNORECASE,
            ):
                c = m.lower()
                if c not in seen:
                    new.append(c)
                    seen.add(c)
            # select(all_CODE)
            for m in re.findall(
                r'select\(\s*all_([0-9A-Za-z]{4})\s*\)',
                t,
                re.IGNORECASE,
            ):
                c = m.lower()
                if c not in seen:
                    new.append(c)
                    seen.add(c)
            if new:
                _write_loaded_codes(prev + new)

        # Track any codes mentioned in this DSL chunk before heavy rewrites
        _update_loaded_codes_from_text(text)

        # Also track PDB codes from the natural-language utterance, as a fallback.
        # This lets things like "Load PDB ID 1kx2" register 1kx2 even if the DSL
        # line isn't matched for some reason.
        if utterance and isinstance(utterance, str):
            # Only accept tokens that look like PDB IDs: digit + 3 alphanumerics
            candidates = re.findall(r"\b([0-9][0-9A-Za-z]{3})\b", utterance)
            if candidates:
                prev = _read_loaded_codes()
                seen = set(c.lower() for c in prev)
                new = []
                for c in candidates:
                    lc = c.lower()
                    if lc not in seen:
                        new.append(lc)
                        seen.add(lc)
                if new:
                    _write_loaded_codes(prev + new)

        # --- If select() references a PDB code, ensure we load it first (case-insensitive) ---
        def _pdb_from_select(t: str):
            m = re.search(
                r'select\(\s*"?(?:name:)?([0-9A-Za-z]{4})"?\s*\)',
                t,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
            m = re.search(
                r"select\(\s*all_([0-9A-Za-z]{4})\s*\)",
                t,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
            m = re.search(
                r'select\(\s*[^)]*\bname\s*=\s*["\']([0-9A-Za-z]{4})["\']',
                t,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
            m = re.search(
                r'select\(\s*[^)]*\bPDBID\s*=\s*["\']([0-9A-Za-z]{4})["\']',
                t,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
            return None

        code = _pdb_from_select(text)
        if code and re.search(
            rf'add_structure\(\s*PDBID\s*=\s*"{re.escape(code)}"\s*\)',
            text,
            re.IGNORECASE,
        ) is None:
            text = f'add_structure(PDBID="{code}")\n' + text

        # --- SELECT named-args → positional QUERY token (intermediate form) ---
        def _strip_quotes(s: str) -> str:
            s = s.strip()
            if (
                (s.startswith('"') and s.endswith('"')) or
                (s.startswith("'") and s.endswith("'"))
            ):
                return s[1:-1]
            return s

        def _select_rewrite(m: re.Match) -> str:
            inside = m.group(1)
            # Parse simple k=v pairs (string or bare)
            kv = dict(
                (k.strip(), _strip_quotes(v))
                for k, v in re.findall(
                    r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^,)\s]+)',
                    inside,
                )
            )

            # Helper: detect 4-char code and "all_<code>" (case-insensitive)
            def _code_from_name(n: Optional[str]) -> Optional[str]:
                if not isinstance(n, str):
                    return None
                if re.fullmatch(r"[0-9a-zA-Z]{4}", n):
                    return n.lower()
                m2 = re.fullmatch(r"all_([0-9a-zA-Z]{4})", n, re.IGNORECASE)
                if m2:
                    return m2.group(1).lower()
                return None

            # 1) If query already is all_<code>, use it directly (case-insensitive)
            q = kv.get("query")
            if isinstance(q, str) and re.fullmatch(
                r"all_[0-9a-zA-Z]{4}",
                q,
                re.IGNORECASE,
            ):
                return f"select({q})"

            # 2) query == "all" + 4-char name → all_<code>
            if isinstance(q, str) and q.lower() == "all":
                code2 = _code_from_name(kv.get("name")) or _code_from_name(kv.get("PDBID"))
                if code2:
                    token2 = (
                        "ALL_" + code2.upper()
                        if os.environ.get("MCL_QUERY_UPPER") == "1"
                        else "all_" + code2
                    )
                    return f"select({token2})"

            # 3) Only name or PDBID provided → try to synthesize all_<code>
            code3 = _code_from_name(kv.get("name")) or _code_from_name(kv.get("PDBID"))
            if code3:
                token3 = (
                    "ALL_" + code3.upper()
                    if os.environ.get("MCL_QUERY_UPPER") == "1"
                    else "all_" + code3
                )
                return f"select({token3})"

            # 4) Fall back to select(query) for safe tokens
            if isinstance(q, str) and re.fullmatch(r"[A-Za-z0-9_:.+-]+", q):
                return f"select({q})"

            return m.group(0)

        text = re.sub(r"\bselect\(\s*([^)]+)\)", _select_rewrite, text)

        # --- Normalize empty select(...) early ---
        sel_token = os.environ.get("MCL_SELECT_TOKEN", "last")
        text = re.sub(r"\bselect\(\s*\)", f"select({sel_token})", text)

        # quoted colon form → select(TOKEN)
        text = re.sub(
            r'(?i)select\(\s*"name:[^"]+"\s*\)',
            lambda _: f"select({sel_token})",
            text,
        )
        # unquoted colon form → select(TOKEN)
        text = re.sub(
            r"(?i)select\(\s*name:[^)]+\)",
            lambda _: f"select({sel_token})",
            text,
        )

        # Just in case: strip quotes around last/all
        text = re.sub(r'select\(\s*"(last|all)"\s*\)', r"select(\1)", text)

        # Optional colon-style rewrite (disabled by default; kept for completeness)
        if os.environ.get("MCL_ENABLE_COLON_SELECT", "0") == "1":
            text = re.sub(
                r'select\(\s*"id:([^"]+)"\s*\)',
                r'select("id:\1")',
                text,
            )
            text = re.sub(
                r'select\(\s*"name:([^"]+)"\s*\)',
                r'select("name:\1")',
                text,
            )
            text = re.sub(
                r'select\(\s*name\s*=\s*["\']([^"\']+)["\']\s*\)',
                r'select("name:\1")',
                text,
            )
            text = re.sub(
                r'select\(\s*PDBID\s*=\s*["\']([^"\']+)["\']\s*\)',
                r'select("name:\1")',
                text,
            )
            text = re.sub(
                r'select\(\s*name\(\s*["\']([^"\']+)["\']\s*\)\s*\)',
                r'select("name:\1")',
                text,
            )
            text = re.sub(
                r'select\(\s*name\s*:\s*([A-Za-z0-9_-]+)\s*\)',
                r'select("name:\1")',
                text,
            )

        # --- Inject select(...) before show/hide/color_by_chain when sel="CODE" (4 letters) ---
        def _inject_select_for_show_like(txt: str) -> str:
            import re as _re

            def repl(m):
                func = m.group("func")
                code4 = m.group("code").lower()
                rest = m.group("rest") or ""
                token4 = (
                    "ALL_" + code4.upper()
                    if os.environ.get("MCL_QUERY_UPPER") == "1"
                    else "all_" + code4
                )
                call = f'{func}(sel="{token4}"{rest})'
                return f"select({token4})\n" + call

            pattern = _re.compile(
                r'(?P<func>show|hide|color_by_chain)\('
                r'\s*sel\s*=\s*"(?P<code>[0-9A-Za-z]{4})"\s*(?P<rest>,[^)]*)?\)',
                _re.IGNORECASE,
            )
            return _re.sub(pattern, repl, txt)

        text = _inject_select_for_show_like(text)

        # --- HIDE everything → hide(all) normalization ---
        text = re.sub(
            r'hide\(\s*scope\s*=\s*["\']all["\']\s*\)',
            "hide(all)",
            text,
        )
        text = re.sub(r'hide\(\s*["\']all["\']\s*\)', "hide(all)", text)
        text = re.sub(r"\bhide\(\s*\)", "hide(all)", text)

        # --- Expand global hide(all) to per-PDB hides ---
        if re.search(r"\bhide\(\s*all\s*\)", text, re.IGNORECASE):
            _codes = _read_loaded_codes()
            if _codes:
                expansion = "\n".join(f'hide(sel="all_{c}")' for c in _codes)
                text = re.sub(
                    r"\bhide\(\s*all\s*\)",
                    expansion,
                    text,
                    flags=re.IGNORECASE,
                )

        # --- Special case: utterance literally says "hide everything" ---
        # Ensure *all* loaded PDB codes are hidden, even if the bank sample only
        # mentions one (e.g., hide(sel="all_1kx2")).
        if utterance and isinstance(utterance, str) and "everything" in utterance.lower():
            _codes = _read_loaded_codes()
            if _codes:
                # Which PDB codes are already being hidden in this DSL?
                present = set(
                    c.lower()
                    for c in re.findall(
                        r'hide\(\s*sel\s*=\s*"all_([0-9A-Za-z]{4})"\s*\)',
                        text,
                        re.IGNORECASE,
                    )
                )
                missing = [c for c in _codes if c.lower() not in present]
                if missing:
                    extra = "\n".join(f'hide(sel="all_{c}")' for c in missing)
                    if text and not text.endswith("\n"):
                        text += "\n"
                    text += extra

        # --- Optionally strip select(...) when we just loaded a PDB ---
        if os.environ.get("MCL_STRIP_SELECT", "1") == "1":
            lines = [ln for ln in (text.splitlines() or []) if ln.strip()]

            has_add = any(
                re.match(r"\s*add_structure\(", ln, re.IGNORECASE)
                for ln in lines
            )
            if has_add:
                # Drop any select(...) lines — validator currently rejects them.
                lines = [
                    ln for ln in lines
                    if not re.match(r"\s*select\s*\(", ln, re.IGNORECASE)
                ]
            text = "\n".join(lines)

        # --- Representation/Coloring enums as bare token---
        text = re.sub(
            r'update_representation\(\s*type\s*=\s*["\']([^"\']+)["\']\s*\)',
            r"update_representation(\1)",
            text,
        )
        text = re.sub(
            r'update_representation\(\s*["\']([^"\']+)["\']\s*\)',
            r"update_representation(\1)",
            text,
        )
        text = re.sub(
            r'update_coloring\(\s*method\s*=\s*["\']([^"\']+)["\']\s*\)',
            r"update_coloring(\1)",
            text,
        )
        text = re.sub(
            r'update_coloring\(\s*["\']([^"\']+)["\']\s*\)',
            r"update_coloring(\1)",
            text,
        )

        return text

