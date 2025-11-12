# semantic_interpreter.py

"""
Semantic Interpreter (bank-backed NL → DSL)
- Entity/context classification (LLM)
- Example retrieval with Chroma
- Prompt construction (syntax + rules + examples)
- LLM synthesis → DSL
- Parse → AST → corrections → cleaned DSL
Returns a result that works with both dict- and attribute-style access.
"""

from __future__ import annotations
import os, re, io, json, difflib, inspect, logging
from typing import List, Dict, Any, Optional, Tuple, Union

from utils import load_config, get_logger
from llm_client import LLMClient
from dsl_interface import DSLInterface

# --- Env config (OLLAMA_HOST) -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(os.path.join(script_dir, "..", "config", "env"))
if (OLLAMA_HOST := config.get("OLLAMA_HOST")):
    print(f"Setting OLLAMA_HOST to {OLLAMA_HOST}")
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST

# --- Logging ------------------------------------------------------------------
loglevel = logging.INFO
log = get_logger(level=loglevel)
alog = get_logger(name="d", level=logging.DEBUG)

# --- Chroma + embeddings ------------------------------------------------------
from langchain_community.embeddings.ollama import OllamaEmbeddings
try:
    from chromadb import PersistentClient as ChromaPersistentClient
except Exception:
    ChromaPersistentClient = None  # legacy path fallback below


# ==============================================================================
# Utilities
# ==============================================================================

def _default_chroma_dir() -> str:
    """
    Priority:
      1) MOLCOMMANDNL_CHROMA_DIR
      2) CHROMA_PATH (legacy)
      3) ~/.cache/molcommandnl/chroma
    """
    path = (
        os.environ.get("MOLCOMMANDNL_CHROMA_DIR")
        or os.environ.get("CHROMA_PATH")
        or os.path.expanduser("~/.cache/molcommandnl/chroma")
    )
    os.makedirs(path, exist_ok=True)
    # Silence Chroma telemetry unless explicitly enabled
    os.environ.setdefault("CHROMADB_TELEMETRY", "0")
    return path


# Small result helper that supports both dict['dsl'] and .program / .cleaned_program
class InterpretResult(dict):
    def __init__(self, dsl: str, ast: List[Dict[str, Any]], cleaned: str):
        super().__init__(dsl=dsl, ast=ast, cleaned_dsl=cleaned)
        # attribute aliases expected by your REPL
        self.program = dsl
        self.cleaned_program = cleaned

    # Make getattr work like attributes (already set above)
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ==============================================================================
# Entity + context classifier
# ==============================================================================

class EntityContextClassifier:
    def __init__(self, llm: LLMClient, dsl: DSLInterface):
        self.llm = llm
        self.dsl = dsl

    def classify(self, utterance: str, context: Optional[Dict]) -> Tuple[List[str], bool]:
        if loglevel == logging.DEBUG:
            self.llm.set_debug(True)

        if context:
            log.info(f"User query: {utterance}\n       Context provided: {context}")
        else:
            log.info(f"User query: {utterance}\n       No context provided!")

        result = self.llm.call_with_template(
            system_template=self.dsl.get_sysprompt_entity_context(),
            user_template=self.dsl.get_usrprompt_entity_context(),
            utterance=utterance,
        )
        log.debug(f"=== LLM returned ===\n{result}\n=== END ===")

        cats = re.search(r"Categories:\s*\[(.*?)\]", result or "")
        ctx  = re.search(r"RequiresContext:\s*(true|false)", result or "", re.IGNORECASE)
        entities = [e.strip().strip('"\'') for e in (cats.group(1).split(',') if cats else []) if e.strip()]
        requires_context = ctx.group(1).lower() == "true" if ctx else False

        # Heuristics: nudge entity coverage based on wording
        ut = (utterance or "").lower()
        if "select" in ut and "selection" not in entities:
            entities.append("selection")
        if any(w in ut for w in ("show", "cartoon", "color", "colour", "hide")) and "representation" not in entities:
            entities.append("representation")
        if any(w in ut for w in ("pdb", "protein", "structure", "load", "insert", "fetch", "add")) and "structure" not in entities:
            entities.append("structure")

        for ent in entities:
            log.info(f"Entity assigned: {ent}")
        log.info(f"It is {requires_context} that we require context.")

        return entities, requires_context


# ==============================================================================
# Sample bank (Chroma)
# ==============================================================================

class SampleBank:
    def __init__(self, dsl: DSLInterface, persist_dir: Optional[str] = None):
        self.dsl = dsl
        persist_dir = persist_dir or _default_chroma_dir()
        os.makedirs(persist_dir, exist_ok=True)

        client = None
        err_new = err_old = None

        # New API
        if ChromaPersistentClient is not None:
            try:
                client = ChromaPersistentClient(path=persist_dir)
            except Exception as e:
                err_new = e
                client = None

        # Legacy API
        if client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_dir
                ))
            except Exception as e:
                err_old = e

        if client is None:
            raise RuntimeError(
                "Failed to initialize Chroma client.\n"
                f"  New API error: {err_new}\n"
                f"  Legacy API error: {err_old}\n"
                f"  persist_dir={persist_dir}"
            )

        self.client = client

        # Collection
        name = dsl.get_collection_name()
        try:
            if hasattr(self.client, "get_or_create_collection"):
                self.collection = self.client.get_or_create_collection(name)
            else:
                try:
                    self.collection = self.client.get_collection(name)
                except Exception:
                    self.collection = self.client.create_collection(name)
        except Exception as e:
            raise RuntimeError(f"Could not open/create Chroma collection '{name}': {e}")

        # Embedding function (we pass query embeddings directly)
        base_url = os.environ.get("OLLAMA_HOST")
        self.embed_query = OllamaEmbeddings(
            model="nomic-embed-text",
            **({"base_url": base_url} if base_url else {})
        ).embed_query

    # ---- retrieval helpers ----------------------------------------------------

    def _count_utterances_and_subsamples_raw(self, metadatas: List[Dict[str, Any]]) -> Tuple[int, int]:
        def parse_sub_samples(ss):
            if isinstance(ss, str):
                try: return json.loads(ss)
                except json.JSONDecodeError: return []
            return ss if isinstance(ss, list) else []
        num_utter = len(metadatas or [])
        num_subs = sum(len(parse_sub_samples(m.get("sub_samples", []))) for m in (metadatas or []))
        return num_utter, num_subs

    def _entity_in_subsamples(self, entity: str, meta: Dict[str, Any]) -> bool:
        ss = meta.get("sub_samples", [])
        if isinstance(ss, str):
            try: ss = json.loads(ss)
            except json.JSONDecodeError: ss = []
        for s in ss:
            if entity in (s.get("entities") or []):
                return True
        return False

    def _filter_entity_results_raw(self, metadatas, distances, allowed_entities):
        processed = {"metadatas": [], "distances": []}
        for meta, dist in zip(metadatas, distances):
            ss = meta.get("sub_samples", [])
            if isinstance(ss, str):
                try: ss = json.loads(ss)
                except json.JSONDecodeError: ss = []
            filtered_ss = [s for s in ss if any(e in allowed_entities for e in (s.get("entities") or []))]
            if filtered_ss:
                processed["metadatas"].append({"utterance": meta.get("utterance"), "sub_samples": filtered_ss})
                processed["distances"].append(dist)
        return processed

    def _context_match(self, stored_ctx: dict, current_ctx: dict) -> float:
        if not stored_ctx or not isinstance(current_ctx, dict):
            return 0.0
        if isinstance(stored_ctx, str):
            try: stored_ctx = json.loads(stored_ctx)
            except json.JSONDecodeError: return 0.0
        # exact structural containment
        for k, v in stored_ctx.items():
            if isinstance(v, dict):
                if not self._context_match(v, current_ctx.get(k, {})):
                    return 0.0
            elif current_ctx.get(k) != v:
                return 0.0
        alog.debug(f"Successful match for stored_ctx: {stored_ctx}")
        return 1.0

    def _get_best_submatch(self, samples, context: dict):
        best, best_score = [], -1.0
        for ss in samples:
            score = self._context_match(ss.get("context"), context)
            if score > best_score:
                best, best_score = [ss], score
            elif score == best_score:
                best.append(ss)
        return best

    def _subsample_match_context(self, filtered, context: dict):
        processed = {"metadatas": [], "distances": []}
        for meta, dist in zip(filtered["metadatas"], filtered["distances"]):
            best = self._get_best_submatch(meta.get("sub_samples", []), context)
            if best:
                processed["metadatas"].append({"utterance": meta.get("utterance"), "sub_samples": best})
                processed["distances"].append(dist)
        return processed

    # ---- main retrieval -------------------------------------------------------

    def entity_aware_search(
        self,
        utterance: str,
        entities: List[str],
        k: int = 5,
        context: Optional[Dict] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        # Normalize + embed
        norm = self.dsl.normalize_text(utterance)
        emb = self.embed_query(norm)

        results = self.collection.query(
            query_embeddings=[emb],
            n_results=100,
            include=["metadatas", "distances"],
        )

        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        if metadatas and isinstance(metadatas[0], list): metadatas = metadatas[0]
        if distances and isinstance(distances[0], list): distances = distances[0]

        num_utts, num_subs = self._count_utterances_and_subsamples_raw(metadatas)
        alog.tmp = f"Utterances: {num_utts}, Sub-samples: {num_subs}\n        "

        filtered = self._filter_entity_results_raw(metadatas, distances, entities)
        num_utts, num_subs = self._count_utterances_and_subsamples_raw(filtered["metadatas"])
        alog.tmp += f"Utterances: {num_utts}, Sub-samples: {num_subs} after entity filter\n        "

        if context:
            filtered = self._subsample_match_context(filtered, context)

        num_utts, num_subs = self._count_utterances_and_subsamples_raw(filtered["metadatas"])
        alog.debug(alog.tmp + f"Utterances: {num_utts}, Sub-samples: {num_subs} after context match")

        # One per entity, then fill
        seen = set()
        final: List[Tuple[float, Dict[str, Any], str]] = []
        target = max(len(entities) or 1, k)

        for ent in (entities or ["_"]):
            log.info(f"Determining matches for entity: {ent}")
            for meta, dist in zip(filtered["metadatas"], filtered["distances"]):
                sid = meta.get("utterance", "")
                if sid in seen:
                    continue
                if ent == "_" or self._entity_in_subsamples(ent, meta):
                    final.append((dist, meta, sid))
                    seen.add(sid)
                    break

        for meta, dist in zip(filtered["metadatas"], filtered["distances"]):
            if len(final) >= target: break
            sid = meta.get("utterance", "")
            if sid in seen: continue
            final.append((dist, meta, sid))
            seen.add(sid)

        if final:
            print("🔍 Top retrieved samples:")
            for i, (score, m, _) in enumerate(final):
                print(f"  [{i+1}] score={score:.6f} | '{m.get('utterance','')}'")
        else:
            print("No matching samples!!!")

        return [(d, m) for (d, m, _) in final]


# ==============================================================================
# Prompt construction
# ==============================================================================

class PromptConstructor:
    def __init__(self, dsl: DSLInterface, sample_bank: SampleBank):
        self.dsl = dsl
        self.bank = sample_bank

    def build_prompt(self, utterance: str, context: Optional[Dict], entities: List[str], k: int = 5) -> Dict[str, str]:
        sys_parts = [
            "### This is a DSL for generating actions in a software. Generate only valid DSL code. "
            "Respond only with DSL code. Do not include markdown or explanations.",
            "### Here are examples of the DSL syntax:",
            self.dsl.get_syntax(entities),
            "### Generate a DSL program to fulfill the given user utterance. "
            "Follow these rules:",
            self.dsl.get_rules(entities),
        ]

        usr_parts: List[str] = []
        samples = self.bank.entity_aware_search(utterance, entities, k, context)
        if samples:
            usr_parts.append("### Below are relevant sample instructions:")
            for score, ex in samples:
                sub_samples = ex.get("sub_samples", [])
                if isinstance(sub_samples, str):
                    try: sub_samples = json.loads(sub_samples)
                    except json.JSONDecodeError: sub_samples = []
                for ss in sub_samples:
                    usr_parts.append(f"\n# User Instruction: {ex.get('utterance','')}  # sim={score:.3f}")
                    if ss.get("context"):
                        usr_parts.append(f"# Context: {ss['context']}")
                    usr_parts.append(ss.get("program","").strip())

        usr_parts.append(f"\n# User Utterance to implement: {utterance}")
        if context:
            usr_parts.append(f"# Context: {context}")
        usr_parts.append("### DSL Program:")
        return {"system": "\n".join(sys_parts), "user": "\n".join(usr_parts)}


# ==============================================================================
# Program synthesis
# ==============================================================================

class ProgramSynthesizer:
    def __init__(self, llm: LLMClient, dsl: DSLInterface, temperature: float = 0.0, max_tokens: int = 512):
        self.llm = llm
        self.dsl = dsl
        self.temperature = temperature
        self.max_tokens = max_tokens

    def synthesize(self, prompt: Dict[str, str]) -> str:
        if loglevel == logging.DEBUG:
            self.llm.set_debug(True)

        raw = self.llm.call(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        lines = self.dsl.extract_dsl_lines(raw or "")
        if lines:
            return "\n".join(lines).strip()
        print("⚠️ Warning: No valid DSL lines detected, returning raw LLM output.")
        return (raw or "").strip()


# ==============================================================================
# DSL Parse / Unparse
# ==============================================================================

class DSLParser:
    def parse(self, program_text: str) -> List[Dict[str, Any]]:
        ast: List[Dict[str, Any]] = []
        for line in (program_text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("await context."):
                continue
            m = re.match(r"(?:(\w+)\s*=\s*)?(\w+)\((.*)\)", line)
            if not m:
                continue

            var, stmt, argstr = m.groups()
            args: Dict[str, Any] = {}

            parts, buf, in_q, qch, depth = [], [], False, None, 0
            for ch in argstr:
                if ch in ('"', "'") and not in_q:
                    in_q, qch = True, ch
                elif in_q and ch == qch:
                    in_q, qch = False, None
                elif not in_q and ch == '[':
                    depth += 1
                elif not in_q and ch == ']':
                    depth -= 1
                if ch == ',' and not in_q and depth == 0:
                    parts.append(''.join(buf).strip()); buf = []
                else:
                    buf.append(ch)
            if buf:
                parts.append(''.join(buf).strip())

            for p in parts:
                if not p or '=' not in p:
                    continue
                k, v = p.split('=', 1)
                k, v = k.strip(), v.strip()
                if v.startswith('[') and v.endswith(']'):
                    try: val = json.loads(v)
                    except Exception: val = v
                elif (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    val = v[1:-1]
                elif v.lower() in ("true","false"):
                    val = v.lower() == "true"
                else:
                    try: val = float(v) if '.' in v else int(v)
                    except Exception: val = v
                args[k] = val

            ast.append({"var": var, "stmt": stmt, "args": args})
        return ast

    def unparse(self, ast: List[Dict[str, Any]]) -> str:
        lines = []
        for node in ast:
            var, stmt, args = node.get("var"), node.get("stmt"), node.get("args", {})
            kv = []
            for k, v in args.items():
                if isinstance(v, bool):
                    sv = "true" if v else "false"
                elif isinstance(v, (int, float)):
                    sv = str(v)
                elif isinstance(v, list):
                    sv = "[" + ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v) + "]"
                else:
                    sv = f'"{v}"'
                kv.append(f"{k}={sv}")
            argstr = ", ".join(kv)
            lines.append(f"{(var + ' = ') if var else ''}{stmt}({argstr})")
        return "\n".join(lines)


# ==============================================================================
# DSL Corrections
# ==============================================================================

class CodeCorrector:
    def __init__(self, dsl: DSLInterface):
        self.dsl = dsl

    def correct(self, ast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stmt_aliases = self.dsl.get_statement_aliases()
        valid_enums  = self.dsl.get_valid_enums()
        out: List[Dict[str, Any]] = []

        for node in ast:
            stmt = node.get("stmt", "")

            # alias normalization
            if stmt in stmt_aliases:
                node["stmt"] = stmt_aliases[stmt]
                stmt = node["stmt"]

            # enum normalization
            for arg, val in list(node.get("args", {}).items()):
                if arg in valid_enums and isinstance(val, str):
                    match = difflib.get_close_matches(val, valid_enums[arg], n=1)
                    if match:
                        node["args"][arg] = match[0]

            # DSL-specific corrections
            node = self.dsl.additional_corrections(node)
            if node is None:
                print("[INFO] Command ignored after DSL corrections.")
                continue
            if node is None:
                print("[INFO] Command ignored after DSL corrections.")
                continue
            if node is None:
                print("[INFO] Command ignored after DSL corrections.")
                continue

            # scope validation (only for select_* office-style)
            node = self.dsl.validate_scope(node)
            if node is None:
                print("[INFO] Command was ignored after verification of command scopes.")
                continue

            out.append(node)
        return out


# ==============================================================================
# Orchestrator
# ==============================================================================

class SemanticInterpreter:
    def __init__(self, dsl: DSLInterface, llm: LLMClient, chroma_path: Optional[str] = None):
        """
        Args:
          dsl: DSLInterface implementation
          llm: LLM client
          chroma_path: optional override for Chroma persistence directory
        """
        persist_dir = chroma_path or getattr(dsl, "CHROMA_PATH", None) or _default_chroma_dir()

        self.sample_bank   = SampleBank(dsl, persist_dir)
        self.classifier    = EntityContextClassifier(llm, dsl)
        self.prompt_builder = PromptConstructor(dsl, self.sample_bank)
        self.synth         = ProgramSynthesizer(llm, dsl)
        self.parser        = DSLParser()
        self.corrector     = CodeCorrector(dsl)
        self.dsl           = dsl

    def interpret(
        self,
        utterance: str,
        entity_hint: Optional[Union[str, List[str]]] = None,
        context: Optional[Dict] = None
    ) -> InterpretResult:
        """
        Return an InterpretResult with fields:
          - program / cleaned_program (attributes)
          - "dsl", "ast", "cleaned_dsl" (dict keys)
        """
        if entity_hint:
            entities = [entity_hint] if isinstance(entity_hint, str) else list(entity_hint)
            needs_ctx = False
        else:
            entities, needs_ctx = self.classifier.classify(utterance, context)

        filtered_ctx = self.dsl.prune_context_tree(context, entities) if needs_ctx else None
        prompt = self.prompt_builder.build_prompt(utterance, filtered_ctx, entities or [], k=5)

        dsl_prog = self.synth.synthesize(prompt)
        ast = self.parser.parse(dsl_prog)
        corrected_ast = self.corrector.correct(ast)
        cleaned = self.parser.unparse(corrected_ast)

        # Final adapter for the validator grammar (positional enums, hide_all, etc.)
        if hasattr(self.dsl, "postprocess_for_validator"):
            cleaned = self.dsl.postprocess_for_validator(cleaned, utterance=utterance)

        return InterpretResult(dsl=dsl_prog, ast=corrected_ast, cleaned=cleaned)


# ==============================================================================
# Library only
# ==============================================================================

if __name__ == "__main__":
    print("semantic_interpreter.py is a library. Use script/mol_repl.py to run the REPL.")

