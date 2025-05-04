# semantic_interpreter.py

"""
Semantic Interpreter approach (Gandhi et al. 2023)
Based on program synthesis from natural language using Ollama and LangChain (nomic-embed-text).
DSL-agnostic implementation with pluggable DSL interface.

Includes:
  - Entity and context analysis
  - Example-based retrieval with semantic search
  - Prompt construction (ARM)
  - Program synthesis via Ollama
  - DSL parsing, correction and back/forth transformation to AST

Tone: pragmatic, minimalistic, and extensible.
"""
import re
import os
import sys
import difflib
import json
import inspect
from typing import List, Dict, Any, Optional, Tuple, Union

#from conda.testing import tmp_pkgs_dir

from utils import load_config
# Load env config
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(script_dir + '/../config/env')

if (OLLAMA_HOST := config.get('OLLAMA_HOST')):
    print(f"Setting OLLAMA_HOST to {OLLAMA_HOST}")
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST


from utils import get_logger
import logging
#loglevel=logging.DEBUG
loglevel=logging.INFO
log = get_logger(level=loglevel)
# trick to create a second logger that will log everything
alog = get_logger(name="d",level=logging.DEBUG)

# --- Use permanent ChromaDB store ---
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
# --- Embedding via LangChain OllamaEmbeddings ---
from langchain_community.embeddings.ollama import OllamaEmbeddings

# --- Prompting via LangChain Ollama ---
from llm_client import LLMClient

# Import the DSL interface
from dsl_interface import DSLInterface




# --- Entity classification and context dependency ---
class EntityContextClassifier:
    def __init__(self, llm: LLMClient, dsl: DSLInterface):
        """
        Entity and context classifier based on an LLM.
        Determines:
        - Which semantic entities are involved in the instruction
        - Whether resolving the instruction requires external context
        """
        self.llm = llm
        self.dsl = dsl

        
    def classify(self, utterance: str, context: Optional[Dict]) -> Tuple[List[str], bool]:
        """
        Classifies the user instruction into semantic categories
        and decides if additional context (e.g., scene state) is required.

        Returns:
        - A list of entity categories involved
        - A boolean indicating if context is required
        """

        if loglevel==logging.DEBUG: self.llm.set_debug(True)

        log.tmp = f"User query: {utterance}\n       "
        if context:
            log.info(log.tmp + f"Context provided: {context}")
        else:
            log.info(log.tmp + f"No context provided!")

    # Step 1: Ask the LLM to classify the utterance
        result = self.llm.call_with_template(
            system_template=self.dsl.get_sysprompt_entity_context(),
            user_template=self.dsl.get_usrprompt_entity_context(),
            utterance=utterance
        )

        log.debug(f"=== LLM returned: ===\n\n{result}\n\n[DEBUG] === END ===\n\n")
        
        # Step 2: Extract entity list from the LLM output
        cats = re.search(r"Categories:\s*\[(.*?)\]", result)

        # Step 3: Extract context requirement flag from output
        ctx = re.search(r"RequiresContext:\s*(true|false)", result, re.IGNORECASE)

        # Step 4: Parse entity names and clean up formatting
        entities = [e.strip().strip('"\'') for e in cats.group(1).split(',')] if cats else []

        log.tmp = ""
        for ent in entities: log.tmp += f"Entity assigned: {ent}\n       "

        # Step 5: Convert string to boolean, default to False if missing
        requires_context = ctx.group(1).lower() == 'true' if ctx else False
        log.info(log.tmp + f"It is {requires_context} that we require context.")
        
        return entities, requires_context


# --- Example-based retrieval (entity-aware semantic search) ---
class SampleBank:
    def __init__(self, dsl: DSLInterface, chroma_path: str):
        """
        Load a persistent ChromaDB collection for example retrieval.
        """
        self.dsl = dsl
        self.client = PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(dsl.get_collection_name())
        self.embed_fn = OllamaEmbeddings(
            model="nomic-embed-text",
            **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
        ).embed_query


    def entity_aware_search(
        self,
        utterance: str,
        entities: List[str],
        k: int = 5,
        context: dict = None,
        utterance_weight: float = 0.7,
        context_weight: float = 0.3
            ) -> List[Dict[str, Any]]:
        """
        Perform an enhanced semantic search among the example bank with context-aware reranking:
        - For each entity, find its single best match
        - Add additional best matches until we reach max(len(entities), k) total
        - Use semantic distance as the score (lower is better)

        Args:
            utterance: The user utterance to search with
            entities: List of entities to match
            k: Number of additional best matches to retrieve
            context: The overall context
            utterance_weight: 0-1 weight for utterance similarity
            context_weight: 0-1 weight for context similarity

        Returns:
            List of tuples (score, metadata) for the top matches
        """
        if loglevel==logging.DEBUG: self.dsl.set_debug(True)

        #alog.info(f"NORMALIZING UTTERANCE {utterance}")
        norm = self.dsl.normalize_text(utterance) # normalize the utterance for search
        #alog.info(f"NORMALIZED TO         {norm}")
        emb = self.embed_fn(norm) # compare against embeddings

        # Dictionary to keep track of seen metadata entries to avoid duplicates
        seen_samples = set()
        final_matches = []
        target_count = max(len(entities), k) # get at least one sample per entity and k samples

        # First get all results since we can't use $contains
        entity_results = self.collection.query(
            query_embeddings=[emb],
            n_results=100,  # Get enough to filter client-side
            include=["metadatas", "distances"]
        )
        
        # How many raw matches do we have?
        num_utts, num_subs = self._count_utterances_and_subsamples(entity_results)
        alog.tmp = f"Utterances: {num_utts}, Sub-samples: {num_subs}\n        "

        # Filter to keep only entries matching the entities list
        entity_results = self._filter_entity_results(entity_results, entities)

        # How many matches do we have after filtering by entity?
        num_utts, num_subs = self._count_utterances_and_subsamples(entity_results)
        alog.tmp += f"Utterances: {num_utts}, Sub-samples: {num_subs} after entity filter\n        "

        if context:
            #alog.debug(f"We have Context: {context}")
            # Filter out sub-samples as well based on matching context
            entity_results = self._subsample_match_context(entity_results, context)

        # How many matches do we have after filtering by context?
        num_utts, num_subs = self._count_utterances_and_subsamples(entity_results)
        alog.debug(alog.tmp + f"Utterances: {num_utts}, Sub-samples: {num_subs} after context match")

        # PART 1: Get single best match for each entity
        for entity in entities:
            log.tmp= f"Determining matches for entity: {entity}\n       "
            # Filter client-side for this entity
            for i, meta in enumerate(entity_results["metadatas"]):
#                alog.debug(f"META IS {meta}")
                # Create a unique identifier for this sample
                sample_id = meta["utterance"]  # Using utterance as a unique ID

                if self._entity_in_subsamples(entity, meta) and sample_id not in seen_samples:
                    # We found our best match for this entity
                    final_matches.append((
                        entity_results["distances"][i],
                        meta,
                        sample_id
                    ))
                    seen_samples.add(sample_id)
                    log.tmp += f"  retained sample for entity {i} : {entity_results["distances"][i]} distance score\n       "
                    break

        # Continue if we do not have enough matches yet
        if len(final_matches) < target_count:
            log.tmp += f"Add more matches from pool, ignoring entity constraint\n       "
            # PART 2: Add additional matches to reach the target count
            for meta, dist in zip(
                entity_results["metadatas"], 
                entity_results["distances"]
            ):
                # Create a unique identifier for this sample
                sample_id = meta["utterance"]  # Using utterance as a unique ID

                if sample_id in seen_samples:
                    continue # is already in our list

                # Use distance directly as the score
                final_matches.append((dist, meta, sample_id))
                seen_samples.add(sample_id)
                log.tmp += f"  retained additional sample: {dist} distance score / {sample_id}\n       "

                # Stop once we've reached the target count
                if len(final_matches) >= target_count:
                    break
        log.info(log.tmp)
        
        # Log results
        print("🔍 Top retrieved samples:") if final_matches else print("No matching samples!!!")
        for i, (score, m, _) in enumerate(final_matches):
            print(f"  [{i+1}] score={score:.6f} | '{m['utterance']}'")

        # Return just score and metadata without the ID
        return [(score, meta) for score, meta, _ in final_matches]

 # --- Helper functions for entity and context filtering ---
    def _count_utterances_and_subsamples(self, entity_results):
        """Count the number of utterances and sub-samples in the entity results"""
        metadatas = entity_results["metadatas"]
    
        # Unwrap nested list if needed
        if isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
    
        def parse_sub_samples(sub_samples):
            if isinstance(sub_samples, str):
                try:
                    return json.loads(sub_samples)
                except json.JSONDecodeError:
                    return []
            elif isinstance(sub_samples, list):
                return sub_samples
            return []
    
        num_utterances = len(metadatas)
        total_sub_samples = sum(len(parse_sub_samples(m.get("sub_samples", []))) for m in metadatas)
    
        return num_utterances, total_sub_samples

    def _entity_in_subsamples(self, entity, meta):
        """Check if a given entity is present in any of the subsamples"""
        sub_samples = meta.get("sub_samples", [])
        for sub_sample in sub_samples:
            entities = sub_sample.get("entities", [])
            if entity in entities:
                return True
        return False


    def _filter_entity_results(self, entity_results, allowed_entities):
        """Filter entity results based on allowed entities"""
        processed_entity_results = {
            "metadatas": [],
            "distances": []
        }

        for metadata, distance in zip(entity_results["metadatas"][0], entity_results["distances"][0]):
#            alog.info(f"\n>>>>>\nMETADATA: {metadata}\nDISTANCE: {distance}\n")
            # Deserialize sub_samples
            sub_samples = json.loads(metadata["sub_samples"])

            # Keep only sub_samples that match at least one allowed entity
            filtered_sub_samples = []
            for sub_sample in sub_samples:
                sample_entities = sub_sample["entities"]
                if any(entity in allowed_entities for entity in sample_entities):
                    filtered_sub_samples.append(sub_sample)

            # Only keep if there are filtered sub_samples
            if filtered_sub_samples:
                processed_entity_results["metadatas"].append({
                    "utterance": metadata["utterance"],
                    "sub_samples": filtered_sub_samples
                })
                processed_entity_results["distances"].append(distance)

        return processed_entity_results


    def _subsample_match_context(self, entity_results, context: dict):
        """Filter entity_results based on matching context"""
        processed_entity_results = {
            "metadatas": [],
            "distances": []
        }

        #alog.debug(f"In function {inspect.currentframe().f_code.co_name}")

        for metadata, distance in zip(entity_results["metadatas"], entity_results["distances"]):
            best_submatch = self._get_best_submatch(metadata["sub_samples"], context)
                    
            # Only keep if there are filtered sub_samples
            if best_submatch:
                processed_entity_results["metadatas"].append({
                    "utterance": metadata["utterance"],
                    "sub_samples": best_submatch
                })
                processed_entity_results["distances"].append(distance)

        return processed_entity_results


    def _get_best_submatch(self, samples, context: dict):
        """Select the best sub-sample based on context match, falling back to all samples if no match"""
        best_submatch = []
        best_ctx_score = -1

        for sub_sample in samples:
            ctx_score = self._context_match( # Score context match
                sub_sample.get("context"),
                context
                )
            if ctx_score > best_ctx_score: # We have a new best match
                best_submatch = [sub_sample]
                best_ctx_score = ctx_score
            elif ctx_score == best_ctx_score: # We have one more identical match
                best_submatch.append(sub_sample)

        return best_submatch


    def _context_match(self, stored_ctx: dict, current_ctx: dict) -> float:
        """Binary match (1=exact match, 0=no match) for key conditions"""
        NO_MATCH=0.0 # set lower to -1.0 to disable fallback
        
        if not stored_ctx: # early exit if empty
            return NO_MATCH

        # Attempt to deserialize if stored_ctx is a string, which should never happen
        if isinstance(stored_ctx, str):
            log.warn(f"Dict is string {stored_ctx}!")
            try:
                stored_ctx = json.loads(stored_ctx)
            except json.JSONDecodeError:
                log.warn(f"Failed to decode stored_ctx: {stored_ctx}!")
                return NO_MATCH

        if not isinstance(current_ctx, dict):
            log.warn(f"current_ctx not dict: {current_ctx} ({type(current_ctx)})!")
            return NO_MATCH

        #alog.debug(f"Comparing stored_ctx with current_ctx:\n{stored_ctx}\n{current_ctx}")
        
        for k, v in stored_ctx.items():
            if isinstance(v, dict):
                if not self._context_match(v, current_ctx.get(k, {})):
                    return NO_MATCH
            elif current_ctx.get(k) != v:
                return NO_MATCH
        alog.debug(f"Successful match for stored_ctx: {stored_ctx}")
        return 1.0

# --- Prompt construction with syntax, rules, and examples ---
class PromptConstructor:
    def __init__(self, dsl: DSLInterface, sample_bank: SampleBank):
        self.dsl = dsl
        self.bank = sample_bank

    # build the prompt for program synthesis
    def build_prompt(self,
                     utterance: str,
                     context: [Dict],
                     entities: List[str],
                     k: int = 5) -> str:

        usr_parts = []
        # Start the prompt with a general instruction, syntax and rules
        sys_parts = [
            "### This is a DSL for generating actions in a software. Generate only valid DSL code. Respond only with DSL code. Do not include markdown (e.g., ```dsl) or comments. Do not output JavaScript or explanations.",
            '### Here are examples of the DSL syntax:',
            self.dsl.get_syntax(entities),
            '### Generate a DSL program to fulfill the given user utterance. Remember to follow the following rules when generating DSL:',
            self.dsl.get_rules(entities)
        ]

        #alog.debug(f"In function {inspect.currentframe().f_code.co_name}")
        #alog.debug(f"Context: {context}")
        
        # retrieve samples that match entities and utterance
        samples = self.bank.entity_aware_search(utterance, entities, k, context)
        if samples:
            log.debug("including samples in the prompt")
            usr_parts.append('### Below are relevant sample instructions:')

        # append samples to prompt
        for score, ex in samples:
            for sub_sample in ex["sub_samples"]:
                usr_parts.append(f"\n# User Instruction: {ex['utterance']}  # score={score:.3f}")
                if sub_sample.get('context'):
                    usr_parts.append(f"# Context: {sub_sample['context']}")
                usr_parts.append(sub_sample['program'].strip())

        # append utterance to prompt
        usr_parts.append(f"\n# User Utterance to implement: {utterance}")

        # optionnally append context to prompt
        if context:
            usr_parts.append(f"# Context: {context}")

        usr_parts.append("### DSL Program:")
        return {"system": "\n".join(sys_parts), "user": "\n".join(usr_parts)}


# --- DSL synthesis via LLM ---
class ProgramSynthesizer:
    def __init__(self, llm: LLMClient, dsl: DSLInterface, temperature: float = 0.0, max_tokens: int = 512):
        # Store the LLM client and generation parameters
        self.llm = llm
        self.dsl = dsl
        self.temperature = temperature
        self.max_tokens = max_tokens

    # Given a synthesis prompt (including syntax, rules, examples, and
    # a user instruction), generate a DSL program as a string
    def synthesize(self, prompt: str) -> str:
        if loglevel==logging.DEBUG: self.llm.set_debug(True)

        # Call the language model with the complete synthesis prompt
        raw = self.llm.call(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Extract valid DSL lines from the raw output (may contain JS, comments, etc.)
        dsl_lines = self.dsl.extract_dsl_lines(raw)

        # If DSL lines were successfully extracted, return them
        if dsl_lines:
            return "\n".join(dsl_lines).strip()
        
        # If no valid DSL lines were found, return the raw output as a
        # fallback Log a warning — this likely indicates hallucinated
        # output or failure to follow instructions
        print("⚠️ Warning: No valid DSL lines detected, returning raw LLM output.")
        return raw.strip()


# --- DSL Parsing ---
class DSLParser:
    def parse(self, program_text: str) -> List[Dict[str, Any]]:
        # Initialize the AST (abstract syntax tree) as a list of instructions
        ast = []

        # Iterate over each line of the input DSL program
        for line in program_text.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                print(f"⚠️ Skipping line: {line}")
                continue

            # Skip already transpiled JavaScript lines
            if line.startswith("await context."):
                print(f"⚠️ Skipping line: {line}")
                continue

            # Match lines of the form:
            #     var = statement(args)   ← with optional variable assignment
            m = re.match(r"(?:(\w+)\s*=\s*)?(\w+)\((.*)\)", line)
            if not m:
                print(f"⚠️ No match in line: {line}")
                continue

            var, stmt, argstr = m.groups()
            args = {}

            # Improved argument parsing that handles commas in strings and lists
            # First split by commas not inside quotes or brackets
            parts = []
            current_part = []
            in_quotes = False
            in_brackets = 0
            quote_char = None

            for char in argstr:
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                elif char == '[' and not in_quotes:
                    in_brackets += 1
                elif char == ']' and not in_quotes:
                    in_brackets -= 1
                elif char == ',' and not in_quotes and in_brackets == 0:
                    parts.append(''.join(current_part).strip())
                    current_part = []
                    continue
                current_part.append(char)

            if current_part:
                parts.append(''.join(current_part).strip())

            for part in parts:
                # Skip empty strings (may occur after a trailing comma)
                if not part:
                    print(f"⚠️ Skipping empty arguments.")
                    continue

                # Each argument must be a key=value pair
                if '=' not in part:
                    print(f"⚠️ Skipping invalid argument: {part}")
                    continue

                # Split key and value
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip()

                # Handle list values
                if v.startswith('[') and v.endswith(']'):
                    try:
                        # Use json.loads to properly parse the list
                        val = json.loads(v)
                    except json.JSONDecodeError:
                        print(f"⚠️ Could not parse list: {v}")
                        val = v
                # Handle string values
                elif (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    val = v[1:-1]
                # Handle boolean values
                elif v.lower() in ('true', 'false'):
                    val = v.lower() == 'true'
                # Handle numeric values
                else:
                    try:
                        if '.' in v:
                            val = float(v)
                        else:
                            val = int(v)
                    except ValueError:
                        val = v  # Leave as string if conversion fails

                args[k] = val

            # Append one instruction to the AST
            ast.append({'var': var, 'stmt': stmt, 'args': args})

        return ast

    def unparse(self, ast: List[Dict[str, Any]]) -> str:
        lines = []

        for instr in ast:
            var = instr.get('var')
            stmt = instr.get('stmt')
            args = instr.get('args', {})

            # Format each argument: key=value
            formatted_args = []
            for k, v in args.items():
                # Convert Python types to string representations
                if isinstance(v, bool):
                    val = 'true' if v else 'false'
                elif isinstance(v, (int, float)):
                    val = str(v)
                elif isinstance(v, list):
                    # Handle lists by converting to proper DSL format without extra quotes
                    items = ', '.join(f'"{item}"' if isinstance(item, str) else str(item)
                                    for item in v)
                    val = f'[{items}]'
                else:
                    val = f'"{v}"'  # Enclose strings in quotes
                formatted_args.append(f"{k}={val}")

            # Join all arguments into a single string
            argstr = ', '.join(formatted_args)

            # Construct the full DSL line
            if var:
                line = f"{var} = {stmt}({argstr})"
            else:
                line = f"{stmt}({argstr})"

            lines.append(line)

        return '\n'.join(lines)

# --- DSL Correction ---
class CodeCorrector:
    def __init__(self, dsl: DSLInterface):
        self.dsl = dsl

    def correct(self, ast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        statement_aliases = self.dsl.get_statement_aliases()
        valid_enums = self.dsl.get_valid_enums()

        corrected_ast = []  # <- on construit un nouvel AST propre

        for node in ast:
            stmt = node['stmt']

            # Step 1: Unify statement names by replacing known aliases with the correct form
            if stmt in statement_aliases:
                node['stmt'] = statement_aliases[stmt]

            # Step 2: Correct enum values via fuzzy matching
            for arg, val in node['args'].items():
                # Only correct string values for arguments that are known enumerations
                if arg in valid_enums and isinstance(val, str):
                    # Use difflib to find the closest valid match
                    match = difflib.get_close_matches(val, valid_enums[arg], n=1)
                    if match:
                        node['args'][arg] = match[0]  # Replace with corrected value

            # Step 3: Apply any other DSL-specific corrections
            node = self.dsl.additional_corrections(node)

            # Step 4: Apply any other DSL-specific corrections
            node = self.dsl.validate_scope(node)
            if node is None:
                # commande ignorée
                print(f"[INFO] Command was ignored after verification of command scopes.")
                continue  # on ne l'ajoute pas
            else:
                corrected_ast.append(node)  # <- on ajoute uniquement les bons nodes

        # Return the corrected AST
        return corrected_ast



# --- Interpreter Orchestrator ---
class SemanticInterpreter:
    def __init__(self, llm: LLMClient, dsl: DSLInterface):
        """
        Initialize the semantic interpreter with a specific DSL implementation.
        
        Args:
            llm: LLM client for language processing
            dsl: Implementation of DSLInterface for the target domain
        """
        # Get the chroma path from the DSL
        chroma_path = dsl.CHROMA_PATH if hasattr(dsl, 'CHROMA_PATH') else None
        
        # Initialize components with the DSL implementation
        self.sample_bank = SampleBank(dsl, chroma_path)

        # delegate the classification to this function
        self.classifier = EntityContextClassifier(llm, dsl)

        # delegate prompt construction to this function        
        self.prompt_builder = PromptConstructor(dsl, self.sample_bank)

        # delegate program synthesis to this function
        self.synth = ProgramSynthesizer(llm, dsl)
        
        # delegate DSL parsing to this function
        self.parser = DSLParser()

        # delegate DSL correction to this function
        self.corrector = CodeCorrector(dsl)
        self.dsl = dsl

    def interpret(self, utterance: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Interpret a user utterance, optionally with context.
        Returns interpretation to DSL and derived AST.
        
        Args:
            utterance: User's natural language request
            context: Optional context information
            
        Returns:
            Dictionary containing DSL program and AST
        """
        # Call a classifier on utterance to extract entities and decide if context is needed
        entities, needs_ctx = self.classifier.classify(utterance, context)

        alog.tmp = f"in function {inspect.currentframe().f_code.co_name}, "
        #alog.debug(f"Do we need context? {needs_ctx}")
        
        # If context is needed, we obtain it
        #alog.debug(f"Original Context is: {context}")
        #self.dsl.set_debug(True)
        filtered_ctx = self.dsl.prune_context_tree(context, entities) if needs_ctx else None
        alog.debug(alog.tmp + f"filtered Context is: {filtered_ctx}")
        
        # construct the DSL synthesis prompt based on utterance,
        # context and entities. This should perform a semantic RAG
        # search
        prompt = self.prompt_builder.build_prompt(utterance, filtered_ctx, entities)
        #log.debug(f"=== Prompt is: ===\n\n{prompt}\n\n[DEBUG] === END ===\n\n")
        
        # Actual program synthesis: use the DSL synthesis prompt, call
        # the LLM and generate the DSL code
        dsl_prog = self.synth.synthesize(prompt)

        # parse the DSL code into AST, also validating the code
        ast = self.parser.parse(dsl_prog)

        # attempt automatic code correction
        corrected_ast = self.corrector.correct(ast)

        # Return both the program text and the AST
        return {"dsl": dsl_prog, "ast": corrected_ast}


# --- Main Example Usage ---
if __name__ == "__main__":
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    
    # # Import the DSL implementation
    # sys.path.append(os.path.abspath(script_dir / "../odsl"))
    # from dsl_definition import DSL
    # # Initialize components
    # llm = LLMClient()  # Use Ollama client
    #
    # dsl = DSL()  # Use Office DSL implementation
    #
    # # Initialize interpreter with llm and dsl
    # interpreter = SemanticInterpreter(llm, dsl)
    #
    #
    # # --- Example Code Snippet #1 : SIMPLE USER QUERY ---
    # # Call the interpreter with a user query
    # # Q: where, how, by whom is decided if we provide context? would
    # # be an argument to the interpret() call
    # result = interpreter.interpret("Make the text look formal")
    #
    # print("\n=== DSL ===")
    # print(result["dsl"])
    #
    # print("\n=== AST ===")
    # print(result["ast"])
    #
    # print("\n=== Cleaned DSL ===")
    # print(interpreter.parser.unparse(result["ast"]))


    # # --- Example Code Snippet #2 : TESTING SCOPE VALIDATION ---
    # # FOR TESTING SCOPE VALIDATION
    # # Valid cases
    # interpreter.dsl.validate_scope({"stmt": "select_text", "args": {"scope": "shape"}})  # OK
    # interpreter.dsl.validate_scope({"stmt": "select_slides", "args": {}})  # Injects "Presentation"
    # interpreter.dsl.validate_scope({"stmt": "select_text", "args": {"scope": "document"}})  # → Maps to "Presentation"
    # interpreter.dsl.validate_scope({"stmt": "select_text", "args": {"scope": "shapes"}})   # → Still valid
    #
    # # Previously invalid cases because not explicitly defined (changed that since)
    # interpreter.dsl.validate_scope({"stmt": "select_text", "args": {"scope": "slide"}})    # → Error (not in valid_scopes)
    #
    # # Invalid cases
    # interpreter.dsl.validate_scope({"stmt": "select_shapes", "args": {"scope": 123}})  # Raises (not a string)
    # interpreter.dsl.validate_scope({"stmt": "select_slide", "args": {"scope": "text"}})    # → Error (not in valid_scopes)


    # # --- Example Code Snippet #3 : TESTING CONTEXT PRUNING ---
    # # FOR TESTING CONTEXT PRUNING
    # from utils import load_json_file
    # context = load_json_file("/opt/src/ai/ai-exploration/semantic-interpreter-test/odsl/example-context.json")
    # # # Ex. 2:
    # # context = {
    # #     "id": "pres1",
    # #     "_type": "presentation",
    # #     "slides": [
    # #         {
    # #          "_type": "slide",
    # #          "shapes": [
    # #              {
    # #               "_type": "shape",
    # #               "textRange": {"_type": "textRange", "text": "Dogs"}
    # #               },
    # #              {
    # #               "name": "Content Placeholder 1"  # No textRange
    # #               }
    # #               ]
    # # }
    # # ]
    # # }
    #
    # tmp = json.dumps(context)
    # print(f"\n=== ORIGINAL CONTEXT ===\n\n{tmp}\n")
    # newctx = interpreter.dsl.prune_context_tree(context,['textRange'])
    # tmp = json.dumps(newctx)
    # print(f"=== PRUNED CONTEXT ===\n\n{tmp}\n")
    # if context == newctx: print(">>> BOTH ARE IDENTICAL <<<")


    # # --- Example Code Snippet #4 : TESTING CONTEXT MATCHING ---
    # # Context match case
    # test_ctx = {'slides': [{'has_textbox': True}]}
    # samples = [
    #             {
    #                 "program": "add_text(slide=1, text='Hello')",
    #                 "context": {'slides': [{'has_textbox': True}]}
    #             },
    #             {
    #                 "program": "add_text(slide=2, text='World')",
    #                 "context": {'slides': [{'has_textbox': False}]}
    #             }
    #         ]
    # tmp = interpreter.sample_bank._get_best_submatch(samples, test_ctx)
    # assert tmp[0] == samples[0]
    # print("Context match case tested.")
    #
    # # No match case
    # assert interpreter.sample_bank._get_best_submatch(samples, {}) == samples
    # print("No match case tested.")
    
    # --- Example Code Snippet #5 : CONTEXT PRUNING PLUS CONTEXT-MATCHING SUB-SAMPLE SELECTION ---
    samples = [
        [
            {"stmt": "add_sticks", "args": {"selection": "protein"}},
            {"stmt": "update_color", "args": {"selection": "active_site", "color": "#FF0000"}},
            {"stmt": "delete_surface", "args": {"selection": "solvent"}}
        ],
        [{'var': None, 'stmt': 'add_structure', 'args': {'PDBID': '1kx2'}}],
        [
            {"stmt": "select_structure", "args": {"name": "1CRN"}},
            {"stmt": "add_sticks", "args": {"selection": "resid 1"}},
            {"stmt": "update_color", "args": {"selection": "last()", "color": "#FF0000"}},
            {"stmt": "delete_structure", "args": {}},
            #            {"stmt": "select_structure", "args": {"name": "toto"}},
            {"stmt": "delete_structure", "args": {}}
        ],
        [
            {'stmt': 'add_structure', 'args': {'PDBID': '1kx2'}},
            {'stmt': 'add_structure', 'args': {'filePath': 'toto.pdb'}},
            {'stmt': 'add_structure', 'args': {'PDBID': '1kx2', 'filePath': 'toto.pdb'}},
            {"stmt": "add_structure", "args": {}}
        ],

        [
            {"stmt": "color_selection", "args": {"selection": "all", "color": "#FF0000"}},
            {"stmt": "add_structure", "args": {"PDBID": "1CRN"}},
            {"stmt": "validate_molecule", "args": {"radius": 10}},
            {"stmt": "add_molecule", "args": {"radius": 6, "type": "organic"}},
            {"stmt": "add_molecule", "args": {"radius": 6, "type": "organic"}},
        ]
    ]
    # Import the DSL implementation
    sys.path.append(os.path.abspath(script_dir / "../molcommand"))
    from dsl_definition import DSL

    # Initialize components
    llm = LLMClient()  # Use Ollama client

    dsl = DSL()  # Use Office DSL implementation

    # Initialize interpreter with llm and dsl
    interpreter = SemanticInterpreter(llm, dsl)

    print("\n=== AST ===")
    print(samples[0])

    print("\n=== Cleaned DSL ===")
    print(interpreter.parser.unparse(samples[0]))

# --- Questions ---

# How universal is AST for my purpose? Limitations?
