# -*- coding: utf-8 -*-
import re
import os
import io
import string
from typing import List, Dict, Any, Optional
from dsl_interface import DSLInterface
from utils import load_config, load_text_file, load_yaml_file
from llm_client import load_prompt
import json
import inspect


class DSL(DSLInterface):
    """Implementation of DSL interface for Office automation DSL"""
    
    def __init__(self):
        """Initialize the Office DSL with configs and rules"""
        self.debug_enabled = False
        # Get the absolute path to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the configuration from the config/env file
        config = load_config(script_dir + '/../config/env')

        # Use the configuration in your script
        self.CONFIG_PATH = config.get('CONFIGPATH')
        self.CHROMA_PATH = config.get('CHROMAPATH') + '/odsl/'
        
        # Load syntax, rules and entity context

        syntax_path = script_dir + "/syntax.yaml"
        rules_path = script_dir + "/rules.yaml"
        hierarchy_path = script_dir + "/hierarchy.yaml"
        entity_ctx_path = script_dir
        # Load the files
        self._syntax = load_yaml_file(syntax_path)
        self._rules = load_yaml_file(rules_path)
        self._hierarchy_schema = load_yaml_file(hierarchy_path)
        self._sysprompt_entity_context, self._usrprompt_entity_context = load_prompt("entity_context",entity_ctx_path)


    def extract_key_terms_as_string(self, query: str) -> str:
        """Extract important keywords from the query and return them as a space-separated string."""
        stop_words = {
            # "and", "or", "the", "to", "a", "an", "in", "with", "by", "of", "for",
            # "is", "are", "can", "how", "what", "when", "where", "which", "who", "why"
            "the", "a", "an", "it", "with", "by", "for", "this", "there",
            "is", "are", "can", "how", "what", "when", "where", "which", "who", "why"
        }
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return " ".join(keywords)

 
    def get_syntax(self, entities: List[str]) -> str:
        """Assemble relevant syntax statements for prompt"""
        # Initialize an empty list to hold the formatted sections
        parts = []

        # Add a title for the Global Items section (if any global items exist)
        parts.append("### Global Syntax")
        # Loop through all statement categories to find global items (those with no entity matching)
        for statement_type, statements in self._syntax.items():
            if statement_type == "global_syntax":  # Assuming "global_items" is a special section
                for entry in statements:
                    if isinstance(entry, dict):
                        parts.append("- " + entry["text"])  # Add global item text

        # Add a section title for Select Statements
        parts.append("\n### Select Statements")
        # add syntax for matching entities
        for entry in self._syntax.get('select_statements', []):
            if isinstance(entry, dict):
                if any(entity in entry["entities"] for entity in entities):
                    parts.append(entry["text"])  # Add the matching select statement

        # Add a section title for Insert Statements
        parts.append("\n### Insert Statements")
        for entry in self._syntax.get('insert_statements', []):
            if isinstance(entry, dict):
                if any(entity in entry["entities"] for entity in entities):
                    parts.append(entry["text"])  # Add the matching insert statement

        # Add a section title for Format Statements
        parts.append("\n### Format Statements")
        for entry in self._syntax.get('format_statements', []):
            if isinstance(entry, dict):
                if any(entity in entry["entities"] for entity in entities):
                    parts.append(entry["text"])  # Add the matching format statement

        # Add a section title for Delete Statements
        parts.append("\n### Delete Statements")
        for entry in self._syntax.get('delete_statements', []):
            if isinstance(entry, dict):
                if any(entity in entry["entities"] for entity in entities):
                    parts.append(entry["text"])  # Add the matching delete statement
        parts.append("\n")  # Add a newline after delete statements

        # Combine all the parts into a single string and return it
        return "\n".join(parts)


    def get_rules(self, entities: List[str]) -> str:
        """Construct rules to be added to the prompt"""
        # Initialize an empty list to hold the formatted sections
        parts = []

        # Loop through all statement categories to find global items (those with no entity matching)
        for statement_type, statements in self._rules.items():
            if statement_type == "global_rules":  # Assuming "global_items" is a special section
                for entry in statements:
                    if isinstance(entry, dict):
                        parts.append("- " + entry["text"])  # Add global item text

        # Add a section title for Select Statements
        for entry in self._rules.get('entity_rules', []):
            if isinstance(entry, dict):
                if any(entity in entry["entities"] for entity in entities):
                    parts.append("- " + entry["text"])  # Add the matching select statement

        # Combine all the parts into a single string and return it
        return "\n".join(parts)
    
    
    def get_sysprompt_entity_context(self) -> str:
        """Return the currently stored system prompt for the LLM"""
        return self._sysprompt_entity_context
    
    def get_usrprompt_entity_context(self) -> str:
        """Return the currently stored user prompt for the LLM"""
        return self._usrprompt_entity_context
    
    def extract_dsl_lines(self, raw: str) -> List[str]:
        """
        Extracts valid, complete DSL instruction lines from raw LLM output.
        Handles both single-line and multi-line DSL statements.
        """
        lines = []                 # Collected valid DSL lines
        buffer = ''               # Temporary buffer to accumulate multi-line statements
        inside_statement = False  # Flag to track whether we're inside a multi-line DSL statement

        for line in io.StringIO(raw):  # Iterate over each line of LLM output
            stripped = line.strip()    # Trim leading/trailing whitespace

            # Skip empty lines
            if not stripped:
                continue

            # Detect start of a DSL statement: optional var assignment + DSL function + (
            if re.match(r"^\s*(\w+\s*=)?\s*(select_|insert_|format_|delete_)\w*\(", stripped):
                buffer = stripped
                inside_statement = True

                # If the statement is already complete (ends with `)`), flush it immediately
                if stripped.endswith(')'):
                    lines.append(buffer)
                    buffer = ''
                    inside_statement = False

            # If we're inside a statement, keep appending lines until we close it
            elif inside_statement:
                buffer += ' ' + stripped  # Continue building the statement
                if stripped.endswith(')'):
                    lines.append(buffer)
                    buffer = ''
                    inside_statement = False

            # Not a valid DSL line, and not inside a statement → ignore it
            else:
                print(f"⚠️ Ignoring non-DSL line: {stripped}")

        # If a statement was opened but not closed, warn and drop the buffer
        if buffer:
            print(f"⚠️ Incomplete DSL line left open: {buffer}")

        return lines


    def normalize_text(self, query_text: str) -> str:
        """
        Normalize input text to improve matching.
        Lowercase, remove punctuation, unify synonyms.
        Preprocess the user query to focus on command intent.
        Remove fluff and standardize common terms.
        """
        if self.debug_enabled:
            print(f"[SUB-DGB] Normalizing : {query_text}")
        # Convert to lowercase for better matching
        text = query_text.lower()

        # Replace common phrases with standardized versions
        replacements = {
            "how do i": "",
            "how to": "",
            "i want to": "",
            "can you tell me how to": "",
            "can you": "",
            "please": "",
            "show me how to": "",
            "what is the command for": "",
            "what is the syntax for": "",
            "command to": "",
            "code for": "",
        }

        for phrase, replacement in replacements.items():
            text = text.replace(phrase, replacement)

        # Focus on key terms
        text = self.extract_key_terms_as_string(text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Add specific keywords for better retrieval if missing
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

        for keyword, related_terms in keyword_mappings.items():
            if keyword in text:
                for term in related_terms:
                    if term not in text:
                        text += f" {text}"

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\b(add|insert|create|generate)\b", "add", text)
        text = re.sub(r"\b(red|blue|green|teal|olive)\b", "color", text)
        text = re.sub(r"\b(picture|photo|image)\b", "image", text)
        text = re.sub(r"\btitles?\b", "heading", text)

        if self.debug_enabled:
            print(f"[SUB-DGB]               {text}")

        return text.strip()

    
    def get_statement_aliases(self) -> Dict[str, str]:
        """Maps alias statement names to canonical DSL function names"""
        return {
            'insert_picture': 'insert_images',
            'delete_image': 'delete_shapes'
        }

    
    def get_valid_enums(self) -> Dict[str, List[str]]:
        """Valid enumerated values for specific parameters"""
        return {
            'underline': ['None', 'Single', 'Double', 'Wavy'],
            'horizontalAlignment': ['Left', 'Center', 'Right']
        }

    
    # --- Context-related methods ---
    def prune_context_tree(self, tree: dict, entity_types: list[str]) -> dict:
        """Only retain relevant tree parts for given entity types"""
        if self.debug_enabled:
            print(f"[SUB-DGB] in function {inspect.currentframe().f_code.co_name}")
            # If context is needed, we obtain it
            print(f"[SUB-DBG] Original Context is: {tree}")

        if not isinstance(tree, dict):
            if self.debug_enabled:
                print(f"[SUB-DGB] Attention: context is not an instance of tree!")
            return None

        # Based on the entity types, we expand for context tree pruning
        entity_types = self._expand_entity_types_for_context(entity_types)
        
        # If current node matches target type, return it
        if tree.get('_type') in entity_types:
            if self.debug_enabled:
                print(f"[SUB-DGB] We matched entity type {tree.get('_type')}!")
            return tree

        # Special handling for slides array
        if 'slides' in tree:
            pruned_slides = []
            for slide in tree['slides']:
                pruned_slide = self.prune_context_tree(slide, entity_types)
                if pruned_slide:
                    pruned_slides.append(pruned_slide)
            if pruned_slides:
                return {'slides': pruned_slides}
            return None
        elif self.debug_enabled:
            print(f"[SUB-DGB] slides not in tree")

        # Special handling for shapes array
        if 'shapes' in tree:
            pruned_shapes = []
            for shape in tree['shapes']:
                pruned_shape = self.prune_context_tree(shape, entity_types)
                if pruned_shape:
                    pruned_shapes.append(pruned_shape)
            if pruned_shapes:
                # Only include the shapes array if we found matches
                return {'shapes': pruned_shapes}
            return None
        elif self.debug_enabled:
            print(f"[SUB-DGB] shapes not in tree")

        # For other nodes, recursively prune while preserving structure
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
                # Only preserve non-structural properties if we're keeping this node
                if any(self._node_contains_entity(child, entity_types) 
                       for child in tree.values() if isinstance(child, (dict, list))):
                    pruned[key] = value

        # If we have pruned the current node, return the pruned versiony
        if self.debug_enabled:
            print(f"[SUB-DGB] Pruning done with result {pruned}")

        return pruned if pruned else None
    
    def _expand_entity_types_for_context(self, entity_types):
        """Expand entity types to include related entities for context pruning"""
        expanded_types = set(entity_types)
        for entity in entity_types:
            children = self._hierarchy_schema["valid_scopes"].get(entity, [])
            expanded_types.update(children)
        return list(expanded_types)
    
    def _node_contains_entity(self, node, entity_types):
        """Helper to check if a node or its children contain target entities"""
        if isinstance(node, dict):
            if node.get('_type') in entity_types:
                return True
            return any(self._node_contains_entity(v, entity_types) for v in node.values())
        elif isinstance(node, list):
            return any(self._node_contains_entity(item, entity_types) for item in node)
        return False    


    def validate_scope(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scope validation for select_* statements"""
        if not node["stmt"].startswith("select_"):
            return node

        target_entity = node["stmt"].split("_")[1]  # "text" from "select_text"
        args = node["args"]

        # Case 1: Metadata-based selection (e.g., select_slides(name="Title"))
        if "scope" not in args and any(k in self._hierarchy_schema["contextual_scopes"].get(target_entity, []) for k in args):
            return node  # Bypass hierarchy checks for contextual queries

        # Case 2: Standard scope validation
        scope_entity = args.get("scope")
        setDefaultFlag = False

        # Add this synonyms check before hierarchy validation
        if scope_entity == "document":
            scope_entity = "Presentation"  # Treat as special scope
            node["args"]["scope"] = "Presentation"  # Update the AST

        # 1. Inject default scope if missing
        if scope_entity is None:
            setDefaultFlag = True

        # 2. Ensure scope is a string (not a number/variable)
        else:
            try:
                if not isinstance(scope_entity, str):
                    raise ValueError(
                        f"Scope must be a string or special keyword. Got: {scope_entity} (type: {type(scope_entity)})"
                        )
            except ValueError as e:
                print(f"[ERROR] : {e} -> will use default scope!")
                setDefaultFlag = True

        if setDefaultFlag:
            default_scope = self._hierarchy_schema["scope_defaults"].get(node["stmt"], "Selection")
            print(f"[INFO] : normalized scope for '{node["stmt"]}' to default {default_scope}")
            node["args"]["scope"] = default_scope
            scope_entity = default_scope
            
        # 3. Skip further validation for special scopes ("Presentation", "Selection")
        if scope_entity in self._hierarchy_schema["special_scopes"]:
            return node

        # 4. Normalize scope_entity (e.g., "shape" → "shapes" if pluralization matters)
        normalized_scope = scope_entity.rstrip("s") if scope_entity.endswith("s") else scope_entity
        valid_parents = self._hierarchy_schema["valid_scopes"][target_entity]

        # 5. Check hierarchy validity (with pluralization support)
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

    
    def additional_corrections(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Office-specific corrections to AST nodes"""
        # Step 3: Clamp numerical values (like transparency) to valid range [0.0, 1.0]
        for key in ('fillTransparency', 'lineTransparency'):
            if key in node['args'] and isinstance(node['args'][key], (int, float)):
                # Prevent invalid values outside expected range
                node['args'][key] = max(0.0, min(1.0, node['args'][key]))
        return node

    
    def get_collection_name(self) -> str:
        """Return the name of the ChromaDB collection for this DSL"""
        return "odsl_samples"

    
    def set_debug(self, dbg: bool) -> None:
        """
        Enable/disable debugging.
        
        Args:
            dbg: The boolean status to set for debug_enabled
        """
        print(">>> DEBUGGING FOR OFFICE DSL TOGGLED <<<")
        self.debug_enabled = dbg
