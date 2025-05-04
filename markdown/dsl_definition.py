# dsl_definition.py

import re
import os
import io
import string
from typing import List, Dict, Any, Optional
from dsl_interface import DSLInterface
from utils import load_config, load_text_file

class DSL(DSLInterface):
    """Implementation of DSL interface for Markdown generation"""
    
    def __init__(self):
        """Initialize the Markdown DSL with configs and rules"""
        self.debug_enabled = False
        # Get the absolute path to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Load the configuration from the config/env file
        config = load_config(script_dir + '/../config/env')
        # Set paths
        self.CONFIG_PATH = config.get('CONFIGPATH')
        self.CHROMA_PATH = config.get('CHROMAPATH') + '/markdown/'
        
        # Define the DSL components as strings (in a real implementation, these would be loaded from files)
        self._syntax = """
# Markdown DSL Syntax

This DSL is for generating markdown documents. The DSL supports the following functions:

add_heading(text, level)
add_paragraph(text)
add_list(items, ordered)
add_code_block(code, language)
add_image(url, alt_text)
add_link(url, text)
add_bold(text)
add_italic(text)
add_table(headers, rows)
"""

        self._rules = """
# Markdown DSL Rules

1. All functions must be called with named parameters.
2. Text parameters should be enclosed in double quotes.
3. For lists, items should be a comma-separated list enclosed in square brackets.
4. For tables, headers should be a list and rows should be a list of lists.
5. level parameter for headings must be between 1 and 6.
6. All parameters are required unless explicitly marked as optional.
"""

        self._sysprompt_entity_context = """
# Entity Context Analysis

Analyze the user instruction to determine which entities are involved and whether context is needed.

Possible entity categories for Markdown DSL:
- text (paragraphs, formatting)
- heading (titles, sections)
- list (bullet points, numbered lists)
- code (code blocks)
- image (pictures, diagrams)
- table (data tables)
- link (hyperlinks)

Instructions that require current document state (like "update the second paragraph") require context.

Output format:
Categories: [comma,separated,entity,names]
RequiresContext: true/false
"""

        self._usrprompt_entity_context = """
Instruction: "{utterance}"

Provide the following:
- Categories: [list of entities involved]
- RequiresContext: true or false
        """
        
    def get_syntax(self, entities: List[str]) -> str:
        return self._syntax
    
    def get_rules(self, entities: List[str]) -> str:
        return self._rules

    def get_sysprompt_entity_context(self) -> str:
        """Return the currently stored system prompt for the LLM"""
        return self._sysprompt_entity_context
    
    def get_usrprompt_entity_context(self) -> str:
        """Return the currently stored user prompt for the LLM"""
        return self._usrprompt_entity_context
    
    def extract_dsl_lines(self, raw: str) -> List[str]:
        """
        Extracts valid, complete DSL instruction lines from raw LLM output.
        """
        lines = []
        buffer = ''
        inside_statement = False

        for line in io.StringIO(raw):
            stripped = line.strip()

            if not stripped:
                continue

            # Detect start of a DSL statement
            if re.match(r"^\s*(\w+\s*=)?\s*(add_\w+)\(", stripped):
                buffer = stripped
                inside_statement = True

                if stripped.endswith(')'):
                    lines.append(buffer)
                    buffer = ''
                    inside_statement = False

            elif inside_statement:
                buffer += ' ' + stripped
                if stripped.endswith(')'):
                    lines.append(buffer)
                    buffer = ''
                    inside_statement = False

            else:
                print(f"⚠️ Ignoring non-DSL line: {stripped}")

        if buffer:
            print(f"⚠️ Incomplete DSL line left open: {buffer}")

        return lines
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize input text to improve matching.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\b(add|insert|create|generate)\b", "add", text)
        text = re.sub(r"\b(title|heading|header)\b", "heading", text)
        text = re.sub(r"\b(paragraph|text|content)\b", "text", text)
        text = re.sub(r"\b(bullet|numbered|item)\b", "list", text)
        return text.strip()
    
    def get_statement_aliases(self) -> Dict[str, str]:
        """Maps alias statement names to canonical DSL function names"""
        return {
            'add_title': 'add_heading',
            'add_bullet_list': 'add_list',
            'add_code': 'add_code_block',
            'add_picture': 'add_image',
            'add_hyperlink': 'add_link'
        }
    
    def get_valid_enums(self) -> Dict[str, List[str]]:
        """Valid enumerated values for specific parameters"""
        return {
            'ordered': ['true', 'false'],
            'level': ['1', '2', '3', '4', '5', '6']
        }
    
    def additional_corrections(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply markdown-specific corrections to AST nodes"""
        # Make sure heading levels are integers between 1 and 6
        if node['stmt'] == 'add_heading' and 'level' in node['args']:
            if isinstance(node['args']['level'], (int, float)):
                node['args']['level'] = max(1, min(6, int(node['args']['level'])))
        return node
    
    def get_collection_name(self) -> str:
        """Return the name of the ChromaDB collection for this DSL"""
        return "markdown_samples"

    def prune_context_tree(self, tree: dict, entity_types: list[str]) -> dict:
        # No pruning implemented yet
        return tree
    
    def set_debug(self, dbg: bool) -> None:
        """
        Enable/disable debugging.
        
        Args:
            dbg: The boolean status to set for debug_enabled
        """
        print(">>> DEBUGGING FOR MARKDOWN DSL TOGGLED <<<")
        self.debug_enabled = dbg

