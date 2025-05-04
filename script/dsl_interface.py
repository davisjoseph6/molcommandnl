# dsl_interface.py

from typing import List, Dict, Any, Optional
import json

class DSLInterface:
    """
    Abstract interface for Domain Specific Language implementations.
    Implement this class to create a new DSL for the semantic interpreter.
    """


    def get_syntax(self, entities: List[str]) -> str:
        """Return the syntax description of the DSL"""
        raise NotImplementedError("Subclasses must implement get_syntax()")
    
    def get_rules(self, entities: List[str]) -> str:
        """Return the rules of the DSL"""
        raise NotImplementedError("Subclasses must implement get_rules()")
    
    def get_sysprompt_entity_context(self) -> str:
        """Return the entity context system prompt for the DSL"""
        raise NotImplementedError("Subclasses must implement get_sysprompt_entity_context()")
    
    def get_usrprompt_entity_context(self) -> str:
        """Return the entity context user prompt for the DSL"""
        raise NotImplementedError("Subclasses must implement get_usrprompt_entity_context()")

    def prune_context_tree(self, tree: dict, entity_types: list[str]) -> dict:
        """Return valid enumeration values for parameters"""
        raise NotImplementedError("Subclasses must implement prune_context_tree()")

    def extract_dsl_lines(self, raw: str) -> List[str]:
        """Extract valid DSL lines from raw LLM output"""
        raise NotImplementedError("Subclasses must implement extract_dsl_lines()")
    
    def normalize_text(self, text: str) -> str:
        """Normalize input text for improved matching"""
        raise NotImplementedError("Subclasses must implement normalize_text()")
    
    def get_statement_aliases(self) -> Dict[str, str]:
        """Return mapping of aliases to canonical statement names"""
        raise NotImplementedError("Subclasses must implement get_statement_aliases()")
    
    def get_valid_enums(self) -> Dict[str, List[str]]:
        """Return valid enumeration values for parameters"""
        raise NotImplementedError("Subclasses must implement get_valid_enums()")

    def validate_scope(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply DSL-specific corrections to AST nodes"""
        return node  # Default implementation: no additional corrections

    def additional_corrections(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply DSL-specific corrections to AST nodes"""
        return node  # Default implementation: no additional corrections
