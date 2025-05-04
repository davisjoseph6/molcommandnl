import os
import time
import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Import DSL utilities
from dsl_utils import load_dsl_from_path, load_transpiler_from_path
from semantic_interpreter import SemanticInterpreter, LLMClient
from utils import load_config, load_json_file

class DSLTester:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.script_dir = Path(__file__).resolve().parent
        self.transpilers = {
            "molcommand": "unitymol_transpile",
            "odsl": "office_transpile",
            "markdown": "markdown_transpile"
        }

    def run_example(self, dsl_name: str, query: str, context: Optional[Dict] = None) -> None:
        """Run a single DSL example with proper formatting and timing"""
        print("=" * 50)
        print(f"EXAMPLE: {dsl_name.upper()} DSL")
        print("=" * 50)

        dsl_class = self._load_dsl(dsl_name)
        if not dsl_class:
            return

        interpreter = SemanticInterpreter(self.llm, dsl_class())

        start_time = time.time()
        result = interpreter.interpret(query, context=context)
        elapsed = time.time() - start_time

        self._print_results(result, interpreter, dsl_name, elapsed)

    def _load_dsl(self, dsl_name: str) -> Optional[Any]:
        """Load DSL class from appropriate path"""
        try:
            dsl_path = self.script_dir / f"../{dsl_name}"
            return load_dsl_from_path(dsl_path, module_name=f"{dsl_name}_dsl")
        except Exception as e:
            print(f"Failed to load {dsl_name} DSL: {e}")
            return None

    def _load_transpiler(self, dsl_name: str) -> Optional[Any]:
        """Load DSL class from appropriate path"""
        try:
            dsl_path = self.script_dir / f"../{dsl_name}"
            return load_transpiler_from_path(dsl_path, self.transpilers[dsl_name])
        except Exception as e:
            print(f"Failed to load {dsl_name} DSL: {e}")
            return None

    def _print_results(self, result: Dict, interpreter: SemanticInterpreter, dsl_name: str, elapsed: float) -> None:
        """Print formatted results of DSL interpretation"""

        transpiler_class = self._load_transpiler(dsl_name)
        if not transpiler_class:
            return

        transpiler = transpiler_class()  # Initialize the transpiler

        print("\n=== DSL ===")
        print(result["dsl"])

        print("\n=== AST ===")
        print(result["ast"])

        print("\n=== Cleaned DSL ===")
        print(interpreter.parser.unparse(result["ast"]))

        print("\n=== Native Code ===")
        native_code = transpiler.transpile(result["ast"])
        print(f"[✓] Done in {elapsed:.2f} seconds.")
        print(native_code)
        print("\n")


def load_office_context() -> Dict:
    """Load office DSL context file"""
    try:
        context_path = Path(__file__).resolve().parent / "../odsl/example-context.json"
        return load_json_file(context_path)
    except Exception as e:
        print(f"Warning: Could not load Office DSL context file: {e}")
        return None


def main():
    # Initialize
    config = load_config(Path(__file__).resolve().parent / '../config/env')
    llm = LLMClient(model=config.get('QUERYLLM'))
    tester = DSLTester(llm)

    # Example queries
    examples = [
        ("molcommand", "Load PDB ID 1kx2"),
        ("molcommand", "insert the 5iuf protein"),
        ("molcommand", "load test.pdb from disk"),
        ("odsl", "Make the text look formal", load_office_context()),
        ("markdown", "Create a document about climate change with headings and bullet points")
    ]

    # Run examples
    for dsl_name, query, *context in examples:
        tester.run_example(dsl_name, query, context[0] if context else None)


if __name__ == "__main__":
    main()