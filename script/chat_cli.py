#!/usr/bin/env python3
"""
molcommandnl.script.chat_cli

Rich terminal UI for chatting with multiple DSLs (MolCommand, Office, Markdown).

Key fixes in this version:
- SemanticInterpreter constructor is called with the correct argument order:
    SemanticInterpreter(dsl_instance, llm)
  (your actual signature is (dsl, llm, chroma_path=None)).
- Adds standard shebang + module docstring.
- Makes DSL/transpiler instantiation robust if loader returns an instance vs a class.
"""

import argparse
import os
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from dsl_utils import load_dsl_from_path, load_transpiler_from_path
from semantic_interpreter import SemanticInterpreter, LLMClient
from stream_capture import capture_python_output
from utils import load_config


class ChatUI:
    def __init__(self, console: Console):
        self.console = console
        self.debug_lines = []
        self.response_lines = []

    def render(self):
        """Render the UI layout."""
        terminal_height = self._get_terminal_height()
        layout = self._create_layout(terminal_height)
        self.console.print(layout)

    def _get_terminal_height(self):
        """Get terminal height with fallback."""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 25

    def _create_layout(self, terminal_height):
        available_height = terminal_height - 1
        debug_height = int(available_height * 0.85)
        response_height = available_height - debug_height

        layout = Layout()
        layout.split_column(
            Layout(name="blank", size=1),
            Layout(name="top", size=debug_height),
            Layout(name="bottom", size=response_height),
        )

        self._update_panels(layout, debug_height, response_height)
        return layout

    def _update_panels(self, layout, debug_height, response_height):
        debug_lines_to_show = max(0, min(len(self.debug_lines), debug_height - 2))
        response_lines_to_show = max(0, min(len(self.response_lines), response_height - 2))

        layout["top"].update(
            Panel(
                "\n".join(self.debug_lines[-debug_lines_to_show:]),
                title="Debug",
                border_style="cyan",
            )
        )
        layout["bottom"].update(
            Panel(
                "\n".join(self.response_lines[-response_lines_to_show:]),
                title="Results",
                border_style="green",
            )
        )

    def add_debug(self, message: str):
        self.debug_lines.append(message)

    def add_response(self, message: str):
        self.response_lines.append(message)


class DSLManager:
    def __init__(self, script_dir: Path, llm: LLMClient):
        self.script_dir = script_dir
        self.llm = llm
        self.transpilers = {
            "molcommand": "unitymol_transpile",
            "odsl": "office_transpile",
            "markdown": "markdown_transpile",
        }

    def load_dsls(self):
        """Load all available DSLs."""
        dsls = {}

        dsls.update(self._load_dsl_module("molcommand"))
        dsls.update(self._load_dsl_module("odsl", "office"))
        dsls.update(self._load_dsl_module("markdown"))

        return dsls

    def _load_dsl_module(self, dsl_name: str, key_name: str = None):
        """Load a single DSL module and its context."""
        if key_name is None:
            key_name = dsl_name

        try:
            dsl_path = (self.script_dir / f"../{dsl_name}").resolve()

            DSLObj = load_dsl_from_path(dsl_path, f"{dsl_name}_dsl")
            dsl_instance = DSLObj() if callable(DSLObj) else DSLObj

            context = None
            try:
                from utils import load_json_file
                context = load_json_file(dsl_path / "example-context.json")
            except Exception:
                # optional; don't break UI if missing
                pass

            TranspilerObj = load_transpiler_from_path(dsl_path, self.transpilers[dsl_name])
            transpiler_instance = TranspilerObj() if callable(TranspilerObj) else TranspilerObj

            # FIX: correct constructor signature (dsl first, llm second)
            interpreter = SemanticInterpreter(dsl_instance, self.llm)

            return {
                key_name: {
                    "dsl": dsl_instance,
                    "interpreter": interpreter,
                    "transpiler": transpiler_instance,
                    "context": context,
                }
            }

        except Exception as e:
            print(f"[red]Failed to load {dsl_name} DSL: {str(e)}[/red]")
            return {}


def parse_arguments():
    parser = argparse.ArgumentParser(description="DSL Chat Interface")
    parser.add_argument("--simple", action="store_true", help="Use simple linear chat interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def clear_screen():
    """Clear the terminal screen in a cross-platform way."""
    os.system("cls" if os.name == "nt" else "clear")


def detect_dsl_type(query: str) -> str:
    """Detect which DSL to use based on the query."""
    query_lower = (query or "").lower()

    if any(kw in query_lower for kw in ["pdb", "protein", "structure", "molecule", "load", "display"]):
        return "molcommand"
    if any(kw in query_lower for kw in ["slide", "presentation", "document", "bold", "italic", "format"]):
        return "office"
    if any(kw in query_lower for kw in ["markdown", "heading", "bullet", "list"]):
        return "markdown"

    return "molcommand"


def process_query(query: str, dsls: dict, ui: ChatUI):
    """Process a query using the appropriate DSL."""
    dsl_type = detect_dsl_type(query)

    if dsl_type not in dsls:
        ui.add_debug(f"[red]DSL '{dsl_type}' not loaded. Using default.[/red]")
        available_dsls = list(dsls.keys())
        if available_dsls:
            dsl_type = available_dsls[0]
        else:
            ui.add_response("[bold red]No DSLs available to process query.[/bold red]")
            return

    ui.add_debug(f"[blue]Using {dsl_type.upper()} DSL for query: '{query}'[/blue]")

    dsl_info = dsls[dsl_type]
    interpreter = dsl_info["interpreter"]
    transpiler = dsl_info["transpiler"]
    context = dsl_info["context"]

    try:
        buffer = []
        start_time = time.time()

        with capture_python_output(buffer):
            result = interpreter.interpret(query, context=context)

        elapsed_time = time.time() - start_time

        for line in "".join(buffer).strip().splitlines():
            ui.add_debug(f"[dim]{line}[/dim]")

        ui.add_debug("[dim]DSL code generated:[/dim]")
        ui.add_debug(f"[yellow]{result.get('dsl', '')}[/yellow]")

        native_code = transpiler.transpile(result["ast"])
        ui.add_response(
            f"[bold green]✓ Success![/bold green] {dsl_type.upper()} result in "
            f"[yellow]{elapsed_time:.2f} seconds[/yellow]:"
        )

        if dsl_type == "molcommand":
            ui.add_response("[cyan]# IronPython Code:[/cyan]")
        elif dsl_type == "office":
            ui.add_response("[cyan]# JavaScript Code:[/cyan]")
        elif dsl_type == "markdown":
            ui.add_response("[cyan]# Markdown:[/cyan]")

        ui.add_response(f"[white]{native_code}[/white]")

    except Exception as e:
        ui.add_debug(f"[red]Error interpreting query: {str(e)}[/red]")
        ui.add_response("[bold red]Failed to process query.[/bold red]")


def run_simple_mode(dsls: dict):
    """Run the simple linear chat interface."""
    print("Simple chat mode. Type 'exit' to quit.")
    while True:
        try:
            query = input("> ")
        except EOFError:
            break

        if query.strip().lower() in ("exit", "quit"):
            break

        dsl_type = detect_dsl_type(query)
        if dsl_type not in dsls:
            print(f"[ERROR] DSL '{dsl_type}' not loaded.")
            continue

        interpreter = dsls[dsl_type]["interpreter"]
        transpiler = dsls[dsl_type]["transpiler"]
        context = dsls[dsl_type]["context"]
        print(f"[INFO] Using {dsl_type.upper()} DSL...")

        start_time = time.time()
        try:
            buffer = []
            with capture_python_output(buffer):
                result = interpreter.interpret(query, context=context)
            elapsed = time.time() - start_time

            for line in "".join(buffer).strip().splitlines():
                print(f"[DEBUG] {line}")

            print("[DSL CODE]")
            print(result["dsl"])

            print("[Cleaned DSL CODE]")
            print(interpreter.parser.unparse(result["ast"]))

            native_code = transpiler.transpile(result["ast"])
            print("[OUTPUT]")
            print(native_code)

            print(f"[✓] Done in {elapsed:.2f} seconds.")

        except Exception as e:
            print(f"[ERROR] {e}")


def run_interactive_mode(dsls: dict, ui: ChatUI, console: Console):
    """Run the interactive chat mode."""
    while True:
        clear_screen()
        ui.render()

        try:
            query = console.input("> ")
            if query.strip().lower() in ("exit", "quit"):
                break
            process_query(query, dsls, ui)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break


def _build_llm_from_env(script_dir: Path) -> LLMClient:
    """Construct LLMClient robustly even if its signature changes."""
    config = load_config(str((script_dir / "../config/env").resolve()))
    model = config.get("QUERYLLM")

    try:
        return LLMClient(model=model)
    except TypeError:
        return LLMClient()


def main():
    args = parse_arguments()

    script_dir = Path(__file__).resolve().parent
    llm = _build_llm_from_env(script_dir)

    dsl_manager = DSLManager(script_dir, llm)
    dsls = dsl_manager.load_dsls()

    if args.simple:
        run_simple_mode(dsls)
    else:
        console = Console()
        ui = ChatUI(console)
        ui.add_debug("[blue]Loading DSLs...[/blue]")
        ui.add_debug(f"[blue]Loaded {len(dsls)} DSLs: {', '.join(dsls.keys())}[/blue]")
        run_interactive_mode(dsls, ui, console)


if __name__ == "__main__":
    main()
