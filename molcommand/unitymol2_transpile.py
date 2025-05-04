import yaml
from typing import Dict, Any, Optional, List
from jinja2 import Template
from pathlib import Path
from unitymol2_pattern_lib import PatternLib

from pathlib import Path
script_dir = Path(__file__).resolve().parent


class Transpiler:
    def __init__(self, template_file: str = script_dir / "unitymol2_templates.yaml"):
        self.debug_enabled = False
        self.pattern_lib = PatternLib(self)  # Pass self reference
        self.templates = self._load_templates(template_file)
        if self.debug_enabled: print(f"[DBG] Loaded templates:", self.templates.keys())  # Debug
        self._process_all_side_effects()  # Add this line
        self.previous_selections = {}
        # Type converters
        self.type_converters = {
            "color": self._convert_color,
            "selection": self._convert_selection,
            "position": lambda x: str(x) if isinstance(x, list) else x
        }

    def _process_all_side_effects(self):
        """Process side effects in all templates"""
        for cmd, config in self.templates.items():
            if isinstance(config, dict):
                if "variants" in config:
                    for variant in config["variants"]:
                        self._process_config(variant)
                self._process_config(config)

    def _load_templates(self, file_path: str) -> Dict:
        """Load templates from YAML with validation"""
        with open(file_path) as f:
            templates = yaml.safe_load(f)

        # Validate all patterns exist
        for cmd, config in templates.items():
            if isinstance(config, dict):
                if "variants" in config:
                    for variant in config["variants"]:
                        self._validate_pattern(variant)
                self._validate_pattern(config)

        return templates

    def _validate_pattern(self, config: Dict):
        """Ensure pattern methods exist, now checking instance methods"""
        if "pattern" not in config or not isinstance(config["pattern"], str):
            return

        for token in config["pattern"].split():
            if token not in ("and", "or", "not"):
                if not hasattr(self.pattern_lib, token):
                    raise ValueError(f"Pattern method '{token}' not found")

        # Split compound pattern into individual components
        pattern_components = [
            token for token in config["pattern"].split()
            if token not in ("and", "or", "not")
        ]

        # Validate each component exists
        for pattern_name in pattern_components:
            if not hasattr(self.pattern_lib, pattern_name):
                raise ValueError(
                    f"Pattern component '{pattern_name}' not found in PatternLib. "
                    f"Full pattern: '{config['pattern']}'"
                )

    def transpile(self, ast: List[Dict[str, Any]]) -> str:
        """Convert AST to output code"""
        lines = []
        for node in ast:
            try:
                result = self._process_node(node)
                if result is not None:
                    lines.append(result.strip())
            except Exception as e:
                print(f"[ERROR] Failed to process node {node}: {e}")
        return "\n".join(lines)

    def _process_config(self, config: Dict):
        """Convert string side effects to callable methods"""
        if "side_effect" in config and isinstance(config["side_effect"], str):
            method_name = config["side_effect"]
            # Try both with and without underscore prefix
            method = getattr(self, method_name, None) or \
                     getattr(self, f"_{method_name}", None)
            if not method or not callable(method):
                raise ValueError(f"Invalid side effect: {method_name}")
            config["side_effect"] = method

    def _process_node(self, node: Dict[str, Any]) -> Optional[str]:
        """Process a single AST node with debug output"""
        if self.debug_enabled: print(f"\n[DBG] Processing node: {node}")  # Debug

        command = node["stmt"]
        args = self._convert_args(node["args"])
        if self.debug_enabled: print(f"[DBG] Converted args: {args}")  # Debug

        try:
            template_config = self._find_template(command, args)
        except ValueError as e:
            print(f"[ERROR] Pattern evaluation failed for {node}: {e}")
            return None
        if self.debug_enabled: print(f"[DBG] Template config: {template_config}")  # Debug

        if not template_config:
            if self.debug_enabled: print(f"[DBG] No template found for command: {command}")  # Debug
            return None

        # PHASE 1: Prepare values needed for rendering
        delete_target = None
        if command == "delete_structure":
            delete_target = self.previous_selections.get("structureName")
            if self.debug_enabled: print(f"[DBG] Delete target: {delete_target}")  # Debug
            if not delete_target:
                return None  # Nothing to delete

        if "side_effect" in template_config:
            if self.debug_enabled: print(f"[DBG] Executing side effect for {command}")  # Debug
            if self.debug_enabled: print(f"[DBG] tc: {template_config} args: {args}")  # Debug
            template_config["side_effect"](args)

        # Prepare context with both args and state
        context = {
            **args,
            "delete_target": delete_target,  # Pre-computed value
            "previous_selections": self.previous_selections.copy(),
            "transpiler": self  # Optional: gives full access if needed
        }

        if self.debug_enabled: print(f"[DBG] Render Context: {context}")

        # Render template with proper context
        template_str = template_config["template"].strip()
        if not template_str:  # Empty template
            return None

        # Add to _process_node before rendering:
        if self.debug_enabled: print(f"[DBG] Render Template: {template_str}")

        # Render template
        rendered = Template(template_str).render(**context)

        if self.debug_enabled: print(f"[DBG] Rendered output: '{rendered}'")  # Debug

        return rendered.strip() if rendered.strip() else None


    def _find_template(self, command: str, args: Dict) -> Optional[Dict]:
        """Find matching template using pattern library"""
        if self.debug_enabled: print(f"[DBG]    Looking for command: {command}")  # Debug
        if self.debug_enabled: print(f"[DBG]    Previous selections: {self.previous_selections}")  # Debug

        if command not in self.templates:
            if self.debug_enabled: print(f"[DBG]    Command '{command}' not in templates")  # Debug
            return None

        config = self.templates[command]
        if self.debug_enabled: print(f"[DBG]    Found config for '{command}': {config}")  # Debug

        # Handle command variants
        if "variants" in config:
            for i, variant in enumerate(config["variants"]):
                if self.debug_enabled: print(f"[DBG]       Checking variant {i}: {variant}")  # Debug
                if self._matches_pattern(variant, args):
                    if self.debug_enabled: print(f"[DBG]       Matched variant {i}")  # Debug
                    return variant
            return config.get("default")

        # Handle simple template
        if self._matches_pattern(config, args):
            if self.debug_enabled: print(f"[DBG]    Matched base template")  # Debug
            return config

        if self.debug_enabled: print(f"[DBG]    No pattern/requirements matched for '{command}'")  # Debug
        return None

    def _matches_pattern(self, config: Dict, args: Dict) -> bool:
        """Check if args match template configuration"""
        # Check required args first
        if "required" in config:
            required_ok = all(arg in args for arg in config["required"])
            if self.debug_enabled: print(f"[DBG]       Required args check: {required_ok}")  # Debug
            if not required_ok:
                return False

        # Check pattern if specified
        if "pattern" in config:
            if self.debug_enabled: print(f"[DBG]       Evaluating pattern: {config['pattern']}")  # Debug
            try:
                result = self.pattern_lib.evaluate(config["pattern"], args)
                if self.debug_enabled: print(f"[DBG]       Pattern result: {result}")  # Debug
                return result
            except Exception as e:
                print(f"[ERROR] Pattern evaluation failed: {e}")
                return False

        return True

    def _convert_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Type conversion for arguments"""
        if self.debug_enabled: print(f"[DBG] Converting args: {args}")  # Debug
        converted = {}
        for k, v in args.items():
            if k in self.type_converters:
                converted[k] = self.type_converters[k](v)
            else:
                converted[k] = v
        return converted

    def store_selection(self, args: Dict):
        """Store the selection reliably"""
        selection_name = args.get("name") or args.get("structureName")
        if selection_name:
            self.previous_selections = {"structureName": selection_name}
            if self.debug_enabled: print(f"[DBG] Stored selection: {self.previous_selections}")
        else:
            print("[WARNING] No selection name provided")

    def _clear_selection(self, args: Dict):
        """Clear selections but maintain the dictionary"""
        self.previous_selections.clear()
        if self.debug_enabled: print(f"[DBG] Cleared selections (after rendering): {self.previous_selections}")

    def _convert_color(self, hex_str: str) -> str:
        """#RRGGBB → Unity Color (matches original)"""
        if hex_str.startswith("#"):
            r = int(hex_str[1:3], 16) / 255
            g = int(hex_str[3:5], 16) / 255
            b = int(hex_str[5:7], 16) / 255
            return f'Color({r}f, {g}f, {b}f)'
        return hex_str

    def _convert_selection(self, sel: str) -> str:
        """DSL selection → MDA syntax (matches original)"""
        return sel.replace("residues", "resid").replace("around", "around ")

# --- Main Example Usage ---
if __name__ == "__main__":
    tests = [
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
    transpiler = Transpiler()
    print(transpiler.transpile(tests[2]))

# Interesting patterns I may want to use
# "pattern": lambda args: "PDBID" in args,  # More flexible
# "pattern": lambda args: "filePath" in args and args["filePath"].endswith(".pdb"),
# Either PDBID or filePath must exist, but not both
# "pattern": lambda args: ("PDBID" in args) ^ ("filePath" in args)  # XOR
# # Only valid if radius > 0
# "pattern": lambda args: "radius" in args and args["radius"] > 0
