from typing import List, Dict, Any, Optional


# --- Transpiler ---
class Transpiler:
    def transpile(self, ast: List[Dict[str, Any]]) -> str:
        """Transpile an AST to target language code"""
        try:
            return self.transpile_ast(ast)
        except KeyError as e:
            raise ValueError(f"Unsupported command type: {e}")
        except Exception as e:
            raise RuntimeError(f"Transpilation failed: {e}")

    def transpile_ast(self, ast: List[Dict[str, Any]]) -> str:
        """Convert AST to IronPython code for UnityMol"""
        lines = []
        for node in ast:
            try:
                cmd_type = node["stmt"].split("_")[0] # add/select/update/delete
                print(f"[transpiler] Implement logic for processing command type '{cmd_type}' with node: {node}")
            except Exception as e:
                print(f"[error] Failed to transpile node {node}: {e}")
        return "\n".join(lines)

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
            {"stmt": "add_sticks", "args": {"selection": "resid 1"}},
            {"stmt": "select_structure", "args": {"name": "1CRN"}},
            {"stmt": "update_color", "args": {"selection": "last()", "color": "#FF0000"}},
            {"stmt": "delete_structure", "args": {}}
        ]
        ]
    transpiler = Transpiler()
    print(transpiler.transpile(tests[2]))
