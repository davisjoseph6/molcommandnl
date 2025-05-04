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
        """Transpile AST to JavaScript for Office API"""
        lines = []
        # translate each node into proper API syntax
        for node in ast:
            # Statement
            stmt = node['stmt']
            # Statement arguments
            args = ', '.join(f"{k}: {repr(v)}" for k, v in node['args'].items())
            # Construct valid javascript
            js = f"await context.{stmt}({{{args}}});"
            lines.append(js)
        return "\n".join(lines)

# --- Main Example Usage ---
if __name__ == "__main__":
    tests = [
        [{'var': 'title', 'stmt': 'select_text', 'args': {'scope': 'Presentation', 'name': 'Title'}}, {'var': None, 'stmt': 'format_text', 'args': {'textRanges': 'title', 'fontName': 'Times New Roman', 'size': 36, 'bold': True, 'color': '#000000'}}, {'var': 'content', 'stmt': 'select_text', 'args': {'scope': 'Presentation', 'name': 'Content Placeholder 1'}}, {'var': None, 'stmt': 'format_text', 'args': {'textRanges': 'content', 'fontName': 'Times New Roman', 'size': 24, 'italic': True, 'color': '#000000'}}]
        ]
    transpiler = Transpiler()
    print(transpiler.transpile_ast(tests[0]))
