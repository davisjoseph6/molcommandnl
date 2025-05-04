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
        """Transpile AST to Markdown"""
        md_lines = []

        for node in ast:
            stmt = node['stmt']
            args = node['args']

            if stmt == 'add_heading':
                level = int(args.get('level', 1))
                text = args.get('text', '')
                md_lines.append(f"{'#' * level} {text}")

            elif stmt == 'add_paragraph':
                md_lines.append(args.get('text', ''))
                md_lines.append('')  # Empty line after paragraph

            elif stmt == 'add_list':
                items = args.get('items', [])
                ordered = args.get('ordered', False)

                if isinstance(items, str):
                    # If items is a string, try to parse it as a list
                    try:
                        # Remove brackets and split by commas
                        items_str = items.strip('[]')
                        items = [item.strip().strip('"\'') for item in items_str.split(',')]
                    except:
                        items = [items]

                for i, item in enumerate(items):
                    if ordered:
                        md_lines.append(f"{i + 1}. {item}")
                    else:
                        md_lines.append(f"- {item}")
                md_lines.append('')  # Empty line after list

            elif stmt == 'add_code_block':
                code = args.get('code', '')
                language = args.get('language', '')
                md_lines.append(f"```{language}")
                md_lines.append(code)
                md_lines.append("```")
                md_lines.append('')

            elif stmt == 'add_image':
                url = args.get('url', '')
                alt_text = args.get('alt_text', '')
                md_lines.append(f"![{alt_text}]({url})")
                md_lines.append('')

            elif stmt == 'add_link':
                url = args.get('url', '')
                text = args.get('text', url)
                md_lines.append(f"[{text}]({url})")

            elif stmt == 'add_bold':
                text = args.get('text', '')
                md_lines.append(f"**{text}**")

            elif stmt == 'add_italic':
                text = args.get('text', '')
                md_lines.append(f"*{text}*")

            elif stmt == 'add_table':
                headers = args.get('headers', [])
                rows = args.get('rows', [])

                if headers:
                    md_lines.append(" | ".join(headers))
                    md_lines.append(" | ".join(["---"] * len(headers)))

                for row in rows:
                    md_lines.append(" | ".join(row))
                md_lines.append('')

        return "\n".join(md_lines)


# --- Main Example Usage ---
if __name__ == "__main__":
    tests = [
        [{'var': None, 'stmt': 'add_heading', 'args': {'text': 'Climate Change', 'level': 1}}, {'var': None, 'stmt': 'add_paragraph', 'args': {'text': "Climate change refers to significant changes in global temperatures, weather patterns, and other aspects of the Earth's climate over time."}}, {'var': None, 'stmt': 'add_heading', 'args': {'text': 'Causes of Climate Change', 'level': 2}}, {'var': None, 'stmt': 'add_list', 'args': {'items': ['Greenhouse gas emissions', 'Deforestation', 'Industrial activities'], 'ordered': False}}, {'var': None, 'stmt': 'add_heading', 'args': {'text': 'Effects of Climate Change', 'level': 2}}, {'var': None, 'stmt': 'add_list', 'args': {'items': ['Rising sea levels', 'Extreme weather events', 'Loss of biodiversity'], 'ordered': False}}]        ]
    transpiler = Transpiler()
    print(transpiler.transpile_ast(tests[0]))
