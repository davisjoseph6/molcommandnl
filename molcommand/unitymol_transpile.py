from typing import List, Dict, Any, Optional


# --- Transpiler ---
class Transpiler:
    def __init__(self):
        # Map DSL commands to UnityMol API patterns
        self.command_map = {
            "add": self._transpile_add,
            "select": self._transpile_select,
            "update": self._transpile_update,
            "delete": self._transpile_delete
        }

        # Type conversion rules (DSL -> IronPython)
        self.type_converters = {
            "color": self._convert_color,
            "selection": self._convert_selection
        }

        # Memory for storing selections
        self.previous_selections = {}

    def transpile(self, ast: List[Dict[str, Any]]) -> str:
        """Transpile an AST to target language code"""
        try:
            return self.transpile_ast(ast)
        except KeyError as e:
            raise ValueError(f"Unsupported command type: {e}")
        except Exception as e:
            raise RuntimeError(f"Transpilation failed: {e}")

#class UnityMolTranspiler:
#    def __init__(self):

    def transpile_ast(self, ast: List[Dict[str, Any]]) -> str:
        """Convert AST to IronPython code for UnityMol"""
        lines = []
        for node in ast:
            try:
                cmd_type = node["stmt"].split("_")[0] # add/select/update/delete
                if cmd_type in self.command_map:
                    result = self.command_map[cmd_type](node)
                    if result is not None:
                        lines.append(result)
                    else:
                        print(f"[transpiler] '{cmd_type}' returned None for node: {node}")
                else:
                    print(f"[transpiler] Unknown command type '{cmd_type}' in node: {node}")
            except Exception as e:
                print(f"[error] Failed to transpile node {node}: {e}")
        return "\n".join(lines)


    def _transpile_add(self, node: Dict) -> str:
        """Convert add_* commands to UnityMol's show/load APIs"""
        target = node["stmt"].replace("add_", "")
        args = self._convert_args(node["args"])

        if target in ["sticks", "surface", "ribbon"]:
            # Visual representations
            sel = args.get('selection', 'last()')
            if target == 'sticks':
                return f'showSelection("{sel}", "hb")'
            elif target == 'surface':
                sel = args.get('selection', 'last()')
                return f'showSelection("{sel}", "s")'
            else:
                return f'showSelection("{sel}", "{target[0]}")'
        elif target == "label":
            # Annotations
            pos = args.get('position', [0, 0, 0])
            text = args.get('text', '')
            colr = args.get('color', 'Color.white')
            lines.append(f'annotateWorldText({pos}, 0.2, "{text}", {colr}')

            return f'annotateWorldText({args["position"]}, 0.2, "{args["text"]}", {args["color"]})'
        elif target == "structure":
            if (tmp := args.get("PDBID")):
                # PDB structure loading
                return f'fetch("{tmp}")'
            if (tmp := args.get("filePath")):
                # PDB structure loading
                return f'load("{tmp}")'
        else:
            # Molecular data loading
            load_arg = args.get("file", "1crn")
            return f'load("{load_arg}")'


    def _transpile_select(self, node: Dict) -> str:
        """Convert select_* to UnityMol's select() with MDA syntax"""
        target = node["stmt"].replace("select_", "")
        args = self._convert_args(node["args"])

        if target == "structure":
            if (tmp := args.get("structureName")):
                self.previous_selections = {"structureName": tmp}
                return '' # No command to log, but no error raised
            elif (tmp := args.get("name")):
                self.previous_selections = {"structureName": tmp}
                return '' # No command to log, but no error raised
            else:
                print(f"[transpiler] 'select_structure' has no structureName for node: {node}")
            return None

            return f'select("resname {args["residue"]} and name {args["atom"]}", "{args["name"]}")'
        if target == "atoms":
            return f'select("resname {args["residue"]} and name {args["atom"]}", "{args["name"]}")'
        # if target == 'atoms':
        #     sel = args.get('selection', 'all')
        #     return f'select("{sel}", "temp_selection")'
        elif target == "residues":
            res = args.get('residues', [])
            # return f'select("resid {"+".join(res)}", "temp_selection")'
            return f'select("resid {args["ids"]}", "{args["name"]}")'
        elif target == "trajectory":
            return f'last().trajectory'  # Reference last loaded trajectory
        return None


    def _transpile_update(self, node: Dict) -> str:
        """Convert update_* to UnityMol's styling APIs"""
        target = node["stmt"].replace("update_", "")
        args = self._convert_args(node["args"])

        if target == "color":
            color_type = args.get("type", "hb")  # ou "selection", ou autre valeur par défaut
            return f'colorSelection("{args["selection"]}", "{color_type}", {args["color"]})'
            # sel = args.get('selection', 'all')
            # color = self._convert_color(args['color'])
            # rep_type = args.get('representation', 'hb')
            # return f'colorSelection("{sel}", "{rep_type}", {color})'

        elif target == 'visibility':
            sel = args.get('selection', 'all')
            visible = args.get('visible', True)
            action = "showSelection" if visible else "hideSelection"
            return f'{action}("{sel}")'

        elif target == "animation":
            return f'setModel("{args["trajectory"]}", {args["frame"]})'
        elif target == "style":
            return f'setHyperBallMetaphore("{args["selection"]}", "{args["style"]}")'
        return None


    def _transpile_delete(self, node: Dict) -> str:
        """Convert delete_* to hide/remove calls"""
        target = node["stmt"].replace("delete_", "")
        args = self._convert_args(node["args"])

        if target == "structure":
            #print(f"[transpiler] we have previous selections: {self.previous_selections}")
            if (tmp := args.get("structureName")):
                return f'delete("{tmp}")'
            elif (tmp := self.previous_selections.get("structureName")):
                self.previous_selections = {}
                return f'delete("{tmp}")'
            else:
                print(f"[transpiler] 'delete_structure' has no structureName for node: {node}")
                return None
        if target in ["sticks", "surface"]:
            return f'hideSelection("{args["selection"]}", "{target[0]}")'
        if target == 'representation':
            sel = args.get('selection', 'all')
            rep_type = args.get('type', 'all')
            if rep_type == 'all':
                return f'deleteRepresentationsInSelection("{sel}")'
            else:
                return f'deleteRepresentationInSelection("{sel}", "{rep_type[0]}")'
        elif target == "label":
            return f'removeAnnotationWorldText({args["position"]}, 0.2, "{args["text"]}")'
        return None


    def _convert_args(self, args: Dict) -> Dict:
        """Type-specific argument preprocessing"""
        converted = {}
        for k, v in args.items():
            if k in self.type_converters:
                converted[k] = self.type_converters[k](v)
            else:
                converted[k] = v
        return converted

    # if we have a color in hex format, we convert it to Unity Color
    def _convert_color(self, hex_str: str) -> str:
        """#RRGGBB → Unity Color"""
        r = int(hex_str[1:3], 16)/255
        g = int(hex_str[3:5], 16)/255
        b = int(hex_str[5:7], 16)/255
        return f'Color({r}f, {g}f, {b}f)'

    # if we have a selection in the form of "residues" or "around"
    def _convert_selection(self, sel: str) -> str:
        """DSL selection → MDA syntax"""
        return sel.replace("residues", "resid").replace("around", "around ")


# 6. Edge Case Handling
# # Input: add_surface(selection="ligand", type="electrostatic")
# # Output:
# if self._has_electrostatic_data():
#     return 'showSelection("ligand", "s", computeElectrostatics=True)'
# else:
#     return 'Debug.LogWarning("No electrostatic data loaded")'

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
    print(transpiler.transpile_ast(tests[2]))
