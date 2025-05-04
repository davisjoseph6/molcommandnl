# pattern_lib.py
class PatternLib:
    """Named patterns for template matching"""
    def __init__(self, transpiler=None):
        self.transpiler = transpiler  # Reference to parent transpiler

    def valid_radius(self, args: dict) -> bool:
        if self.transpiler.debug_enabled: print(f"[DBG] Checking radius: {args.get('radius')}")  # Debug
        return args.get("radius", 0) > 0

    @staticmethod
    def is_organic(args: dict) -> bool:
        return args.get("type") == "organic"

    @staticmethod
    def has_pdbid(args: dict) -> bool:
        return "PDBID" in args
    
    @staticmethod
    def has_valid_filepath(args: dict) -> bool:
        return ("filePath" in args and 
                args["filePath"].endswith(".pdb"))
    
    @staticmethod
    def xor_pdbid_filepath(args: dict) -> bool:
        return ("PDBID" in args) ^ ("filePath" in args)  # XOR
    
    # Complex patterns are now simple method calls
    @staticmethod
    def complex_condition(args):
        return (args.get("A") and 
            not args.get("B") and
            args["value"] > 100)

    @staticmethod
    def radius_gt_5(args):
        return args.get('radius', 0) > 5

    def has_structure_name_or_previous(self, args):
        """More reliable pattern check"""
        has_direct = "structureName" in args
        has_previous = bool(getattr(self.transpiler, 'previous_selections', {}).get('structureName'))
        if self.transpiler.debug_enabled: print(f"[DBG] Pattern has_direct: {has_direct}, has_previous: {has_previous}")
        return has_direct or has_previous

    @staticmethod
    def has_special_property(args):
        return args.get("special", False)

    def evaluate(self, expr: str, args: dict) -> bool:
        """Safe evaluation of compound patterns"""
        from operator import and_, or_, not_

        if not expr:
            return False

        if not isinstance(args, dict):
            raise ValueError("Args must be a dictionary")

        # Handle simple single pattern
        if expr.isidentifier():
            if not hasattr(self, expr):
                raise ValueError(f"Unknown pattern: {expr}")
            return getattr(self, expr)(args)

        # Handle compound patterns
        tokens = expr.split()
        result = None
        current_op = None

        for token in tokens:
            if token in ('and', 'or'):
                current_op = token
            else:
                if not hasattr(self, token):
                    raise ValueError(f"Unknown pattern: {token}")
                value = getattr(self, token)(args)

                if current_op == 'and':
                    result = result and value
                elif current_op == 'or':
                    result = result or value
                else:
                    result = value

        return bool(result)