# dsl_interface.py
import os
import importlib.util

def load_dsl_from_path(path, module_name="dsl_definition"):
    """Dynamically load a DSL class from a specific file path."""
    file_path = os.path.join(path, "dsl_definition.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DSL

def load_transpiler_from_path(path, module_name="transpiler"):
    """Dynamically load a Transpiler class from a specific file path."""
    file_path = os.path.join(path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location("Transpiler", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transpiler
