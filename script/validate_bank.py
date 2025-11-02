#!/usr/bin/env python3
import os, sys, json, importlib.util

# Prefer the DSL living with the bank
molcmd_dir = os.path.join(os.path.dirname(__file__), "..", "molcommand")
sys.path.append(os.path.abspath(molcmd_dir))

def _import_validator():
    try:
        from validator import parse_and_validate_molcommand  # noqa
        return parse_and_validate_molcommand
    except ModuleNotFoundError:
        # Fallback to unitymol_copilot/validator.py (if present)
        alt = os.path.expanduser("~/unitymol_copilot/validator.py")
        if os.path.exists(alt):
            spec = importlib.util.spec_from_file_location("validator", alt)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.parse_and_validate_molcommand
        raise

from bank_loader import load_bank
parse_and_validate_molcommand = _import_validator()

def main():
    items = load_bank()
    errors = []
    for it in items:
        dsl = (it.get("dsl") or "").strip()
        if not dsl:
            errors.append((it.get("id"), "empty dsl"))
            continue
        ok, ast, errs = parse_and_validate_molcommand(dsl)
        if not ok:
            errors.append((it.get("id"), errs))
    if errors:
        print("❌ Bank has validation issues:")
        for i, (id_, es) in enumerate(errors, 1):
            print(f"  {i:02d}) {id_}: {es}")
        sys.exit(2)
    print(f"✅ Bank OK: {len(items)} items validated.")

if __name__ == "__main__":
    main()

