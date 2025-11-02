#!/usr/bin/env python3
"""
Convert the versioned sample bank (v0.1 shards) into the legacy
molcommand/samples.yaml format used by the current retrieval code.
"""
import os, sys
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "molcommand"))

from bank_loader import load_bank

try:
    import yaml  # pip install pyyaml
except Exception as e:
    raise RuntimeError("Please `pip install pyyaml` in this venv") from e

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
LEGACY_OUT = os.path.join(REPO_ROOT, "molcommand", "samples.yaml")

def to_legacy(items: List[Dict[str, Any]]) -> list:
    """Produce the legacy `samples.yaml` structure."""
    # Group by utterance so multiple DSL variants can attach as sub_samples
    by_utter = {}
    for it in items:
        utter = it.get("utterance", "").strip()
        dsl   = (it.get("dsl") or "").strip()
        ents  = it.get("entities") or []
        ctx   = it.get("context")
        if not utter or not dsl:
            continue
        by_utter.setdefault(utter, []).append({
            "program": dsl,
            "entities": ents,
            "context": ctx,
        })
    legacy = []
    for utter, subs in by_utter.items():
        legacy.append({
            "utterance": utter,
            "sub_samples": subs
        })
    # Stable ordering to avoid noisy diffs
    legacy.sort(key=lambda r: r["utterance"].lower())
    return legacy

def main():
    items = load_bank()
    legacy = to_legacy(items)
    os.makedirs(os.path.dirname(LEGACY_OUT), exist_ok=True)
    with open(LEGACY_OUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(legacy, f, sort_keys=False, allow_unicode=True)
    print(f"✅ Wrote {LEGACY_OUT} with {len(legacy)} utterances "
          f"({sum(len(u['sub_samples']) for u in legacy)} sub-samples total).")

if __name__ == "__main__":
    main()

