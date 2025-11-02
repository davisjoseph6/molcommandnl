#!/usr/bin/env python3
"""
Bank loader for molcommand examples.

- Prefers MCL_BANK_PATH (e.g., ~/molcommandnl/bank/v0.1)
- Else falls back to ~/molcommandnl/molcommand/samples.yaml (legacy)
"""

import os, glob, json
from typing import List, Dict, Any

try:
    import yaml  # PyYAML
except Exception as e:
    raise RuntimeError("Please `pip install pyyaml` in this venv") from e

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
LEGACY_SAMPLES = os.path.join(REPO_ROOT, "molcommand", "samples.yaml")

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def load_bank() -> List[Dict[str, Any]]:
    """
    Returns a flat list of items with keys:
      {id, utterance, dsl, entities, context, tags, verb}
    """
    bank_dir = os.environ.get("MCL_BANK_PATH",
                              os.path.expanduser("~/molcommandnl/bank/v0.1"))

    items: List[Dict[str, Any]] = []

    if os.path.isdir(bank_dir):
        # New format (versioned shards)
        index_path = os.path.join(bank_dir, "bank.index.yaml")
        shards = []
        if os.path.exists(index_path):
            doc = _load_yaml(index_path)
            shards = [os.path.join(bank_dir, s) for s in doc.get("shards", [])]
        else:
            # glob if index is missing
            shards = sorted(glob.glob(os.path.join(bank_dir, "*.yaml")))

        for shard in shards:
            doc = _load_yaml(shard)
            verb = doc.get("verb", "").strip()
            for it in doc.get("items", []) or []:
                it = dict(it)  # shallow copy
                it["verb"] = verb or it.get("verb")
                items.append(it)
        return items

    # Legacy fallback: molcommand/samples.yaml (convert on the fly)
    if os.path.exists(LEGACY_SAMPLES):
        doc = _load_yaml(LEGACY_SAMPLES)
        legacy = []
        for row in doc or []:
            utter = row.get("utterance", "")
            for ss in (row.get("sub_samples") or []):
                legacy.append({
                    "id": utter.replace(" ", "_")[:64],
                    "utterance": utter,
                    "dsl": (ss.get("program") or "").strip(),
                    "entities": ss.get("entities") or [],
                    "context": ss.get("context"),
                    "tags": ["legacy"],
                    "verb": None,
                })
        return legacy

    raise FileNotFoundError(
        f"No bank found. Set MCL_BANK_PATH or keep legacy at {LEGACY_SAMPLES}"
    )

if __name__ == "__main__":
    all_items = load_bank()
    print(json.dumps(all_items[:5], indent=2, ensure_ascii=False))
    print(f"... total {len(all_items)} items")

