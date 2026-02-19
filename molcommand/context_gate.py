#!/usr/bin/env python3
"""
LLM-steered context gate.

Decides whether an utterance should be processed with scene context, using:
1) An LLM classifier (entity categories + RequiresContext)
2) A config-driven threshold policy (update_count / complexity score)

This module is intentionally policy-driven and contains minimal hardcoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import yaml


@dataclass
class ContextDecision:
    categories: List[str]
    requires_context: bool
    raw: str


class ContextGate:
    """LLM-based context gating with config-driven thresholds."""

    def __init__(self, policy: Dict[str, Any], llm_client: Any) -> None:
        self.policy = policy
        self.llm_client = llm_client

        gate_cfg = policy["context_gate"]
        self.update_threshold = int(gate_cfg.get("update_threshold", 0))

        self.complexity_cfg = gate_cfg.get("complexity", {"enabled": False})
        self.output_format = gate_cfg["llm"].get("output_format", "yaml").lower()

        self.system_prompt = self._read_text(gate_cfg["llm"]["system_prompt_path"])
        self.user_prompt_tmpl = self._read_text(gate_cfg["llm"]["user_prompt_path"])

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _complexity_score(self, scene_stats: Dict[str, Any]) -> float:
        if not self.complexity_cfg.get("enabled", False):
            return 0.0
        weights = self.complexity_cfg.get("weights", {})
        return (
            float(weights.get("updates", 0.0)) * float(scene_stats.get("updates", 0))
            + float(weights.get("structures", 0.0)) * float(scene_stats.get("structures", 0))
            + float(weights.get("selections", 0.0)) * float(scene_stats.get("selections", 0))
        )

    def decide(self, utterance: str, scene_stats: Dict[str, Any]) -> ContextDecision:
        """Return categories + requires_context, applying threshold bias."""

        user_prompt = self.user_prompt_tmpl.format(utterance=utterance)

        raw = self.llm_client.complete(
            system=self.system_prompt,
            user=user_prompt,
        )

        categories, requires = self._parse_output(raw)

        # Policy bias: if scene is complex enough, force context (config-driven).
        updates = int(scene_stats.get("updates", 0))
        if updates >= self.update_threshold:
            requires = True

        score = self._complexity_score(scene_stats)
        threshold = float(self.complexity_cfg.get("threshold", 0.0))
        if self.complexity_cfg.get("enabled", False) and score >= threshold:
            requires = True

        return ContextDecision(categories=categories, requires_context=requires, raw=raw)

    def _parse_output(self, raw: str) -> Tuple[List[str], bool]:
        raw = raw.strip()

        if self.output_format == "json":
            obj = json.loads(raw)
            cats = obj.get("Categories", []) or obj.get("categories", [])
            req = obj.get("RequiresContext", None)
            if req is None:
                req = obj.get("requires_context", False)
            return list(cats), bool(req)

        # default YAML
        obj = yaml.safe_load(raw)
        cats = obj.get("Categories", [])
        req = obj.get("RequiresContext", False)
        return list(cats), bool(req)
