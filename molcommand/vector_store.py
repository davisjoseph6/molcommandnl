#!/usr/bin/env python3
"""
molcommand.vector_store

Chroma-backed history vector store for context retrieval.

Goal:
- Provide an optional, best-effort vector store that can be enabled/disabled via policy.
- Store per-turn "documents" (User + Exec) and retrieve turn IDs by semantic similarity.
- Avoid hard dependency failures: if chromadb or embeddings are missing, it disables itself.

Important:
- Chroma's EmbeddingFunction interface changed (0.4.16+). It now expects:
      __call__(self, input) -> embeddings
  where the parameter name is literally "input".
- Many embedding helpers return callables with *args/**kwargs signatures.
  We wrap any callable into a compliant adapter so Chroma accepts it.

This module is designed to be imported AFTER you set Chroma telemetry env vars
(if you want them disabled), because importing chromadb may initialize telemetry.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _short_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


class _EmbeddingAdapter:
    """
    Chroma EmbeddingFunction adapter with the exact required signature:
        __call__(self, input)

    It wraps arbitrary callable embedding functions and tries common calling conventions.
    """

    def __init__(self, fn: Any):
        self._fn = fn

    def __call__(self, input):  # noqa: A002  (must be named 'input' for Chroma's signature check)
        # Try calling patterns commonly used by embedding functions.
        try:
            out = self._fn(input)
        except TypeError:
            try:
                out = self._fn(texts=input)
            except TypeError:
                try:
                    out = self._fn(documents=input)
                except TypeError:
                    out = self._fn(input=input)

        # Normalize numpy arrays to python lists (Chroma accepts lists)
        if hasattr(out, "tolist") and callable(getattr(out, "tolist")):
            return out.tolist()
        return out


class HistoryVectorStore:
    """
    Optional Chroma-backed vector store for scene history retrieval.

    Stores "turn documents" as:
        User: <utterance>\nExec: <dsl>

    Public fields:
        enabled: bool
        init_stage: str
        init_error: Optional[str]
        embedding_source: Optional[str]
    """

    def __init__(self, chroma_dir: str, collection: str) -> None:
        self.enabled = False
        self._client = None
        self._collection = None

        self.init_stage = "start"
        self.init_error: Optional[str] = None
        self.embedding_source: Optional[str] = None

        # Import chromadb
        try:
            import chromadb  # type: ignore
        except Exception as e:
            self.init_stage = "import_chromadb"
            self.init_error = _short_exc(e)
            return

        # Load embedding function
        embed_fn, embed_diag, embed_src = self._try_get_embedding_function()
        self.embedding_source = embed_src
        if embed_fn is None:
            self.init_stage = "embedding_function"
            self.init_error = embed_diag or "No usable embedding function found"
            return

        # Wrap into compliant interface
        embed_fn = _EmbeddingAdapter(embed_fn)

        # Initialize persistent client + collection
        try:
            self.init_stage = "persistent_client"
            self._client = chromadb.PersistentClient(path=chroma_dir)

            self.init_stage = "collection"
            try:
                self._collection = self._client.get_or_create_collection(
                    name=collection,
                    embedding_function=embed_fn,
                )
            except TypeError:
                # Some Chroma versions don't accept embedding_function here.
                self._collection = self._client.get_or_create_collection(name=collection)
                # Fallback injection (works for older APIs)
                self._collection._embedding_function = embed_fn  # type: ignore[attr-defined]

            self.enabled = True
            self.init_stage = "ready"
            self.init_error = None
        except Exception as e:
            self.enabled = False
            self.init_stage = "collection_init"
            self.init_error = _short_exc(e)

    @staticmethod
    def _try_get_embedding_function() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """
        Try to locate an embedding function in the repo, without hardcoding dependencies.

        Returns:
            (embed_fn_or_none, diagnostic_string, source_path_or_none)

        Priority:
        - script/get_embedding_function.py
        - get_embedding_function.py
        """
        candidates = [
            os.path.join(REPO_ROOT, "script", "get_embedding_function.py"),
            os.path.join(REPO_ROOT, "get_embedding_function.py"),
        ]

        diagnostics: List[str] = []

        for p in candidates:
            if not os.path.exists(p):
                diagnostics.append(f"{p}: not found")
                continue

            try:
                spec = importlib.util.spec_from_file_location("get_embedding_function", p)
                if spec is None or spec.loader is None:
                    diagnostics.append(f"{p}: could not create import spec")
                    continue

                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                fn = getattr(mod, "get_embedding_function", None)
                if not callable(fn):
                    diagnostics.append(f"{p}: get_embedding_function not callable/missing")
                    continue

                embed_fn = fn()
                if embed_fn is None:
                    diagnostics.append(f"{p}: get_embedding_function() returned None")
                    continue

                return embed_fn, None, p

            except Exception as e:
                diagnostics.append(f"{p}: {_short_exc(e)}")

        return None, " | ".join(diagnostics), None

    def upsert_turn(self, turn_id: int, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._collection is None:
            return
        try:
            self._collection.upsert(
                ids=[f"t{turn_id}"],
                documents=[text],
                metadatas=[meta or {}],
            )
        except Exception:
            return

    def query_turn_ids(self, query: str, top_k: int) -> List[int]:
        if not self.enabled or self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[query], n_results=top_k)
            ids = (res.get("ids") or [[]])[0]
            out: List[int] = []
            for rid in ids:
                m = re.match(r"^t(\d+)$", str(rid))
                if m:
                    out.append(int(m.group(1)))
            return out
        except Exception:
            return []
