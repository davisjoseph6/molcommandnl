#!/usr/bin/env python3
"""
import_raw_scripts.py

Ingest raw PyMOL (.pml) and VMD (.tcl) scripts into a separate Chroma collection.

Design:
- Keep molcommand_samples for "utterance -> DSL" pairs (execution-oriented).
- Store raw scripts in molcommand_raw_scripts (retrieval/context-oriented).

This avoids polluting DSL training with raw scripts.
"""

from __future__ import annotations

import os
import argparse
import hashlib
from pathlib import Path

from chromadb import PersistentClient
from langchain_community.embeddings import OllamaEmbeddings


def file_sha_id(source: str, relpath: str) -> str:
    s = f"{source}:{relpath}".encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pymol", "vmd"], required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--collection", default="molcommand_raw_scripts")
    ap.add_argument("--chroma-dir", default=os.path.expanduser("~/.cache/molcommandnl/chroma"))
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    indir = Path(args.input_dir).expanduser().resolve()
    assert indir.exists() and indir.is_dir(), f"Not a directory: {indir}"

    # Ollama base url is optional (your llm_client.py already supports OLLAMA_HOST)
    base_url = os.environ.get("OLLAMA_HOST")
    embed = OllamaEmbeddings(
        model=args.embed_model,
        **({"base_url": base_url} if base_url else {})
    )

    client = PersistentClient(path=args.chroma_dir)
    col = client.get_or_create_collection(args.collection)

    exts = {".pml"} if args.source == "pymol" else {".tcl"}

    files = [fp for fp in indir.rglob("*") if fp.is_file() and fp.suffix.lower() in exts]
    print(f"Found {len(files)} {args.source} files under {indir}")

    ids, docs, metas, embs = [], [], [], []

    def flush():
        if not ids:
            return
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        ids.clear(); docs.clear(); metas.clear(); embs.clear()

    for fp in files:
        rel = str(fp.relative_to(indir))
        raw = read_text(fp).strip()
        if not raw:
            continue

        doc = (
            f"SOURCE: {args.source}\n"
            f"FILE: {rel}\n"
            f"EXT: {fp.suffix.lower()}\n\n"
            f"{raw}\n"
        )

        # Embedding text: include a tiny header so queries can match by filename/source too
        emb_text = f"{args.source} {rel}\n{raw}"
        vec = embed.embed_query(emb_text)

        ids.append(file_sha_id(args.source, rel))
        docs.append(doc)
        metas.append({
            "source": args.source,
            "file": rel,
            "ext": fp.suffix.lower(),
            "bytes": fp.stat().st_size,
        })
        embs.append(vec)

        if len(ids) >= args.batch_size:
            flush()

    flush()
    print("Done. Collection:", args.collection, "count =", col.count())


if __name__ == "__main__":
    main()
