#!/usr/bin/env python3
import os, json, argparse, sys
from pathlib import Path

# --- repo paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "molcommand"))  # for dsl_definition
sys.path.append(str(HERE))                 # for sharing env utils if needed

# Load DSL to get collection name and optional CHROMA_PATH on DSL
try:
    from dsl_definition import DSL
except Exception as e:
    print(f"[index_bank] ERROR: cannot import molcommand.dsl_definition.DSL: {e}")
    sys.exit(1)

# Prefer new Chroma API; fall back to legacy
try:
    from chromadb import PersistentClient as ChromaPersistentClient
except Exception:
    ChromaPersistentClient = None

# We’ll embed utterances with Ollama, same as the interpreter
from langchain_community.embeddings.ollama import OllamaEmbeddings

def default_persist_dir():
    return (
        os.environ.get("MOLCOMMANDNL_CHROMA_DIR")
        or os.environ.get("CHROMA_PATH")
        or os.path.expanduser("~/.cache/molcommandnl/chroma")
    )

def get_client(persist_dir):
    client = None
    err_new = err_old = None
    if ChromaPersistentClient is not None:
        try:
            client = ChromaPersistentClient(path=persist_dir)
        except Exception as e:
            err_new = e
    if client is None:
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_dir
            ))
        except Exception as e:
            err_old = e
    if client is None:
        raise RuntimeError(f"Failed to init Chroma (new={err_new}; legacy={err_old})")
    return client

def load_samples_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    # normalize structure: [{"utterance": "...", "sub_samples":[{program,entities,context?}], ...}]
    norm = []
    for item in data:
        utt = item.get("utterance")
        subs = item.get("sub_samples") or []
        if not utt or not subs:
            continue
        norm.append({"utterance": utt, "sub_samples": subs})
    return norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=str(ROOT / "molcommand" / "samples.yaml"),
                    help="Path to samples.yaml (default: molcommand/samples.yaml)")
    ap.add_argument("--persist", default=default_persist_dir(),
                    help="Chroma persist dir (default: ~/.cache/molcommandnl/chroma or env)")
    ap.add_argument("--reset", action="store_true",
                    help="Drop and recreate the collection before indexing")
    args = ap.parse_args()

    persist_dir = os.path.abspath(args.persist)
    os.makedirs(persist_dir, exist_ok=True)

    dsl = DSL()
    coll_name = dsl.get_collection_name()
    print(f"[index_bank] persist_dir={persist_dir}")
    print(f"[index_bank] collection = {coll_name}")
    print(f"[index_bank] loading YAML: {args.yaml}")

    samples = load_samples_yaml(args.yaml)
    if not samples:
        print("[index_bank] nothing to index (0 samples) — aborting.")
        sys.exit(1)

    client = get_client(persist_dir)

    if args.reset:
        try:
            client.delete_collection(coll_name)
            print(f"[index_bank] dropped existing collection '{coll_name}'")
        except Exception:
            pass

    # get/create collection
    if hasattr(client, "get_or_create_collection"):
        coll = client.get_or_create_collection(coll_name)
    else:
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            coll = client.create_collection(coll_name)

    # Prepare docs
    utterances = [s["utterance"] for s in samples]
    metadatas = [{"utterance": s["utterance"], "sub_samples": json.dumps(s["sub_samples"])} for s in samples]
    ids = [f"s:{i}" for i in range(len(samples))]

    # Embeddings
    embed = OllamaEmbeddings(
        model="nomic-embed-text",
        **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
    ).embed_documents
    embeddings = embed(utterances)

    # Add
    coll.add(documents=utterances, metadatas=metadatas, ids=ids, embeddings=embeddings)

    try:
        n = coll.count()
    except Exception:
        # fallback count
        n = len(coll.get(include=[]).get("ids", []))
    print(f"[index_bank] done. collection size now: {n}")

if __name__ == "__main__":
    main()

