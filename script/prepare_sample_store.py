# prepare_sample_store.py

import argparse
import os
import shutil
import hashlib
import json
import yaml
from pathlib import Path
from chromadb import PersistentClient
from chromadb.config import Settings

from utils import load_config, load_yaml_file
from get_embedding_function import get_embedding_function
from dsl_utils import load_dsl_from_path


class SampleStoreBuilder:
    def __init__(self, dsl_name: str, data_path: str, chroma_path: str, collection_name: str):
        self.dsl_name = dsl_name
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embed_model = get_embedding_function()

        # ❌ Don’t create client yet!
        self.client = None
        self.collection = None

        script_dir = Path(__file__).resolve().parent

        # Import the DSL implementation
        tmp_path = os.path.abspath(script_dir / ".." / self.dsl_name)
        DSL = load_dsl_from_path(tmp_path, module_name=self.dsl_name)

        self.dsl_instance = DSL()
        print(f"🔧 Loaded DSL: {self.dsl_instance}")


    def normalize(self, text: str) -> str:
        #self.dsl_instance.set_debug(True)
        return self.dsl_instance.normalize_text(text.strip().lower())

    def init_chroma(self):
        # 🔁 Reinitialize Chroma AFTER reset
        self.client = PersistentClient(path=self.chroma_path, settings=Settings(allow_reset=True))
        print(f"🧭 Chroma client initialized at path: {self.chroma_path}")
        self.collection = self.client.get_or_create_collection(self.collection_name)


    def load_yaml_samples(self, filename):
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
    
        for sample in data:
            # Normalize flat samples into sub_samples
            if 'sub_samples' not in sample:
                sample['sub_samples'] = [{
                    'entities': sample.pop('entities'),
                    'context': sample.pop('context', {}),
                    'program': sample.pop('program')
                }]
    
            # Normalize context in all sub_samples
            for sub in sample['sub_samples']:
                context = sub.get('context', {})
                if isinstance(context, str):
                    try:
                        sub['context'] = json.loads(context)
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse sub_sample context: {context}")
                        sub['context'] = {}
    
        return data
    

    def build(self):
        self.init_chroma()
        
        samples = self.load_yaml_samples(self.data_path)
        
        texts, metas, ids = [], [], []

        # Set debug mode for the DSL instance
        #self.dsl_instance.set_debug(True)

        for s in samples:
            norm = self.normalize(s["utterance"])
            hash_input=f"{norm}"
            for ss in s["sub_samples"]:
                # During storage - convert entity list to pipe-delimited string
                sorted_entities = sorted(ss["entities"])  # Sort for reproducibility
                entities_string = "|" + "|".join(sorted_entities) + "|"  # Add boundary markers
                serialized_context = json.dumps(ss.get("context", None), sort_keys=True)
                hash_input += f"::{entities_string}::{serialized_context}"
            uid = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            serialized_ss = json.dumps(s["sub_samples"])
            
            texts.append(norm)
            ids.append(uid)
            metas.append({
                "utterance": s["utterance"],
                "sub_samples": serialized_ss
            })

        # Filter out existing
        existing = self.collection.get(include=[])
        existing_ids = set(existing["ids"])
        print(f"📦 Existing samples in collection '{self.collection_name}': {len(existing_ids)}")

        new_samples = [(i, texts[i], metas[i]) for i in range(len(samples)) if ids[i] not in existing_ids]

        if new_samples:
            print(f"👉 Adding new samples: {len(new_samples)}")
            new_ids = [ids[i] for i, _, _ in new_samples]
            new_texts = [text for _, text, _ in new_samples]
            new_metas = [meta for _, _, meta in new_samples]
            new_embeddings = [self.embed_model.embed_query(text) for text in new_texts]

            print(f"📝 Attempting to add {len(new_ids)} samples to collection {self.collection_name}")
            print(f"📄 First sample ID: {new_ids[0]}")

            try:
                self.collection.add(
                    ids=new_ids,
                    documents=new_texts,
                    metadatas=new_metas,
                    embeddings=new_embeddings
                )
            except Exception as e:
                print(f"❌ Failed to add samples: {e}")
                import traceback
                traceback.print_exc()

        else:
            print("✅ No new samples to add.")

    def reset_store(self):
        if os.path.exists(self.chroma_path):
            print(f"✨ Deleting: {self.chroma_path}")
            shutil.rmtree(self.chroma_path)
            print(f"✅ Deleted? {not os.path.exists(self.chroma_path)}")


# --- CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsl", type=str, required=True, help="DSL name (e.g., odsl, markdown)")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store.")
    args = parser.parse_args()

    dsl = args.dsl

    from pathlib import Path
    script_dir = Path(__file__).resolve().parent

    # # Import the DSL implementation
    # tmp_path = os.path.abspath(script_dir / ".." / dsl)
    # DSL = load_dsl_from_path(tmp_path, module_name=dsl)
    # 
    # dsl_instance = DSL()
    # print(f"🔧 Loaded DSL: {dsl_instance}")
    
    # Load env config
    config = load_config(script_dir / '../config/env')

    CHROMAPATH = config.get('CHROMAPATH')
    print(f"🔧 Config says CHROMAPATH = {CHROMAPATH}")
    chroma_path = CHROMAPATH + dsl + "/"

    data_path = config.get(dsl)
    data_path = data_path + "/samples.yaml"

    collection_name = f"{dsl}_samples"

    builder = SampleStoreBuilder(
        dsl_name=dsl,
        data_path=data_path,
        chroma_path=chroma_path,
        collection_name=collection_name
    )

    if args.reset:
        builder.reset_store()

    builder.build()


if __name__ == "__main__":
    main()
