import os
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings

from utils import load_config


def get_embedding_function():
    """Get the configured embedding function based on environment settings"""
    # Load configuration
    script_dir = Path(__file__).resolve().parent
    config = load_config(script_dir / '../config/env')
    
    # Get model name from config with fallback
    model_name = config.get('EMBEDDINGMDL', 'nomic-embed-text')
    
    # Configure Ollama with optional host setting
    kwargs = {}
    if "OLLAMA_HOST" in os.environ:
        kwargs['base_url'] = os.environ["OLLAMA_HOST"]
        
    return OllamaEmbeddings(
        model=model_name,
        **kwargs
    )