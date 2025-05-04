import os
import re
import hashlib
import mimetypes
import ollama
from collections import defaultdict
from bs4 import BeautifulSoup
import json
import yaml

__version__ = "1.4.0"

###
### GENERAL UTILITY FUNCTIONS
###

import logging

def get_logger(name="myutils", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

def replace_extension_with_thumbnail(path):
    """From a base filename, generate the corresponding thumbnail filename"""
    base, _ = os.path.splitext(path)  # Split filename and extension
    return base + "_t.png"

def replace_with_refdir(path, refdir):
    """replace everything up to the actual filename by a specified reference directory"""
    # Ensure refdir ends with "/"
    if not refdir.endswith("/"):
        refdir += "/"
    return re.sub(r'^.*/', refdir, path) if '/' in path else refdir + path

def calculate_md5(file_path):
    """Calculate MD5 checksum of a file."""
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def derive_url(file_name,ref_path):
    """Generate wisdom URL from the file name."""
    if not ref_path.endswith("/"):
        ref_path += "/"
    base_url = "https://data.mol3d.tech/"
    file_new = file_name.replace("//","/")
    file_new = file_new.replace(ref_path,base_url)
    return f"{file_new}"

def determine_file_type(file_path):
    """Determine whether the file is an image or a movie and return its type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("image"):
        return "img"
    elif mime_type and mime_type.startswith("video"):
        return "mov"
    else:
        return "unknown"


###
### Web-related
###

def extract_text_from_html(file_path):
    """Extracts raw text from an HTML file, preserving title if available."""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    title = soup.title.string.strip() if soup.title else os.path.basename(file_path)
    text = soup.get_text(separator=" ", strip=True)  # Extract visible text
    return title, text[:4000]  # Limit text to first 4000 chars
#    return title, text[:2000]  # Limit text to first 2000 chars

###
### Ollama-related
###

def summarize_html_content(title, content, model):
    """Uses Ollama to generate a brief summary of the HTML content."""
    prompt = f"Title: {title}\n\nContent:\n{content}\n\nSummarize this in one or two sentences in English:"
    response = ollama.chat(model, messages=[{"role": "user", "content": prompt}])
    # response = "toto"
    return response["message"]["content"] if "message" in response else "Summary not available."

###
### SEATABLE RELATED
###

def extract_unique_labels(base, table_name):
    """Extract all unique values from the 'type', 'category', and 'tags' columns in a SeaTable table."""
    rows = base.list_rows(table_name)

    types = set()
    categories = set()
    tags = set()

    for row in rows:
        if "type" in row and row["type"]:
            types.add(row["type"])

        if "category" in row and row["category"]:
            categories.add(row["category"])

        if "tags" in row and row["tags"]:
            # Assuming tags are stored as a list or comma-separated string
            if isinstance(row["tags"], list):
                tags.update(row["tags"])
            else:
                tags.update(map(str.strip, row["tags"].split(",")))

    return list(types), list(categories), list(tags)

def find_empty_column(base, table_name, column="index"):
    """Find empty entries in the specified column of a SeaTable table."""
    rows = base.list_rows(table_name)

    empty_rows = [row["_id"] for row in rows if not row.get(column)]

    if empty_rows:
        print(f"⚠️ Empty entries found in column '{column}' for rows: {empty_rows}")
    else:
        print(f"info:: ✅ No empty entries in column '{column}'.")

def find_duplicate_column(base, table_name, column="index"):
    """Find duplicate MD5 hashes in the SeaTable table."""
    rows = base.list_rows(table_name)

    my_map = defaultdict(list)  # Dictionary to store data

    # Collect MD5 hashes and their corresponding row IDs
    for row in rows:
        my_hash = row.get(column, "")
        if my_hash:
            my_map[my_hash].append(row["_id"])

    # Find and print duplicates
    duplicates_found = False
    for my_hash, row_ids in my_map.items():
        if len(row_ids) > 1:
            duplicates_found = True
            print(f"❌ Duplicate column {column} entry: {my_hash} found in rows {row_ids}")

    if not duplicates_found:
        print(f"info:: ✅ no duplicate {column} entries found.")

def find_duplicate_md5(base, table_name, md5_column="md5"):
    """Find duplicate MD5 hashes in the SeaTable table."""
    rows = base.list_rows(table_name)

    md5_map = defaultdict(list)  # Dictionary to store MD5 hashes and row IDs

    # Collect MD5 hashes and their corresponding row IDs
    for row in rows:
        md5_hash = row.get(md5_column, "")
        if md5_hash:
            md5_map[md5_hash].append(row["_id"])

    # Find and print duplicates
    duplicates_found = False
    for md5_hash, row_ids in md5_map.items():
        if len(row_ids) > 1:
            duplicates_found = True
            print(f"❌ Duplicate MD5: {md5_hash} found in rows {row_ids}")

    if not duplicates_found:
        print("info:: ✅ no duplicate MD5 hashes found.")

def match_tags(row_tags, query):
    """Evaluate a logical query string against a list of tags."""
    
    # Return True if query is empty
    if not query.strip():
        return True

    row_tags = set(row_tags)  # Convert list to a set for fast lookups

    # Replace logical operators with Python syntax
    query = re.sub(r'\bAND\b', ' and ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', ' or ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\b', ' not ', query, flags=re.IGNORECASE)

    # Ensure tag names are correctly wrapped as membership expressions
    query = re.sub(
        r'\b(\w+)\b',
        lambda m: f"('{m.group(1)}' in row_tags)" if m.group(1).lower() not in {"and", "or", "not"} else m.group(1),
        query
    )

    try:
        return eval(query, {"__builtins__": None}, {"row_tags": row_tags})
    except Exception as e:
        print(f"Error evaluating query: {e}")
        return False

###
### CONFIGURATION FILES
###
    
def load_config(config_file):
    """
    Reads the config file, handles any quoted values, and returns a dictionary of properties.
    """
    config = {}

    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or comments (lines starting with '#')
                if not line or line.startswith("#"):
                    continue

                # Split line into key and value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")  # Strip quotes around the value
                    config[key] = value
    except FileNotFoundError:
        print(f"Error: {config_file} not found.")
    except Exception as e:
        print(f"Error: {e}")

    return config

def load_yaml_file(file_path):
    """Load YAML data from file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return default

def load_json_file(filepath: str, default=None):
    """Load JSON data from file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}

def load_text_file(filepath: str, default=""):
    """Load text data from file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return default

def load_py_data(path: str, varname: str = "samples"):
    """
    Dynamically loads a variable from a standalone Python (.py) file located at an arbitrary path.

    This is useful when you have structured Python data (like a list of dicts)
    stored in a .py file outside the normal module search path, and you want to
    access its contents without modifying sys.path or converting to JSON.

    Parameters:
        path (str): Full filesystem path to the Python file (e.g. "/some/dir/data.py").
        varname (str): Name of the variable inside the module to load (default: "samples").

    Returns:
        Any: The value of the specified variable from the loaded module.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the specified variable is not found in the module.

    Example:
        >>> data = load_py_data("/my/project/sample_data.py", "samples")
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Step 1: Define the module name (any name, does not need to match file name)
    module_name = Path(path).stem

    # Step 2: Load the module from a file path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Step 3: Return the `samples` variable from that module
    return getattr(module, varname)

