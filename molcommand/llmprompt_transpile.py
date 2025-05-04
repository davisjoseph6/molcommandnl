import requests
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from pathlib import Path
script_dir = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(script_dir / "../script"))
from utils import load_config

debug_enabled = False

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        """Send prompt to LLM and return completion"""
        pass


class OllamaProvider(LLMProvider):
    """Provider for local Ollama models"""

    def __init__(self, model: str = "llama3", endpoint: str = "http://localhost:11434/api/generate"):
        # Load configuration
        if debug_enabled: print(f"[DBG] Loading config from {script_dir / '../config/env'}")
        config = load_config(script_dir / '../config/env')

        if (OLLAMA_HOST := config.get('OLLAMA_HOST')):
            self.endpoint = OLLAMA_HOST + "/api/generate"
            if debug_enabled: print(f"[DBG] Ollama host set via environment to {self.endpoint}")
        else:
            self.endpoint = endpoint
            if debug_enabled: print(f"[DBG] Using default Ollama endpoint: {self.endpoint}")

        if (QUERYLLM := config.get('QUERYLLM')):
            self.model = QUERYLLM
            if debug_enabled: print(f"[DBG] Ollama model set via environment to {self.model}")
        else:
            self.model = model
            if debug_enabled: print(f"[DBG] Using default Ollama model: {self.model}")

    def generate_completion(self, prompt: str) -> str:
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API call failed with status {response.status_code}: {response.text}")

        return response.json().get("response", "").strip()


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI's ChatGPT API"""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it explicitly.")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate_completion(self, prompt: str) -> str:
        response = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API call failed with status {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude API"""

    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it explicitly.")
        self.endpoint = "https://api.anthropic.com/v1/messages"

    def generate_completion(self, prompt: str) -> str:
        response = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API call failed with status {response.status_code}: {response.text}")

        return response.json()["content"][0]["text"].strip()


class DSLTranspiler:
    """Transpiler for DSL code using various LLM providers"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def create_prompt(self, dsl: str, target_lang: str) -> str:
        """Create a prompt for the LLM to translate DSL to target language"""
        return f"""
You are a specialized code transpiler. Your task is to translate the following domain-specific language (DSL) code to {target_lang}.
Follow these rules:
1. Translate the semantics of the DSL, not just the syntax
2. Use idiomatic {target_lang} patterns
3. Preserve all functionality
4. Include helpful comments
5. Return ONLY the translated code without any explanations or markdown formatting

Here is the DSL code to translate:

```
{dsl.strip()}
```

Translated {target_lang} code:
"""

    def transpile(self, dsl: str, target_lang: str = "PyMol") -> str:
        """Transpile DSL to target language using the configured LLM provider"""
        try:
            prompt = self.create_prompt(dsl, target_lang)
            if debug_enabled: print(f"\n[DBG] Full prompt is:\n{prompt.strip()}\n")
            print(f"[transpiler] Sending DSL to LLM provider:\n{dsl.strip()}\n")

            response = self.llm_provider.generate_completion(prompt)

            # Clean up the response by removing markdown code blocks if present
            if response.startswith("```"):
                # Find the first and last code block delimiters
                lines = response.split("\n")
                start_idx = 0
                while start_idx < len(lines) and not lines[start_idx].startswith("```"):
                    start_idx += 1

                # Skip language identifier if present
                if start_idx < len(lines) and lines[start_idx].startswith("```"):
                    start_idx += 1

                end_idx = len(lines) - 1
                while end_idx >= 0 and not lines[end_idx].startswith("```"):
                    end_idx -= 1

                if start_idx <= end_idx:
                    response = "\n".join(lines[start_idx:end_idx])
                else:
                    # Just remove markdown backticks if structure is not as expected
                    response = response.strip("`\n ")

            return response.strip()

        except Exception as e:
            raise RuntimeError(f"Transpilation failed: {e}")


# Factory function to create a transpiler with a specific provider
def create_transpiler(provider_type: str, **kwargs) -> DSLTranspiler:
    """Create a transpiler with the specified LLM provider"""
    provider_mapping = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider
    }

    if provider_type not in provider_mapping:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. Choose from: {', '.join(provider_mapping.keys())}")

    provider_class = provider_mapping[provider_type]
    provider = provider_class(**kwargs)
    return DSLTranspiler(provider)


# --- Main Example Usage ---
if __name__ == "__main__":
    dsl_code = """
add_structure(PDBID="1kx2")
r1 = add_representation(type="surface",selection="protein", color=Blue)
update_representation(ref=r1, isTransparent=True)
r2 = add_representation(type="cartoon", selection="protein", color=SecondaryStructure)
r3 = add_representation(type="sticks", selection="not protein", color=CPK)
"""

    # Example 1: Using Ollama (local)
    transpiler = create_transpiler("ollama", model="llama3.2:latest")
    python_code = transpiler.transpile(dsl_code, target_lang="PyMol molecular visualization software")
    print("[result] Transpiled code (using Ollama):\n")
    print(python_code)

    # # Example 2: Using OpenAI (requires API key)
    # # Uncomment to use OpenAI
    # transpiler = create_transpiler("openai", model="gpt-4")
    # python_code = transpiler.transpile(dsl_code, target_lang="PyMol molecular visualization software")
    # print("[result] Transpiled code (using OpenAI):\n")
    # print(python_code)

    # Example 3: Using Anthropic Claude (requires API key)
    # Uncomment to use Anthropic
    # transpiler = create_transpiler("anthropic", model="claude-3-opus-20240229")
    # python_code = transpiler.transpile(dsl_code, target_lang="C++")
    # print("[result] Transpiled code (using Claude):\n")
    # print(python_code)