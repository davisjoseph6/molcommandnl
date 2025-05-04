import requests
import json
from typing import List, Dict, Any


class Transpiler:
    endpoint: str = "https://chatmol.org/qa/lite/"

    def transpile(self, dsl: str) -> str:
        """Transpile a DSL to target language code via external API"""
        try:
            return self.transpile_dsl(dsl)
        except KeyError as e:
            raise ValueError(f"Unsupported command type: {e}")
        except Exception as e:
            raise RuntimeError(f"Transpilation failed: {e}")

    def transpile_dsl(self, dsl: str) -> str:
        """Send DSL to ChatMol API and return the 'answer' portion of the JSON response"""
        prompt = dsl.strip()
        print(f"[transpiler] Sending DSL to ChatMol:\n{prompt}\n")
        prompt = f"Instructions: {prompt}"

        response = requests.post(
            self.endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"question": prompt}
        )

        if response.status_code != 200:
            raise RuntimeError(f"API call failed with status {response.status_code}: {response.text}")

        try:
            response_json = response.json()
            answer = response_json.get("answer", "").strip()

            # If the answer is a code block, strip the markdown-style backticks
            if answer.startswith("```"):
                answer = "\n".join(answer.strip("`\n").splitlines()[1:])  # remove ```lang and closing ```

            return answer
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse JSON response")


# --- Main Example Usage ---
if __name__ == "__main__":
    dsl_code = """
add_structure(PDBID="1kx2")
r1 = add_representation(type="surface",selection="protein", color=Blue)
update_representation(ref=r1, isTransparent=True)
r2 = add_representation(type="cartoon", selection="protein", color=SecondaryStructure)
r3 = add_representation(type="sticks", selection="not protein", color=CPK)
"""
    transpiler = Transpiler()
    output = transpiler.transpile(dsl_code)
    print("[result] Transpiled code:\n")
    print(output)
