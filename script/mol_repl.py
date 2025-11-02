#!/usr/bin/env python3
import os, sys, asyncio, re, json
sys.path.append(os.path.dirname(__file__))              # semantic_interpreter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # project root

from semantic_interpreter import SemanticInterpreter, LLMClient
# use your FastMCP tools from unitymol_copilot
sys.path.append(os.path.expanduser('~/unitymol_copilot'))
from mcp_server import validate_dsl, execute_dsl

def cleaned(line: str) -> str:
    # extract the cleaned DSL line exactly like example-usage did
    return line.strip()

async def main():
    si = SemanticInterpreter(LLMClient())   # config picks local RAG/bank first
    print("molREPL — type natural language, 'quit' to exit.")
    while True:
        q = input("NL> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        resp = si.interpret(q, entity_hint=None, context=None)
        dsl = cleaned(resp.cleaned_program)
        print("DSL>", dsl)
        v = await validate_dsl(dsl)
        if not v.get("ok"):
            print("Validator errors:", v.get("errors"))
            continue
        out = await execute_dsl(dsl)
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

