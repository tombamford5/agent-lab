"""Minimal agent loop: Claude with two tools, no framework."""
import os, json, subprocess
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a text file from the local filesystem and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_bash",
        "description": "Run a bash command and return stdout+stderr. Use sparingly.",
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
]

def dispatch(name: str, args: dict) -> str:
    if name == "read_file":
        return Path(args["path"]).read_text()
    if name == "run_bash":
        r = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=30)
        return f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    return f"Unknown tool: {name}"

def agent(user_prompt: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    for turn in range(max_turns):
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        print(resp)

        if resp.stop_reason == "end_turn":
            # Extract the text reply
            return "".join(b.text for b in resp.content if b.type == "text")

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    print(f"[tool] {block.name}({json.dumps(block.input)})")
                    try:
                        output = dispatch(block.name, block.input)
                    except Exception as e:
                        output = f"ERROR: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:8000],  # truncate
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        break
    return "Hit max turns."

if __name__ == "__main__":
    print(agent("List the Python files in this directory and summarise what each one does."))