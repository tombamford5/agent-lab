"""Agent loop that gets its tools from an MCP server. Stage 2 of the lab."""
import asyncio
import json
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
anthropic = Anthropic()
MODEL = "claude-opus-4-7"


def mcp_tools_to_anthropic(mcp_tools) -> list[dict]:
    """Translate an MCP ListToolsResult into Anthropic's tools=[...] schema."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def agent(session: ClientSession, user_prompt: str, max_turns: int = 10) -> str:
    # Step 1: ask the MCP server what tools it has, translate them for Claude
    tools_result = await session.list_tools()
    tools = mcp_tools_to_anthropic(tools_result.tools)
    print(f"[setup] MCP server exposes {len(tools)} tools: "
          f"{[t['name'] for t in tools]}\n")

    messages = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        resp = anthropic.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            return "".join(b.text for b in resp.content if b.type == "text")

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    print(f"[tool] {block.name}({json.dumps(block.input)})")
                    # Step 2: route the call through MCP, not a local dispatcher
                    result = await session.call_tool(block.name, block.input)
                    # MCP results come back as a list of content blocks; flatten to text
                    output = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    print(f"[result] {output[:120]}{'...' if len(output) > 120 else ''}\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:8000],
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return "Hit max turns."


async def main():
    print(f"[start] {datetime.now().isoformat(timespec='seconds')}")
    # Spawn the MCP server as a subprocess and talk to it over stdio
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "notes_server.py"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            prompt = (
                "Add three notes: 'learn MCP', 'build agent loop', 'connect them'. "
                "Then list all notes and tell me how many there are."
            )
            answer = await agent(session, prompt)
            print("─" * 60)
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())