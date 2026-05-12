"""Agent loop that gets its tools from one or more MCP servers. Stage 2 of the lab."""
import asyncio
import json
from contextlib import AsyncExitStack
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
anthropic = Anthropic()
MODEL = "claude-opus-4-7"

SERVERS = [
    ("notes", ["uv", "run", "python", "notes_server.py"]),
    ("web-fetcher", ["uv", "run", "python", "web_fetcher_server.py"]),
]


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


async def agent(
    tool_to_session: dict[str, ClientSession],
    tools: list[dict],
    user_prompt: str,
    max_turns: int = 10,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]

    for _ in range(max_turns):
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
                    # Truncate input echo so a huge summarise_text(text=...) doesn't flood the log
                    shown = json.dumps(block.input)
                    if len(shown) > 200:
                        shown = shown[:200] + "..."
                    print(f"[tool] {block.name}({shown})")
                    session = tool_to_session[block.name]
                    result = await session.call_tool(block.name, block.input)
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
    async with AsyncExitStack() as stack:
        # Spawn every MCP server as a subprocess and open a session to each.
        # TODO: when we add a third server, lift this routing dict + the loop
        # below into a small MCPRouter class (owns the ExitStack, exposes
        # .tools and .call(name, input)). Two servers isn't worth the abstraction yet.
        tool_to_session: dict[str, ClientSession] = {}
        all_tools: list[dict] = []

        for label, cmd in SERVERS:
            params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_result = await session.list_tools()
            anthropic_tools = mcp_tools_to_anthropic(tools_result.tools)
            for t in tools_result.tools:
                tool_to_session[t.name] = session
            all_tools.extend(anthropic_tools)
            print(f"[setup] {label}: {[t['name'] for t in anthropic_tools]}")

        print(f"[setup] {len(all_tools)} tools total across {len(SERVERS)} servers\n")

        prompt = (
            "Fetch https://modelcontextprotocol.io, summarise it in about 150 words, "
            "then save the summary as a note. Confirm when done."
        )
        answer = await agent(tool_to_session, all_tools, prompt)
        print("─" * 60)
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
