"""A minimal MCP server exposing two note-taking tools."""
from pathlib import Path
from mcp.server.fastmcp import FastMCP

NOTES_FILE = Path(__file__).parent / "notes.md"

mcp = FastMCP("notes")

@mcp.tool()
def add_note(text: str) -> str:
    """Append a single note to the notes file. Returns confirmation."""
    NOTES_FILE.touch(exist_ok=True)
    with NOTES_FILE.open("a") as f:
        f.write(f"- {text}\n")
    return f"Added: {text}"

@mcp.tool()
def list_notes() -> str:
    """Return the full contents of the notes file."""
    if not NOTES_FILE.exists():
        return "(no notes yet)"
    return NOTES_FILE.read_text()

if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport