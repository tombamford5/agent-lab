"""MCP server: fetch a URL, extract main text, summarise via Claude Haiku."""
import json
import time
import urllib.robotparser
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
anthropic = Anthropic()

mcp = FastMCP("web-fetcher")

USER_AGENT = "agent-lab/0.1 (learning MCP)"
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 10.0
MAX_TEXT_CHARS = 50_000
SUMMARISER_MODEL = "claude-haiku-4-5-20251001"
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")


def _robots_allows(url: str) -> bool:
    """Return True if robots.txt permits fetching url (fail-open on errors)."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        with httpx.Client(timeout=5.0, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(robots_url)
    except httpx.HTTPError:
        return True
    if resp.status_code >= 400:
        return True
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())
    return rp.can_fetch(USER_AGENT, url)


def _fetch_html(url: str) -> str:
    """Fetch url. Retries on 5xx, enforces content-type allowlist and size cap."""
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
                headers=headers,
            ) as client:
                resp = client.get(url)
        except httpx.HTTPError as e:
            last_error = e
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise

        if 500 <= resp.status_code < 600 and attempt < 2:
            time.sleep(0.5 * (attempt + 1))
            continue
        resp.raise_for_status()

        ctype = resp.headers.get("content-type", "").lower()
        if not any(allowed in ctype for allowed in ALLOWED_CONTENT_TYPES):
            raise ValueError(f"unsupported content-type: {ctype!r}")
        if len(resp.content) > MAX_BYTES:
            raise ValueError(
                f"response too large: {len(resp.content)} bytes (cap {MAX_BYTES})"
            )
        return resp.text

    raise last_error if last_error else RuntimeError("unreachable")


@mcp.tool()
def read_url(url: str) -> dict[str, Any]:
    """Fetch a URL and extract its main text content (strips nav/ads/boilerplate).

    Returns a dict with: url, title, text, author, date, truncated.
    On failure returns {"error": "..."}.
    """
    if not _robots_allows(url):
        return {"error": f"disallowed by robots.txt or invalid url: {url}"}
    try:
        html = _fetch_html(url)
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    extracted = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
    )
    if not extracted:
        return {"error": "no main content found"}

    data = json.loads(extracted)
    text = (data.get("text") or "").strip()
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS] + "\n\n[truncated]"

    return {
        "url": data.get("url") or url,
        "title": data.get("title"),
        "text": text,
        "author": data.get("author"),
        "date": data.get("date"),
        "truncated": truncated,
    }


@mcp.tool()
def summarise_text(text: str, max_words: int = 200) -> str:
    """Summarise text using Claude Haiku. max_words is a target length."""
    if not text.strip():
        return "(no text to summarise)"
    prompt = (
        f"Summarise the following text in at most {max_words} words. "
        f"Plain prose, no preamble, no headings.\n\n{text}"
    )
    resp = anthropic.messages.create(
        model=SUMMARISER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    mcp.run()
