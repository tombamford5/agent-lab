"""Deterministic unit tests for web_fetcher_server. HTTP is fully mocked."""
from unittest.mock import MagicMock

import httpx
import pytest

import web_fetcher_server
from web_fetcher_server import MAX_BYTES, _fetch_html, read_url, summarise_text


def _make_response(
    *,
    status_code: int = 200,
    content: bytes = b"<html></html>",
    content_type: str = "text/html; charset=utf-8",
) -> MagicMock:
    """Build a MagicMock that quacks like httpx.Response for our fetcher's use."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("utf-8", errors="replace")
    resp.headers = {"content-type": content_type}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"status {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class _Recorder:
    """Captures requests and replays a queue of responses (or exceptions)."""

    def __init__(self) -> None:
        self.queue: list = []
        self.requests: list[str] = []

    def add(self, **kwargs) -> None:
        self.queue.append(_make_response(**kwargs))

    def add_error(self, exc: Exception) -> None:
        self.queue.append(exc)


@pytest.fixture
def mock_httpx(monkeypatch):
    rec = _Recorder()

    class FakeClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url: str):
            rec.requests.append(url)
            if not rec.queue:
                raise RuntimeError(f"no mocked response for {url}")
            item = rec.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(web_fetcher_server.httpx, "Client", FakeClient)
    # Skip the retry backoff so tests stay fast and deterministic.
    monkeypatch.setattr(web_fetcher_server.time, "sleep", lambda _: None)
    return rec


# ── _fetch_html ────────────────────────────────────────────────────────────

def test_fetch_html_happy_path(mock_httpx):
    mock_httpx.add(content=b"<html><body>hello</body></html>")
    html = _fetch_html("https://example.com")
    assert "<body>hello</body>" in html
    assert mock_httpx.requests == ["https://example.com"]


def test_fetch_html_rejects_unsupported_content_type(mock_httpx):
    mock_httpx.add(content=b"%PDF-1.4", content_type="application/pdf")
    with pytest.raises(ValueError, match="unsupported content-type"):
        _fetch_html("https://example.com/x.pdf")


def test_fetch_html_accepts_xhtml(mock_httpx):
    mock_httpx.add(
        content=b"<html xmlns='http://www.w3.org/1999/xhtml'><body>x</body></html>",
        content_type="application/xhtml+xml",
    )
    html = _fetch_html("https://example.com")
    assert "<body>x</body>" in html


def test_fetch_html_content_type_is_case_insensitive(mock_httpx):
    # Some upstreams send "TEXT/HTML" or "Text/HTML; charset=..." — should match.
    mock_httpx.add(content=b"<html>ok</html>", content_type="TEXT/HTML; charset=UTF-8")
    assert "ok" in _fetch_html("https://example.com")


def test_fetch_html_rejects_missing_content_type(mock_httpx):
    # Server omitted Content-Type entirely → empty string after .get() default.
    mock_httpx.add(content=b"<html></html>", content_type="")
    with pytest.raises(ValueError, match="unsupported content-type"):
        _fetch_html("https://example.com")


def test_fetch_html_rejects_oversized_response(mock_httpx):
    mock_httpx.add(content=b"x" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="response too large"):
        _fetch_html("https://example.com")


def test_fetch_html_retries_on_5xx_then_succeeds(mock_httpx):
    mock_httpx.add(status_code=503, content=b"")
    mock_httpx.add(status_code=502, content=b"")
    mock_httpx.add(content=b"<html>ok</html>")
    html = _fetch_html("https://example.com")
    assert "ok" in html
    assert len(mock_httpx.requests) == 3


def test_fetch_html_gives_up_after_three_5xx(mock_httpx):
    for _ in range(3):
        mock_httpx.add(status_code=503, content=b"")
    with pytest.raises(httpx.HTTPStatusError):
        _fetch_html("https://example.com")
    assert len(mock_httpx.requests) == 3


def test_fetch_html_retries_on_network_error(mock_httpx):
    mock_httpx.add_error(httpx.ConnectError("dns fail"))
    mock_httpx.add(content=b"<html>ok</html>")
    html = _fetch_html("https://example.com")
    assert "ok" in html
    assert len(mock_httpx.requests) == 2


def test_fetch_html_retries_on_read_timeout(mock_httpx):
    mock_httpx.add_error(httpx.ReadTimeout("read timeout"))
    mock_httpx.add(content=b"<html>ok</html>")
    html = _fetch_html("https://example.com")
    assert "ok" in html
    assert len(mock_httpx.requests) == 2


def test_fetch_html_gives_up_after_three_timeouts(mock_httpx):
    for _ in range(3):
        mock_httpx.add_error(httpx.ReadTimeout("read timeout"))
    with pytest.raises(httpx.ReadTimeout):
        _fetch_html("https://example.com")
    assert len(mock_httpx.requests) == 3


def test_fetch_html_propagates_persistent_network_error(mock_httpx):
    for _ in range(3):
        mock_httpx.add_error(httpx.ConnectError("dns fail"))
    with pytest.raises(httpx.ConnectError):
        _fetch_html("https://example.com")


def test_fetch_html_does_not_retry_on_4xx(mock_httpx):
    mock_httpx.add(status_code=404, content=b"not found")
    with pytest.raises(httpx.HTTPStatusError):
        _fetch_html("https://example.com")
    assert len(mock_httpx.requests) == 1


def test_fetch_html_sends_custom_user_agent(mock_httpx, monkeypatch):
    seen_headers = {}

    class CapturingClient:
        def __init__(self, **kwargs):
            seen_headers.update(kwargs.get("headers", {}))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            return _make_response(content=b"<html></html>")

    monkeypatch.setattr(web_fetcher_server.httpx, "Client", CapturingClient)
    _fetch_html("https://example.com")
    assert "agent-lab" in seen_headers.get("User-Agent", "")


# ── read_url ──────────────────────────────────────────────────────────────

ARTICLE_HTML = """
<html>
  <head><title>Test Article</title></head>
  <body>
    <nav>nav junk that should be stripped</nav>
    <article>
      <h1>Test Article</h1>
      <p>This is a paragraph with substantial content so that trafilatura
         recognises it as the main body of the article. The sentence is
         long enough to clear the extractor's minimum-text heuristic.</p>
      <p>A second paragraph here to give the article some real substance,
         again with enough words to look like real content rather than
         boilerplate or chrome.</p>
    </article>
    <footer>footer junk</footer>
  </body>
</html>
"""


def test_read_url_happy_path(mock_httpx):
    mock_httpx.add(content=ARTICLE_HTML.encode())
    result = read_url("https://example.com/article")
    assert "error" not in result
    assert "warning" not in result
    assert result["title"] == "Test Article"
    assert "trafilatura" in result["text"]
    assert "nav junk" not in result["text"]
    assert "footer junk" not in result["text"]
    assert result["truncated"] is False
    assert result["url"] == "https://example.com/article"


def test_read_url_returns_error_on_fetch_failure(mock_httpx):
    for _ in range(3):
        mock_httpx.add(status_code=503, content=b"")
    result = read_url("https://example.com")
    assert "error" in result
    assert "fetch failed" in result["error"]


def test_read_url_returns_error_on_bad_content_type(mock_httpx):
    mock_httpx.add(content=b"%PDF-1.4", content_type="application/pdf")
    result = read_url("https://example.com/x.pdf")
    assert "error" in result
    assert "unsupported content-type" in result["error"]


def test_read_url_returns_error_on_invalid_url(mock_httpx):
    # httpx.InvalidURL is raised synchronously by client.get() — it does NOT
    # inherit from HTTPError, so the retry loop won't catch it. read_url's
    # broad `except Exception` should still surface it as a clean error dict.
    mock_httpx.add_error(httpx.InvalidURL("malformed url"))
    result = read_url("htp://broken")
    assert "error" in result
    assert "fetch failed" in result["error"]
    assert len(mock_httpx.requests) == 1  # no retries for non-HTTPError


def test_read_url_warning_when_no_main_text(mock_httpx):
    spa_html = (
        "<html><head><title>JS App</title></head>"
        "<body><div id='root'></div></body></html>"
    )
    mock_httpx.add(content=spa_html.encode())
    result = read_url("https://spa.example/app")
    assert result["text"] == ""
    assert result["warning"] == "no main text extracted"
    assert result["title"] == "JS App"
    assert result["url"] == "https://spa.example/app"
    assert "error" not in result


def test_read_url_warning_preserves_title_when_metadata_present(mock_httpx):
    html = (
        "<html><head>"
        "<title>Article Headline</title>"
        "<meta name='author' content='Jane Doe'>"
        "</head><body></body></html>"
    )
    mock_httpx.add(content=html.encode())
    result = read_url("https://example.com/empty")
    assert result["warning"] == "no main text extracted"
    assert result["title"] == "Article Headline"


def test_read_url_truncates_long_text(mock_httpx, monkeypatch):
    monkeypatch.setattr(web_fetcher_server, "MAX_TEXT_CHARS", 200)
    long_paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50
    html = f"<html><body><article><p>{long_paragraph}</p></article></body></html>"
    mock_httpx.add(content=html.encode())
    result = read_url("https://example.com/long")
    assert result["truncated"] is True
    assert result["text"].endswith("[truncated]")
    # extracted text body capped at the limit; "\n\n[truncated]" appended
    assert len(result["text"]) <= 200 + len("\n\n[truncated]")


# ── summarise_text ────────────────────────────────────────────────────────

def test_summarise_text_short_circuits_on_empty_input(monkeypatch):
    # If the API is called at all, fail loudly — a "return canned string but
    # still hit the API" refactor would otherwise pass silently while burning credit.
    def fake_create(**_kwargs):
        raise AssertionError("API must not be called for empty input")

    monkeypatch.setattr(web_fetcher_server.anthropic.messages, "create", fake_create)
    assert summarise_text("") == "(no text to summarise)"
    assert summarise_text("   \n\t  ") == "(no text to summarise)"


def test_summarise_text_calls_haiku_with_max_words_in_prompt(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = "mocked summary."
        resp.content = [block]
        return resp

    monkeypatch.setattr(
        web_fetcher_server.anthropic.messages, "create", fake_create
    )
    out = summarise_text("Some real input text.", max_words=42)
    assert out == "mocked summary."
    assert captured["model"] == "claude-haiku-4-5-20251001"
    user_msg = captured["messages"][0]["content"]
    assert "42 words" in user_msg
    assert "Some real input text." in user_msg
