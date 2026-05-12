"""Behavioural eval for summarise_text.

Re-run whenever the summariser prompt (or the model) changes, to catch
regressions in length compliance, key-term coverage, and prose style.

Usage:  uv run python eval_summariser.py
Exit code is non-zero if any case fails any required property.

Each call hits the real Anthropic API (Haiku 4.5) — cheap, but not free.
"""
import re
import sys
from dataclasses import dataclass, field

from web_fetcher_server import summarise_text


@dataclass
class Case:
    name: str
    text: str
    max_words: int
    must_mention: list[str]
    length_slack: float = 1.5  # allow 50% over the target before failing


CASES: list[Case] = [
    Case(
        name="mcp_intro",
        max_words=80,
        must_mention=["MCP"],
        text=(
            "The Model Context Protocol (MCP) is an open standard for connecting "
            "AI applications to external systems. It defines a uniform way for AI "
            "clients like Claude Desktop and Cursor to discover and call tools "
            "exposed by servers. A typical MCP server runs as a subprocess and "
            "communicates over stdio using JSON-RPC. Servers expose tools "
            "(callable functions), resources (data the model can read), and "
            "prompts (templates). The protocol was introduced by Anthropic in "
            "late 2024 and has been adopted by multiple vendors including OpenAI."
        ),
    ),
    Case(
        name="python_gil",
        max_words=60,
        must_mention=["GIL", "Python"],
        text=(
            "Python's Global Interpreter Lock (GIL) is a mutex that protects "
            "access to Python objects, preventing multiple native threads from "
            "executing Python bytecodes at once. This was originally a "
            "simplification for CPython's memory management. The GIL means that "
            "threading is mainly useful for I/O-bound work rather than CPU-bound "
            "work — for true parallelism, multiprocessing is the traditional "
            "answer. Python 3.13 introduced an experimental no-GIL build mode, "
            "expected to mature in subsequent releases."
        ),
    ),
    Case(
        name="wine_note",
        max_words=50,
        must_mention=["Barolo"],
        text=(
            "A 2018 Barolo from Piedmont shows classic Nebbiolo character — "
            "pale garnet in the glass, with a perfumed nose of rose petals, tar, "
            "and dried cherries. On the palate the wine is firm and structured, "
            "with grippy tannins and bright acidity that promise another decade "
            "of cellaring. Notes of liquorice and forest floor develop with air. "
            "Drink with braised meats or aged hard cheeses; the wine needs food "
            "to show its best."
        ),
    ),
]


# Phrases an LLM uses when meta-commenting instead of just summarising.
PREAMBLE_PATTERNS = [
    r"^here\s+(is|are)\s+(a|the)\s+(brief\s+)?summary",
    r"^this\s+(text|article|passage|piece)\b",
    r"^the\s+(text|article|passage|piece)\s+(discusses|describes|explains|covers)",
    r"^in\s+summary\b",
    r"^summary\s*:",
    r"^to\s+summari[sz]e\b",
]

# Markdown-ish structure the prompt says not to produce: headings, bullets, bold-labels.
STRUCTURE_PATTERN = re.compile(
    r"^\s*#{1,6}\s|^\s*\*\*[^*]+\*\*\s*:|^\s*[-*]\s|^\s*\d+\.\s",
    re.MULTILINE,
)


def check_length(summary: str, case: Case) -> tuple[bool, str]:
    words = len(summary.split())
    cap = int(case.max_words * case.length_slack)
    return words <= cap, f"{words} words (target ≤{case.max_words}, hard cap {cap})"


def check_must_mention(summary: str, case: Case) -> tuple[bool, str]:
    lower = summary.lower()
    missing = [t for t in case.must_mention if t.lower() not in lower]
    return not missing, "all present" if not missing else f"missing {missing}"


def check_no_preamble(summary: str, _case: Case) -> tuple[bool, str]:
    head = summary.strip().lower()
    for pat in PREAMBLE_PATTERNS:
        if re.match(pat, head):
            return False, f"opens with preamble matching /{pat}/"
    return True, "clean opening"


def check_no_structure(summary: str, _case: Case) -> tuple[bool, str]:
    m = STRUCTURE_PATTERN.search(summary)
    return (m is None), "plain prose" if m is None else f"found structure: {m.group()!r}"


CHECKS = [
    ("length", check_length),
    ("must_mention", check_must_mention),
    ("no_preamble", check_no_preamble),
    ("no_structure", check_no_structure),
]


@dataclass
class CaseResult:
    name: str
    summary: str
    results: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return all(ok for ok, _ in self.results.values())


def run_case(case: Case) -> CaseResult:
    print(f"\n── {case.name} (max_words={case.max_words}) ──")
    summary = summarise_text(case.text, max_words=case.max_words)
    print(f"  output: {summary}")
    cr = CaseResult(name=case.name, summary=summary)
    for label, fn in CHECKS:
        ok, detail = fn(summary, case)
        cr.results[label] = (ok, detail)
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}: {detail}")
    return cr


def main() -> int:
    print("Behavioural eval for summarise_text")
    print("=" * 60)

    all_results = [run_case(c) for c in CASES]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_checks = 0
    total_passed = 0
    for cr in all_results:
        passed = sum(1 for ok, _ in cr.results.values() if ok)
        total = len(cr.results)
        total_checks += total
        total_passed += passed
        marker = "PASS" if cr.all_pass else "FAIL"
        print(f"  [{marker}] {cr.name}: {passed}/{total} properties")

    print(f"\nOverall: {total_passed}/{total_checks} checks passed")
    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
