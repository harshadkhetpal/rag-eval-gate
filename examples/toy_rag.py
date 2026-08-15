"""A tiny fake RAG system the examples and CI run against.

It "retrieves" from a hard-coded corpus and "generates" by returning the stored
answer. The point is to make the gate demonstrable end to end - swap `answer` for a
call into your real pipeline and everything else stays the same.
"""
from __future__ import annotations

from rageval.runner import Answer

_CORPUS = {
    "refund": Answer(
        text="Refunds are processed within 5 business days to the original payment method.",
        sources=("policies/refunds.md",),
    ),
    "sla": Answer(
        text="The API uptime SLA is 99.9 percent, with service credits below that.",
        sources=("legal/sla.md",),
    ),
    "gdpr": Answer(
        text="Export requests are fulfilled within 30 days as required by GDPR.",
        sources=("legal/gdpr.md", "policies/data-export.md"),
    ),
}


def answer(question: str) -> Answer:
    q = question.lower()
    for key, stored in _CORPUS.items():
        if key in q:
            return stored
    return Answer(text="I could not find that in the documentation.", sources=())


def broken_answer(question: str) -> Answer:
    """The same system after a simulated bad deploy - refund answers lost the
    timeframe fact. Used by tests and the README to show the gate catching it."""
    result = answer(question)
    if "refund" in question.lower():
        return Answer(
            text="Refunds go back to the original payment method.",
            sources=result.sources,
        )
    return result
