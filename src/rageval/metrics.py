"""Scoring metrics for RAG answers.

Every metric here is deterministic and runs offline. That is a deliberate constraint:
a quality gate that calls an LLM to judge answers is itself non-deterministic, so a
build can fail because the judge had a bad day rather than because the system got
worse. Non-deterministic judges belong in analysis, not in a merge gate.

Each metric returns a float in [0.0, 1.0], higher is better.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Set

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Words too common to carry signal when comparing short answers.
_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the",
    "their", "then", "there", "they", "this", "to", "was", "were", "will", "with",
}


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def tokens(text: str, *, drop_stopwords: bool = True) -> List[str]:
    words = normalise(text).split()
    if drop_stopwords:
        words = [w for w in words if w not in _STOPWORDS]
    return words


def exact_match(answer: str, expected: str) -> float:
    """1.0 only if the normalised strings are identical."""
    return 1.0 if normalise(answer) == normalise(expected) else 0.0


def token_f1(answer: str, expected: str) -> float:
    """Harmonic mean of token precision and recall - the standard QA overlap score.

    Chosen over raw accuracy because RAG answers are free text: an answer can be
    correct while being phrased differently, and F1 degrades gracefully where exact
    match falls off a cliff.
    """
    a, e = tokens(answer), tokens(expected)
    if not a and not e:
        return 1.0
    if not a or not e:
        return 0.0

    # Multiset intersection: repeated words should only match as often as they occur.
    remaining = list(e)
    overlap = 0
    for tok in a:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(a)
    recall = overlap / len(e)
    return 2 * precision * recall / (precision + recall)


def contains_all(answer: str, required: Sequence[str]) -> float:
    """Fraction of required phrases present in the answer.

    This is the metric that catches the failure people actually care about: an
    answer that reads fluently but has dropped the one fact that mattered.
    """
    if not required:
        return 1.0
    haystack = normalise(answer)
    hits = sum(1 for phrase in required if normalise(phrase) in haystack)
    return hits / len(required)


def contains_none(answer: str, forbidden: Sequence[str]) -> float:
    """1.0 when no forbidden phrase appears. Used for safety and hallucination checks."""
    if not forbidden:
        return 1.0
    haystack = normalise(answer)
    return 0.0 if any(normalise(p) in haystack for p in forbidden) else 1.0


def context_recall(retrieved: Iterable[str], expected_sources: Sequence[str]) -> float:
    """Fraction of expected source documents that retrieval actually returned.

    Scored separately from answer quality on purpose. When a RAG system regresses,
    this tells you whether retrieval broke or generation did - which are different
    bugs owned by different parts of the pipeline.
    """
    if not expected_sources:
        return 1.0
    got = {normalise(r) for r in retrieved}
    hits = sum(1 for s in expected_sources if normalise(s) in got)
    return hits / len(expected_sources)


METRICS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
}
