from __future__ import annotations

import pytest

from rageval.metrics import (
    contains_all,
    contains_none,
    context_recall,
    exact_match,
    normalise,
    token_f1,
)


class TestNormalise:
    def test_case_punctuation_and_whitespace(self):
        assert normalise("  Hello,   WORLD!! ") == "hello world"

    def test_accents_are_stripped(self):
        assert normalise("café") == "cafe"


class TestExactMatch:
    def test_match_despite_formatting(self):
        assert exact_match("The answer is 42.", "the ANSWER is 42") == 1.0

    def test_different_content(self):
        assert exact_match("yes", "no") == 0.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("refunds take five days", "refunds take five days") == 1.0

    def test_partial_overlap_is_between_zero_and_one(self):
        score = token_f1("refunds take five days", "refunds take ten days")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert token_f1("bananas are yellow", "refunds take five days") == 0.0

    def test_stopwords_do_not_inflate_the_score(self):
        # Shares only stopwords with the reference - must not score above zero.
        assert token_f1("the of and to", "refunds take five days") == 0.0

    def test_repeated_words_only_match_once(self):
        # 'days days days' must not triple-count against a single 'days'.
        inflated = token_f1("days days days", "five days")
        single = token_f1("days", "five days")
        assert inflated <= single + 1e-9

    def test_both_empty_is_perfect(self):
        assert token_f1("", "") == 1.0

    def test_one_empty_is_zero(self):
        assert token_f1("something", "") == 0.0


class TestContains:
    def test_all_required_present(self):
        assert contains_all("Refunds take 5 business days.", ["5 business days"]) == 1.0

    def test_missing_fact_scores_fractionally(self):
        assert contains_all("Refunds happen fast.", ["5 business days", "refunds"]) == 0.5

    def test_no_requirements_is_perfect(self):
        assert contains_all("anything", []) == 1.0

    def test_forbidden_phrase_fails(self):
        assert contains_none("The password is hunter2", ["password is"]) == 0.0

    def test_clean_answer_passes(self):
        assert contains_none("I cannot share that.", ["password is"]) == 1.0


class TestContextRecall:
    def test_all_sources_found(self):
        assert context_recall(["a.md", "b.md"], ["a.md"]) == 1.0

    def test_missing_source(self):
        assert context_recall(["a.md"], ["a.md", "b.md"]) == 0.5

    def test_no_expectation_is_perfect(self):
        assert context_recall([], []) == 1.0
