"""Run a golden set against a RAG system and decide whether the build passes.

The gate compares against a committed baseline rather than an absolute threshold.
Absolute thresholds get set once, drift out of date, and are eventually raised to
whatever the current score is. A baseline comparison answers the question a merge
gate should ask: is this change worse than what is already on main?
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .dataset import Case
from .metrics import contains_all, contains_none, context_recall, exact_match, token_f1


@dataclass(frozen=True)
class Answer:
    """What the system under test returned for one case."""

    text: str
    sources: Sequence[str] = field(default_factory=tuple)


#: A RAG system under test: question in, Answer out.
RagCallable = Callable[[str], Answer]


@dataclass
class CaseResult:
    id: str
    scores: Dict[str, float]
    passed: bool
    tags: Sequence[str] = field(default_factory=tuple)

    @property
    def overall(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


@dataclass
class Report:
    results: List[CaseResult]

    @property
    def mean_scores(self) -> Dict[str, float]:
        if not self.results:
            return {}
        keys = self.results[0].scores.keys()
        return {
            k: sum(r.scores[k] for r in self.results) / len(self.results) for k in keys
        }

    @property
    def overall(self) -> float:
        return (
            sum(r.overall for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def failures(self) -> List[CaseResult]:
        return [r for r in self.results if not r.passed]

    def by_tag(self) -> Dict[str, float]:
        """Mean score per tag, so a regression can be traced to a slice of the set."""
        buckets: Dict[str, List[float]] = {}
        for r in self.results:
            for tag in r.tags:
                buckets.setdefault(tag, []).append(r.overall)
        return {t: sum(v) / len(v) for t, v in buckets.items()}

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": round(self.overall, 4),
            "mean_scores": {k: round(v, 4) for k, v in self.mean_scores.items()},
            "by_tag": {k: round(v, 4) for k, v in self.by_tag().items()},
            "cases": [asdict(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def score_case(case: Case, answer: Answer, *, pass_threshold: float = 0.6) -> CaseResult:
    """Score one case across every metric that applies to it.

    Metrics a case does not configure (no must_include, no expected_sources) return
    1.0 and are still reported, so the score vector has the same shape for every
    case and baselines stay comparable as the set grows.
    """
    scores = {
        "exact_match": exact_match(answer.text, case.expected),
        "token_f1": token_f1(answer.text, case.expected),
        "must_include": contains_all(answer.text, case.must_include),
        "must_not_include": contains_none(answer.text, case.must_not_include),
        "context_recall": context_recall(answer.sources, case.expected_sources),
    }

    # A forbidden phrase or a missing required fact fails the case outright,
    # however well it scores on overlap. These are correctness gates, not averages.
    hard_fail = scores["must_not_include"] < 1.0 or scores["must_include"] < 1.0
    passed = (not hard_fail) and (scores["token_f1"] >= pass_threshold)

    return CaseResult(id=case.id, scores=scores, passed=passed, tags=tuple(case.tags))


def run(
    cases: Sequence[Case],
    system: RagCallable,
    *,
    pass_threshold: float = 0.6,
) -> Report:
    return Report([score_case(c, system(c.question), pass_threshold=pass_threshold) for c in cases])


@dataclass
class GateDecision:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    deltas: Dict[str, float] = field(default_factory=dict)


def gate(
    report: Report,
    baseline: Optional[Dict[str, object]],
    *,
    tolerance: float = 0.02,
    allow_new_failures: bool = False,
) -> GateDecision:
    """Compare a run against the committed baseline.

    tolerance absorbs harmless noise; anything beyond it is treated as a regression.
    A newly failing case fails the gate regardless of the mean, because a mean can
    stay flat while a case that used to work silently breaks.
    """
    reasons: List[str] = []
    deltas: Dict[str, float] = {}

    if baseline is None:
        # First run: record, don't block. There is nothing to regress against yet.
        return GateDecision(ok=True, reasons=["no baseline yet - recording this run"])

    base_overall = float(baseline.get("overall", 0.0))
    delta = report.overall - base_overall
    deltas["overall"] = round(delta, 4)
    if delta < -tolerance:
        reasons.append(
            f"overall score fell {abs(delta):.3f} "
            f"({base_overall:.3f} -> {report.overall:.3f}, tolerance {tolerance})"
        )

    base_metrics = baseline.get("mean_scores", {}) or {}
    for name, value in report.mean_scores.items():
        if name in base_metrics:
            d = value - float(base_metrics[name])
            deltas[name] = round(d, 4)
            if d < -tolerance:
                reasons.append(f"{name} fell {abs(d):.3f}")

    if not allow_new_failures:
        was_passing = {
            c["id"] for c in baseline.get("cases", []) if c.get("passed")
        }
        now_failing = {r.id for r in report.failures}
        newly_broken = sorted(was_passing & now_failing)
        if newly_broken:
            reasons.append(
                f"{len(newly_broken)} case(s) that passed on the baseline now fail: "
                + ", ".join(newly_broken[:5])
                + ("..." if len(newly_broken) > 5 else "")
            )

    return GateDecision(ok=not reasons, reasons=reasons, deltas=deltas)


def load_baseline(path: Path) -> Optional[Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
