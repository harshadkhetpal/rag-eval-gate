"""Golden-set loading and validation.

A golden set is a JSONL file, one case per line. JSONL rather than JSON because
golden sets grow by appending, and appending to a JSON array means rewriting the
whole file - which produces noisy diffs and merge conflicts on the one artefact
a team most needs to review line by line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class DatasetError(ValueError):
    """Raised when a golden set is malformed. Always names the offending line."""


@dataclass(frozen=True)
class Case:
    """One evaluation case.

    id                 stable identifier - keep it stable, results are tracked by it
    question           the input to the RAG system
    expected           reference answer, used by exact_match / token_f1
    must_include       phrases the answer must contain (facts that matter)
    must_not_include   phrases the answer must not contain (safety, known hallucinations)
    expected_sources   document ids retrieval should return
    tags               free-form labels for slicing results
    """

    id: str
    question: str
    expected: str = ""
    must_include: Sequence[str] = field(default_factory=tuple)
    must_not_include: Sequence[str] = field(default_factory=tuple)
    expected_sources: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)

    @staticmethod
    def from_dict(raw: Dict[str, Any], *, line_no: int) -> "Case":
        missing = [k for k in ("id", "question") if not raw.get(k)]
        if missing:
            raise DatasetError(
                f"line {line_no}: case is missing required field(s): {', '.join(missing)}"
            )
        if not isinstance(raw["id"], str):
            raise DatasetError(f"line {line_no}: 'id' must be a string")

        def seq(key: str) -> tuple:
            value = raw.get(key, ())
            if isinstance(value, str):
                # A bare string is almost always a typo for a one-element list.
                raise DatasetError(
                    f"line {line_no}: '{key}' must be a list, got a string. "
                    f'Did you mean ["{value}"]?'
                )
            return tuple(value)

        return Case(
            id=raw["id"],
            question=raw["question"],
            expected=raw.get("expected", ""),
            must_include=seq("must_include"),
            must_not_include=seq("must_not_include"),
            expected_sources=seq("expected_sources"),
            tags=seq("tags"),
        )


def load_golden_set(path: Path) -> List[Case]:
    """Read and validate a JSONL golden set.

    Validation is strict and fails on the first bad line with its number. A gate
    that silently skips malformed cases quietly shrinks its own coverage, which is
    the one failure mode a quality gate must not have.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"golden set not found: {path}")

    cases: List[Case] = []
    seen: Dict[str, int] = {}

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_no}: invalid JSON - {exc.msg}") from exc

        case = Case.from_dict(raw, line_no=line_no)
        if case.id in seen:
            raise DatasetError(
                f"line {line_no}: duplicate case id {case.id!r} "
                f"(first seen on line {seen[case.id]})"
            )
        seen[case.id] = line_no
        cases.append(case)

    if not cases:
        raise DatasetError(f"golden set {path} contains no cases")
    return cases
