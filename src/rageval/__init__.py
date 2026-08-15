"""rag-eval-gate: golden-set evaluation for RAG systems, as a CI gate.

A change that degrades answer quality should fail the build the same way a failing
unit test does. Deterministic offline metrics, a committed baseline, and an exit code.
"""
from .dataset import Case, DatasetError, load_golden_set
from .metrics import contains_all, contains_none, context_recall, exact_match, token_f1
from .runner import Answer, GateDecision, Report, gate, load_baseline, run

__version__ = "0.1.0"
__all__ = [
    "Case", "DatasetError", "load_golden_set",
    "contains_all", "contains_none", "context_recall", "exact_match", "token_f1",
    "Answer", "GateDecision", "Report", "gate", "load_baseline", "run",
]
