"""Command line entry point.

    rageval run  --golden golden.jsonl --system examples.toy_rag:answer
    rageval gate --golden golden.jsonl --system examples.toy_rag:answer \\
                 --baseline baseline.json [--update-baseline]

`gate` exits non-zero on regression, which is the entire integration story with CI:
no plugin, no service, just an exit code.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .dataset import DatasetError, load_golden_set
from .runner import RagCallable, gate, load_baseline, run


def resolve_system(spec: str) -> RagCallable:
    """Import 'package.module:function' and return the callable."""
    if ":" not in spec:
        raise SystemExit(
            f"--system must look like 'module.path:function', got {spec!r}"
        )
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import {module_name!r}: {exc}") from exc
    func = getattr(module, func_name, None)
    if func is None:
        raise SystemExit(f"{module_name!r} has no attribute {func_name!r}")
    return func


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="rageval")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--golden", required=True, type=Path, help="golden set (.jsonl)")
    common.add_argument("--system", required=True, help="module.path:function to test")
    common.add_argument("--threshold", type=float, default=0.6, help="per-case token_f1 pass threshold")
    common.add_argument("--report", type=Path, default=None, help="write full JSON report here")

    sub.add_parser("run", parents=[common], help="run the golden set and print scores")

    g = sub.add_parser("gate", parents=[common], help="fail (exit 1) on regression vs baseline")
    g.add_argument("--baseline", required=True, type=Path)
    g.add_argument("--tolerance", type=float, default=0.02)
    g.add_argument("--update-baseline", action="store_true",
                   help="write this run as the new baseline (only when the gate passes)")

    args = parser.parse_args(argv)

    try:
        cases = load_golden_set(args.golden)
    except DatasetError as exc:
        print(f"golden set error: {exc}", file=sys.stderr)
        return 2

    system = resolve_system(args.system)
    report = run(cases, system, pass_threshold=args.threshold)

    print(f"cases      : {len(report.results)}")
    print(f"overall    : {report.overall:.4f}")
    for name, value in report.mean_scores.items():
        print(f"  {name:<18} {value:.4f}")
    if report.by_tag():
        print("by tag:")
        for tag, value in sorted(report.by_tag().items()):
            print(f"  {tag:<18} {value:.4f}")
    if report.failures:
        print(f"failing cases ({len(report.failures)}):")
        for r in report.failures:
            worst = min(r.scores, key=r.scores.get)  # type: ignore[arg-type]
            print(f"  {r.id:<24} worst metric: {worst}={r.scores[worst]:.3f}")

    if args.report:
        report.save(args.report)
        print(f"report written to {args.report}")

    if args.command == "run":
        return 0

    baseline = load_baseline(args.baseline)
    decision = gate(report, baseline, tolerance=args.tolerance)

    if decision.deltas:
        print("deltas vs baseline:")
        for name, d in decision.deltas.items():
            print(f"  {name:<18} {d:+.4f}")

    if not decision.ok:
        print("\nGATE FAILED:", file=sys.stderr)
        for reason in decision.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    for reason in decision.reasons:
        print(reason)

    if args.update_baseline or baseline is None:
        report.save(args.baseline)
        print(f"baseline written to {args.baseline}")

    print("gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
