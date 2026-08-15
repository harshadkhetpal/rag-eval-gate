# rag-eval-gate

**A change that degrades your RAG system's answers should fail the build the same way
a failing unit test does.**

rag-eval-gate runs a committed golden set of question/answer cases against your RAG
pipeline, scores the answers with deterministic offline metrics, compares against the
baseline on `main`, and exits non-zero on regression. That exit code is the whole CI
integration — no plugin, no service, no API key.

```bash
git clone https://github.com/harshadkhetpal/rag-eval-gate
cd rag-eval-gate
PYTHONPATH=src:examples python3 -m rageval.cli gate \
    --golden examples/golden.jsonl --system toy_rag:answer --baseline baseline.json
```

## What it catches

The demo includes a simulated bad deploy (`toy_rag:broken_answer`) where refund
answers silently lose the *"5 business days"* fact — the answer still reads fluently,
still cites the right source, still scores 0.88 on token overlap. The gate fails it
anyway and names the case:

```
GATE FAILED:
  - overall score fell 0.123 (1.000 -> 0.877, tolerance 0.02)
  - must_include fell 0.250
  - 1 case(s) that passed on the baseline now fail: refund-timeframe
```

That is the failure mode that matters in production RAG: not nonsense output, but a
fluent answer with the load-bearing fact missing.

## Golden sets

One JSON case per line (`.jsonl` — appending never rewrites the file, so diffs stay
reviewable):

```jsonl
{"id": "refund-timeframe",
 "question": "How long do refunds take?",
 "expected": "Refunds are processed within 5 business days to the original payment method.",
 "must_include": ["5 business days"],
 "must_not_include": [],
 "expected_sources": ["policies/refunds.md"],
 "tags": ["billing"]}
```

| Field | Purpose |
|---|---|
| `expected` | Reference answer for `exact_match` and `token_f1` |
| `must_include` | Facts the answer must contain. Missing one **fails the case outright**, regardless of overlap score |
| `must_not_include` | Phrases that must never appear (safety, known hallucinations). Also a hard fail |
| `expected_sources` | What retrieval should have returned — scored separately, so you can tell *retrieval broke* from *generation broke* |
| `tags` | Slice scores by area (`billing`, `legal`, …) to localise a regression |

Loading is strict: duplicate ids, missing fields and type mistakes fail with the line
number. A gate that silently skips malformed cases quietly shrinks its own coverage.

## Wiring in your system

Anything callable that takes a question and returns an `Answer`:

```python
# my_rag.py
from rageval.runner import Answer

def answer(question: str) -> Answer:
    result = my_pipeline.query(question)          # your existing RAG code
    return Answer(text=result.text, sources=result.document_ids)
```

```bash
rageval gate --golden golden.jsonl --system my_rag:answer --baseline baseline.json
```

## CI

```yaml
- name: RAG quality gate
  run: |
    pip install -e .
    rageval gate --golden eval/golden.jsonl \
                 --system my_rag:answer \
                 --baseline eval/baseline.json
```

The baseline is a committed JSON file. When a change legitimately improves quality,
re-run with `--update-baseline` and commit the new file — the improvement is then
protected the same way the old score was.

## Design decisions worth arguing with

**No LLM-as-judge in the gate.** A judge model makes the gate non-deterministic: the
same code can pass on retry, and a build that flakes teaches people to re-run until
green — at which point the gate protects nothing. LLM judges are useful for offline
analysis; a merge gate needs the same input to produce the same verdict every time.

**Baseline comparison, not absolute thresholds.** "token_f1 must exceed 0.7" gets set
once and drifts out of date; teams eventually raise it to whatever today's score is.
The question a merge gate should answer is *"is this worse than main?"* — so that is
the comparison it makes, with a small tolerance for noise.

**A newly failing case fails the gate even if the mean is flat.** Means hide
casualties: one case can break while ten others wobble upward. The gate names the
broken case instead of letting the average absorb it.

**`must_include` is a hard fail, not a weighted term.** An answer that drops the one
fact that mattered should not be rescuable by fluent phrasing. Correctness conditions
gate; similarity metrics score.

**Retrieval is scored separately from generation.** `context_recall` failing tells you
the retriever broke. `token_f1` failing with healthy recall tells you generation
broke. Different bugs, different owners, one report.

## Testing

```bash
PYTHONPATH=src:examples python3 -m pytest tests -q      # 29 tests, offline, <1s
```

The regression scenario itself is a test: the suite asserts that the broken deploy
fails the gate *and* that the gate names the right case for the right reason.

## Status

v0.1.0. Metrics, dataset validation, runner, baseline gate and CLI are complete and
tested. The obvious extensions — embedding-based similarity as an *opt-in* metric,
per-tag tolerances — are documented issues rather than half-built code.

## Licence

MIT
