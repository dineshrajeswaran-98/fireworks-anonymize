# Gold-based evaluator

`scripts/evaluate.py` reads saved predictions and frozen gold locally. It makes no
API requests. The protocol, normalization, preview IDs and failure policy are in
`evaluation_protocol.json`. The full run records hashes of the input, labels,
client, runner, evaluator and protocol before requests start.

```bash
.venv/bin/python scripts/evaluate.py --run pipeline/outputs/RUN_DIRECTORY
```

The evaluator verifies input and label hashes against the run manifest. It joins
by model and row ID, rejects duplicates, rejects extra rows, and rejects changed
source text. Missing outputs remain in denominators. A deliberately incomplete
run can be scored, but its missing row count and failures are explicit.

## Metrics

- **Typed precision, recall, micro F1:** unique `(normalized value, canonical
  label)` pairs per row; true positives, false positives and false negatives are
  summed across rows before ratios. Matching is case-sensitive. Normalize by
  stripping Python ASCII `string.punctuation` and whitespace from both edges;
  internal punctuation and spacing stay unchanged. Undefined ratios are null,
  not an invented perfect score.
- **Leak rate:** unique normalized gold values per row not contained in any
  normalized predicted value that occurs in source text, ignoring labels.
  Partial values do not cover a whole gold value. Hallucinated strings cannot
  hide gold. This follows the proposed value-based metric; with no gold offsets
  in the frozen file, it is not a span-level or complete document privacy audit.
- **Per-label precision, recall, F1 and support:** same matching and failure
  policy as the headline. CARDISSUER has only one gold instance, so its score
  provides weak evidence. Report this shortfall instead of modifying frozen gold.
- **Hallucination rate:** unique normalized predicted values absent from source,
  divided by unique normalized predicted values. Null when no values were emitted.
- **JSON validity:** reparses `raw_output` and independently checks the expected
  object/list/entity shape and label vocabulary. A parseable truncated completion
  can be schema-valid but is still unusable and excluded as a prediction.
- **Usable output rate:** valid schema, normal `stop` finish, and no request error.
- **Exact row match:** the complete predicted pair set matches gold, and the
  completion is usable. Failed zero-PII rows do not count as correct.
- **Zero-PII over-redaction:** share of empty-gold rows with nonempty predictions.
  Read alongside usable output rate; failures are not evidence of good redaction.
- **Latency:** client elapsed time including parsing and failed attempts; median
  p50 and nearest-rank p95. Timing starts when a worker begins the request,
  excluding time waiting in the client executor queue. Missing outputs without timing are explicitly counted.
- **Cost:** provider-reported prompt and completion tokens, with cached prompt
  discounts and dated Standard prices from `prices_2026-11-02.json`. Completion
  tokens already include reasoning tokens. Invalid completions still cost money.
  Missing usage yields unknown total cost and unknown cost per 1,000 documents;
  the sum of known usage estimates is also reported, not substituted for total.

Invalid, failed, truncated and missing completions contribute empty predictions:
all their gold pairs become false negatives, and all their gold values leak.
They are never silently dropped or automatically retried.

## Outputs

- `metrics.json`: complete model metrics, supports, errors, cost completeness,
  pricebook and run manifest.
- `row_metrics.jsonl`: row-level diagnostics, missed and extra entities, leaked
  values, and validity/error status for debugging.
- `evaluation.md`: human-readable comparison and per-label F1 table.
- `metrics_unpreviewed.json`: the same metrics restricted to the 490 rows not
  included in the earlier ten-row preview, when evaluating the complete set.

The frozen gold already merged numbered name labels and discarded original raw
labels. Strict raw-label F1 cannot be computed from this file; it is marked
unavailable instead of being mislabeled as canonical F1. The gold is synthetic,
and matching its annotations is not proof that every prediction is semantically
right or wrong on real business text. No gold is sent to models.

The full run uses the same concurrency five and 30-second timeout per model.
Reasoning differs by model: GPT-OSS/GLM low, DeepSeek off. The ten previewed test
rows and the change from the earlier sequential smoke tests must be disclosed.
The prompts and model settings are not tuned against the 500-row results.
