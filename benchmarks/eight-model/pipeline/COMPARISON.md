# Secret Keepers: matched 500-row model comparison

Each ranked model was attempted on the same 500 frozen rows. Canonical typed precision, recall and micro F1 use the same scorer. All failed requests remain in the denominators. These are untuned model baselines.

| Model | Precision | Recall | Canonical F1 | Leak rate | p50 s | p95 s | USD / 1,000 docs | Usable outputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | 80.32% | 69.93% | 74.77% | 17.23% | 1.044 | 2.156 | Unknown usage | 495/500 |
| deepseek-v4p1-flash-1017 | 81.96% | 79.09% | 80.50% | 5.36% | 1.356 | 2.331 | $0.2293 | 500/500 |
| glm-5p3-flash | 77.23% | 78.39% | 77.80% | 3.09% | 1.275 | 4.568 | $0.1169 | 500/500 |
| gpt-6-luna | 83.05% | 85.58% | 84.30% | 4.12% | 3.004 | 5.764 | $0.1422 | 500/500 |
| gpt-6.1-terra | 84.27% | 83.23% | 83.75% | 7.02% | 2.605 | 4.712 | Unknown usage | 495/500 |
| gpt-6.1-sol | 85.19% | 89.15% | 87.12% | 2.79% | 2.380 | 5.474 | $2.0553 | 500/500 |
| minimax-m3 | 77.44% | 74.68% | 76.03% | 11.44% | 1.862 | 4.883 | $0.7213 | 486/500 |
| qwen3p8-max | 78.70% | 79.18% | 78.94% | 5.48% | 1.575 | 3.763 | Unknown usage | 494/500 |

## Reliability and cost completeness

| Model | Hallucination | Empty-gold over-redaction | Exact-row match | Known-usage USD / 500 | Unknown-cost rows |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | 0.28% | 11.76% | 30.40% | $0.078957 | 5 |
| deepseek-v4p1-flash-1017 | 0.28% | 13.73% | 40.20% | $0.114673 | 0 |
| glm-5p3-flash | 0.36% | 11.76% | 35.80% | $0.058474 | 0 |
| gpt-6-luna | 0.09% | 17.65% | 42.80% | $0.071108 | 0 |
| gpt-6.1-terra | 0.00% | 17.65% | 42.60% | $0.514850 | 5 |
| gpt-6.1-sol | 0.06% | 17.65% | 46.40% | $1.027670 | 0 |
| minimax-m3 | 0.63% | 19.61% | 33.60% | $0.360663 | 0 |
| qwen3p8-max | 0.21% | 7.84% | 35.60% | $0.665642 | 6 |

Unknown-cost rows have no reliable token accounting or require unpriced cache-write details. Known-usage costs are lower bounds when any costs are unknown. Failed requests are never assumed free. Estimates use dated Standard rates, not billing invoices.

Request failures by model:
- gpt-oss-120b: {"Fireworks network request failed or timed out.": 5}
- deepseek-v4p1-flash-1017: {}
- glm-5p3-flash: {}
- gpt-6-luna: {}
- gpt-6.1-terra: {"OpenAI network request failed or timed out.": 5}
- gpt-6.1-sol: {}
- minimax-m3: {"Completion did not finish normally: length": 14}
- qwen3p8-max: {"Fireworks network request failed or timed out.": 6}

Length failures mean the completion exceeded the fixed output budget. They are retained as empty predictions, with all gold counted as missed. Their reported token usage still contributes to cost.

## Uncertainty

95% row-bootstrap intervals, 1,000 replicates, seed 7. These are descriptive intervals, not a multiple-comparison-corrected significance test.

| Model | Canonical F1 interval | Leak-rate interval |
| --- | --- | --- |
| gpt-oss-120b | 72.15% to 76.88% | 14.74% to 20.11% |
| deepseek-v4p1-flash-1017 | 77.91% to 82.71% | 4.01% to 6.78% |
| glm-5p3-flash | 75.39% to 79.99% | 2.24% to 4.01% |
| gpt-6-luna | 82.07% to 86.35% | 2.93% to 5.40% |
| gpt-6.1-terra | 81.73% to 85.71% | 5.40% to 8.87% |
| gpt-6.1-sol | 85.33% to 88.69% | 2.09% to 3.63% |
| minimax-m3 | 73.07% to 78.59% | 7.24% to 15.91% |
| qwen3p8-max | 76.59% to 81.07% | 4.08% to 7.15% |

Paired difference intervals versus the prespecified Sol reference are in confidence_intervals.json. Small ranking differences should not be treated as certain wins.

## Original preview excluded

The following view excludes the ten fixed IDs from the original manual preview. The remaining 490 are not a newly unused test set: they were already scored for earlier open-model runs. No model prompts were tuned from these comparison results.

| Model | Precision | Recall | Canonical F1 | Leak rate |
| --- | --- | --- | --- | --- |
| gpt-oss-120b | 80.15% | 69.84% | 74.64% | 17.21% |
| deepseek-v4p1-flash-1017 | 81.87% | 78.97% | 80.40% | 5.44% |
| glm-5p3-flash | 77.10% | 78.29% | 77.69% | 3.12% |
| gpt-6-luna | 82.98% | 85.54% | 84.24% | 4.17% |
| gpt-6.1-terra | 84.25% | 83.16% | 83.71% | 7.14% |
| gpt-6.1-sol | 85.04% | 89.05% | 87.00% | 2.81% |
| minimax-m3 | 77.36% | 74.50% | 75.90% | 11.65% |
| qwen3p8-max | 78.63% | 79.06% | 78.84% | 5.56% |

## Definitions and limitations

- Canonical F1 matches exact values after fixed edge punctuation/whitespace trimming and merged labels. Matching is case-sensitive, unique per value/label pair per row. Precision = TP/(TP+FP); recall = TP/(TP+FN).
- Leak rate ignores labels and counts gold values not covered by a grounded predicted value. It is value-based, not a complete span-level privacy audit. Invalid, failed and missing outputs contribute empty predictions.
- Every ranked model uses concurrency five, a 30-second timeout and a 2,048-token output cap. OpenAI uses low reasoning and omits temperature; Fireworks uses temperature 0 and the model-specific reasoning settings recorded in each source manifest.
- Runs took place at different wall-clock times. Provider load, schema warmup, tokenizers and cache behavior differ; latency is descriptive and not proof of an infrastructure-only speedup.
- The gold is synthetic and may contain annotation errors. CARDISSUER has only one instance. Per-label support and scores are in per_label.md. Strict numbered raw-label F1 is unavailable because those labels were merged in the frozen gold.
- Nemotron was explicitly skipped after timeouts. Its earlier five-row training smoke returned 4/5 schema-valid outputs; a later ten-row batch was interrupted after four saved records. It has no matched 500-row score and is not ranked here.
- Astra and Claude have not been tested and are not included. No fine-tuned model has been evaluated yet.

## Source runs

- 20261102T224047677720Z: accounts/fireworks/models/gpt-oss-120b, accounts/fireworks/models/deepseek-v4p1-flash-1017, accounts/fireworks/models/glm-5p3-flash; concurrency 5; retries 0.
- 20261102T233804779024Z: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol; concurrency 5; retries 0.
- 20261102T234129761516Z: accounts/fireworks/models/minimax-m3, accounts/fireworks/models/qwen3p8-max; concurrency 5; retries 0.

Artifacts: metrics.json, metrics_preview_excluded.json, predictions.jsonl, row_metrics.jsonl, confidence_intervals.json, per_label.md, and manifest.json. The aggregation made no API calls.

Published results include the [per-label scores](per_label.md) and [paired confidence intervals](confidence_intervals.json). The original prediction files and row diagnostics are retained in the local benchmark workspace.
