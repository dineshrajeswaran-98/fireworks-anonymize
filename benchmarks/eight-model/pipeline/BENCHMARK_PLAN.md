# Eight-model comparison plan

All ranked models use `data/test_set.jsonl` unchanged, with the 25 canonical
labels in `data/labels.json`. The headline is **canonical typed micro F1**;
precision, recall and leak rate are always reported alongside it.

## Scorecard

| Metric | Direction | Interpretation |
| --- | --- | --- |
| Canonical typed precision | Higher | Share of predicted value/label pairs that match gold |
| Canonical typed recall | Higher | Share of gold value/label pairs recovered |
| Canonical typed micro F1 | Higher | Precision/recall balance after aggregating pair counts |
| Untyped value leak rate | Lower | Share of gold values not covered by a grounded predicted value |
| Per-label precision, recall, F1 and support | Contextual | Performance by PII type and amount of supporting evidence |
| Empty-gold over-redaction | Lower | Share of no-PII rows with at least one predicted entity |
| Hallucination rate | Lower | Share of predicted values absent from source |
| Exact-row match | Higher | Complete prediction set matches gold on the document |
| JSON validity | Higher | Response parses and matches the extraction schema |
| Usable output rate and failure counts | Higher / lower | Normal completed response, valid schema, no request error |
| p50 and p95 client latency | Lower | Median and nearest-rank p95 over recorded attempts, including failures |
| Estimated USD per 1,000 documents | Lower | Actual token usage priced at dated Standard rates |
| Bootstrap intervals for F1/leak rate | Narrower is more informative | Descriptive uncertainty over rows; paired differences versus Sol |

Failures contribute empty predictions; all their gold remains missed. No retries
replace failed rows. Missing usage means unknown cost, never free usage. Exact
definitions and normalization are in EVALUATION.md and evaluation_protocol.json.

## Models and execution

Reuse the completed 500-row run `20261102T224047677720Z` for:
- Fireworks GPT-OSS 120B
- Fireworks DeepSeek V4.1 Flash 1017
- Fireworks GLM 5.3 Flash

New complete runs:
- Direct OpenAI GPT-6 Luna, GPT-6.1 Terra and GPT-6.1 Sol
- Fireworks MiniMax M3 and Qwen 3.8 Max

All runs use concurrency five, a 30-second timeout and a 2,048-token cap.
Reasoning is low for GPT-OSS, GLM, MiniMax and direct OpenAI, and off for DeepSeek
and Qwen. OpenAI omits temperature; Fireworks uses zero. These differences and
different run times are disclosed. The extraction instruction and schema remain
fixed; model settings are not tuned against full-set predictions.

MiniMax and Qwen each ran on all 500 rows. This comparison uses untuned model baselines.

## Reporting

`scripts/compare_runs.py` combines explicit source directories only after every
model has one record for each of the 500 IDs. It rejects duplicate models,
incomplete source runs, changed data hashes, and changed source text. It produces
metrics, predictions, per-row diagnostics, per-label tables, confidence intervals,
and a public summary at COMPARISON.md without any API calls.

The first ten rows were manually previewed. The 490-row preview-excluded view is
also reported, but is not a fresh, unused test set across the entire project.
The gold is synthetic; label annotation errors remain possible. CARDISSUER has
support one. Strict raw-label F1 cannot be recovered from merged gold. Nemotron
remains historical only after the user asked to skip it; pilot-size scores are
not mixed into the 500-row ranking. Astra and Claude have not been run.

Catalog sources:
- https://app.fireworks.ai/models/fireworks/minimax-m3
- https://app.fireworks.ai/models/fireworks/qwen3p8-max

Price snapshots: comparison_prices_2026-11-02.json. The evaluator marks any
separately reported cache-write usage as unknown until duration/rate is established.
