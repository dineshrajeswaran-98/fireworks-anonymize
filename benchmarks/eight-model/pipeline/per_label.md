# Frozen-set PII evaluation

All metrics use gold annotations. Invalid, failed, and missing outputs contribute empty predictions and missed gold entities.

| Model | Typed F1 | Precision | Recall | Leak rate | JSON valid | Usable output | p50 s | p95 s | USD / 1,000 docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | 74.77% | 80.32% | 69.93% | 17.23% | 99.00% | 99.00% | 1.044 | 2.156 | Unknown (missing usage) |
| deepseek-v4p1-flash-1017 | 80.50% | 81.96% | 79.09% | 5.36% | 100.00% | 100.00% | 1.356 | 2.331 | $0.2293 |
| glm-5p3-flash | 77.80% | 77.23% | 78.39% | 3.09% | 100.00% | 100.00% | 1.275 | 4.568 | $0.1169 |
| gpt-6-luna | 84.30% | 83.05% | 85.58% | 4.12% | 100.00% | 100.00% | 3.004 | 5.764 | $0.1422 |
| gpt-6.1-terra | 83.75% | 84.27% | 83.23% | 7.02% | 99.00% | 99.00% | 2.605 | 4.712 | Unknown (missing usage) |
| gpt-6.1-sol | 87.12% | 85.19% | 89.15% | 2.79% | 100.00% | 100.00% | 2.380 | 5.474 | $2.0553 |
| minimax-m3 | 76.03% | 77.44% | 74.68% | 11.44% | 97.20% | 97.20% | 1.862 | 4.883 | $0.7213 |
| qwen3p8-max | 78.94% | 78.70% | 79.18% | 5.48% | 98.80% | 98.80% | 1.575 | 3.763 | Unknown (missing usage) |

Value normalization: strip Python ASCII string.punctuation and whitespace at both edges; case-sensitive; unique pairs per row.

Leak coverage ignores labels: a gold value is protected only when contained in a predicted value that actually occurs in the source. Full-value coverage is required. Hallucination counts normalized predicted values absent from the source. Both rates use unique values per row.

Latency includes failed attempts with recorded elapsed times. Costs use reported tokens and dated standard prices; unknown usage is never treated as zero. Known-usage estimates are in metrics.json, including billed invalid completions.

Frozen gold contains canonical labels only; original numbered labels cannot be recovered.

## Per-label scores

| Label | Gold support | gpt-oss-120b F1 | deepseek-v4p1-flash-1017 F1 | glm-5p3-flash F1 | gpt-6-luna F1 | gpt-6.1-terra F1 | gpt-6.1-sol F1 | minimax-m3 F1 | qwen3p8-max F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOD | 157 | 6.17% | 73.96% | 5.99% | 78.39% | 85.22% | 86.96% | 20.88% | 14.04% |
| BUILDING | 153 | 52.21% | 74.51% | 55.11% | 80.27% | 59.39% | 80.45% | 52.81% | 57.02% |
| CARDISSUER | 1 | 100.00% | 40.00% | 14.29% | 100.00% | 100.00% | 100.00% | 100.00% | 66.67% |
| CITY | 152 | 90.79% | 83.45% | 89.33% | 91.14% | 93.04% | 94.64% | 87.07% | 93.29% |
| COUNTRY | 107 | 82.85% | 88.70% | 86.29% | 86.53% | 85.94% | 85.60% | 82.91% | 87.24% |
| DATE | 104 | 52.49% | 70.21% | 51.95% | 72.99% | 77.47% | 80.65% | 55.80% | 52.38% |
| DRIVERLICENSE | 147 | 66.38% | 69.57% | 71.07% | 70.04% | 70.29% | 89.29% | 70.25% | 69.96% |
| EMAIL | 171 | 95.29% | 97.70% | 95.91% | 97.98% | 98.27% | 96.17% | 92.77% | 95.58% |
| GEOCOORD | 19 | 97.44% | 77.27% | 95.00% | 97.44% | 97.44% | 97.44% | 92.31% | 90.48% |
| GIVENNAME | 165 | 64.47% | 64.77% | 75.64% | 79.47% | 75.48% | 80.65% | 77.44% | 78.36% |
| IDCARD | 155 | 63.16% | 66.67% | 62.77% | 68.92% | 67.86% | 74.83% | 67.95% | 68.35% |
| IP | 151 | 94.50% | 94.34% | 95.24% | 96.79% | 97.09% | 97.11% | 91.67% | 95.24% |
| LASTNAME | 181 | 66.67% | 75.61% | 79.37% | 77.72% | 78.47% | 82.63% | 77.69% | 80.00% |
| PASS | 110 | 62.43% | 84.93% | 85.07% | 90.58% | 91.59% | 92.31% | 81.45% | 70.97% |
| PASSPORT | 138 | 71.43% | 75.00% | 70.45% | 74.26% | 76.49% | 76.34% | 66.91% | 70.04% |
| POSTCODE | 144 | 85.71% | 82.61% | 86.99% | 90.20% | 90.55% | 89.61% | 82.93% | 89.84% |
| SECADDRESS | 58 | 60.24% | 75.00% | 71.01% | 76.77% | 94.64% | 95.65% | 80.00% | 65.28% |
| SEX | 100 | 84.75% | 83.24% | 93.26% | 94.36% | 94.90% | 96.48% | 91.49% | 92.06% |
| SOCIALNUMBER | 140 | 77.31% | 78.46% | 72.84% | 76.70% | 81.85% | 77.58% | 78.85% | 78.79% |
| STATE | 127 | 91.80% | 87.66% | 89.83% | 97.64% | 96.41% | 96.47% | 89.45% | 95.20% |
| STREET | 148 | 61.39% | 75.00% | 70.43% | 83.60% | 64.74% | 85.16% | 53.47% | 82.14% |
| TEL | 122 | 86.14% | 89.68% | 88.30% | 83.69% | 87.73% | 85.20% | 84.38% | 84.67% |
| TIME | 263 | 89.94% | 90.23% | 89.53% | 92.31% | 91.01% | 91.98% | 87.70% | 89.94% |
| TITLE | 106 | 74.44% | 75.00% | 84.85% | 82.83% | 78.00% | 84.73% | 81.55% | 79.35% |
| USERNAME | 190 | 81.40% | 85.56% | 83.26% | 83.87% | 84.50% | 85.71% | 72.73% | 83.12% |

## Reliability and annotation notes

- gpt-oss-120b: 500/500 recorded rows; hallucination rate 0.28%; zero-PII over-redaction 11.76%; exact-row match 30.40%; 5 rows with unknown cost.
  Errors: {"Fireworks network request failed or timed out.": 5}
- deepseek-v4p1-flash-1017: 500/500 recorded rows; hallucination rate 0.28%; zero-PII over-redaction 13.73%; exact-row match 40.20%; 0 rows with unknown cost.
  Errors: {}
- glm-5p3-flash: 500/500 recorded rows; hallucination rate 0.36%; zero-PII over-redaction 11.76%; exact-row match 35.80%; 0 rows with unknown cost.
  Errors: {}
- gpt-6-luna: 500/500 recorded rows; hallucination rate 0.09%; zero-PII over-redaction 17.65%; exact-row match 42.80%; 0 rows with unknown cost.
  Errors: {}
- gpt-6.1-terra: 500/500 recorded rows; hallucination rate 0.00%; zero-PII over-redaction 17.65%; exact-row match 42.60%; 5 rows with unknown cost.
  Errors: {"OpenAI network request failed or timed out.": 5}
- gpt-6.1-sol: 500/500 recorded rows; hallucination rate 0.06%; zero-PII over-redaction 17.65%; exact-row match 46.40%; 0 rows with unknown cost.
  Errors: {}
- minimax-m3: 500/500 recorded rows; hallucination rate 0.63%; zero-PII over-redaction 19.61%; exact-row match 33.60%; 0 rows with unknown cost.
  Errors: {"Completion did not finish normally: length": 14}
- qwen3p8-max: 500/500 recorded rows; hallucination rate 0.21%; zero-PII over-redaction 7.84%; exact-row match 35.60%; 6 rows with unknown cost.
  Errors: {"Fireworks network request failed or timed out.": 6}

The first ten frozen rows were previewed earlier. Report the full 500-row score with that disclosure and report the remaining 490 separately; these are not 500 previously unseen cases. CARDISSUER has support 1, below the intended minimum of 10. Gold is synthetic and may have annotation errors; scores measure agreement with this frozen gold.
