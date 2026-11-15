# Eight-model PII evaluation

The [500-row scorecard](pipeline/COMPARISON.md) compares precision, recall, F1, leak rate, latency and estimated token cost for five Fireworks models and three models called directly through OpenAI. See the [per-label scores](pipeline/per_label.md), [benchmark plan](pipeline/BENCHMARK_PLAN.md), and [scoring definitions](pipeline/EVALUATION.md).

This evaluation measures untuned model baselines. The [team submission](../../README.md) reports separate prompt harness and LoRA experiments. Both use the same frozen synthetic dataset in the repository's `data/` directory.

## Reproduce

Run the following commands from this directory with Python 3.10 or later. Copy `.env.example` to `.env` and fill in the provider keys. Each full inference command makes 500 paid requests per model. Add `--dry-run` to check the configuration without calling the API.

```bash
python3 scripts/run_models.py --provider fireworks \
  --input ../../data/test_set.jsonl --labels ../../data/labels.json \
  --limit 500 --concurrency 5 \
  --model accounts/fireworks/models/gpt-oss-120b \
  --model accounts/fireworks/models/deepseek-v4p1-flash-1017 \
  --model accounts/fireworks/models/glm-5p3-flash \
  --model accounts/fireworks/models/minimax-m3 \
  --model accounts/fireworks/models/qwen3p8-max

python3 scripts/run_models.py --provider openai \
  --input ../../data/test_set.jsonl --labels ../../data/labels.json \
  --limit 500 --concurrency 5 \
  --model gpt-6-luna --model gpt-6.1-terra --model gpt-6.1-sol

python3 scripts/evaluate.py --run FIREWORKS_RUN_DIRECTORY \
  --prices pipeline/comparison_prices_2026-11-02.json
python3 scripts/compare_runs.py --runs FIREWORKS_RUN_DIRECTORY OPENAI_RUN_DIRECTORY
```

Replace the directory placeholders with the paths printed by the runner. Scoring runs offline. The pricebook records the rates used in the November 2, 2026 benchmark. New results can vary with provider traffic and token usage. Raw API responses and credentials stay local.

Run the offline test suite with `python3 -m pytest tests -q`.
