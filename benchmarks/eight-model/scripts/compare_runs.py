"""Combine completed matched 500-row runs and score them locally; no API calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from evaluate import DATA_ROOT, ROOT, evaluate, write_markdown


def interval(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]


def bootstrap(diagnostics, models, ids, reps=1000, seed=7, reference="gpt-6.1-sol"):
    """Paired row bootstrap: all models use the same row draw per replicate."""
    indexed = {(r["model"], r["id"]): r for r in diagnostics}
    vectors = {m: [tuple(indexed[m, row_id][k] for k in ["tp", "fp", "fn", "leaked_values", "gold_values"])
                   for row_id in ids] for m in models}
    samples = {m: {"f1": [], "leak_rate": []} for m in models}
    rng = random.Random(seed)
    for _ in range(reps):
        draw = [rng.randrange(len(ids)) for _ in ids]
        for model in models:
            tp = fp = fn = leaked = gold = 0
            values = vectors[model]
            for index in draw:
                a, b, c, d, e = values[index]
                tp += a; fp += b; fn += c; leaked += d; gold += e
            samples[model]["f1"].append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
            samples[model]["leak_rate"].append(leaked / gold if gold else None)
    output = {"replicates": reps, "seed": seed, "reference": reference,
              "note": "95% descriptive row-bootstrap percentile intervals; paired differences use identical draws; no multiple-comparison correction.", "models": {}}
    for model in models:
        item = {key + "_95ci": interval(values) for key, values in samples[model].items()}
        if reference in samples:
            for key in ["f1", "leak_rate"]:
                differences = [a - b for a, b in zip(samples[model][key], samples[reference][key])
                               if a is not None and b is not None]
                item[key + "_difference_vs_reference_95ci"] = interval(differences)
        output["models"][model] = item
    return output


def load_matched(runs, gold_path, labels_path, expected_rows=500):
    gold_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    gold = [json.loads(x) for x in gold_path.read_text().splitlines() if x.strip()]
    ids = [str(r["id"]) for r in gold]
    if len(ids) != expected_rows or len(set(ids)) != expected_rows:
        raise ValueError("Gold row count/uniqueness does not match the comparison")
    records, models, source_runs = [], [], []
    for directory in runs:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["input_sha256"] != gold_hash or manifest["labels_sha256"] != labels_hash:
            raise ValueError("Source run used different gold/labels")
        if len(manifest["ids"]) != expected_rows or set(manifest["ids"]) != set(ids):
            raise ValueError("Source run is not the matched full dataset")
        if set(manifest["models"]) & set(models):
            raise ValueError("Duplicate model across source runs; select one run per model explicitly")
        rows = [json.loads(x) for x in (directory / "predictions.jsonl").read_text().splitlines() if x.strip()]
        expected = {(m, row_id) for m in manifest["models"] for row_id in ids}
        actual = [(r["model"], str(r["id"])) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Source run is incomplete or contains duplicate/extra predictions")
        models.extend(manifest["models"])
        records.extend(rows)
        source_runs.append({"path": str(directory.resolve()), "manifest": manifest,
                            "predictions_sha256": hashlib.sha256((directory / "predictions.jsonl").read_bytes()).hexdigest()})
    combined = {"rows": len(ids), "ids": ids, "models": models, "input_sha256": gold_hash,
                "labels_sha256": labels_hash, "source_runs": source_runs,
                "note": "Local aggregation of explicit source runs; no new inference and no selective retry replacement."}
    return gold, records, combined


def render_comparison(report, confidence, subset_report):
    pct = lambda v: f"{100*v:.2f}%" if v is not None else "N/A"
    fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    lines = ["# Secret Keepers: matched 500-row model comparison", "",
             "Each ranked model was attempted on the same 500 frozen rows. Canonical typed precision, recall and micro F1 use the same scorer. All failed requests remain in the denominators. These are untuned model baselines.", "",
             "| Model | Precision | Recall | Canonical F1 | Leak rate | p50 s | p95 s | USD / 1,000 docs | Usable outputs |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in report["models"]:
        cost = f"${m['estimated_usd_per_1000_docs']:.4f}" if m["estimated_usd_per_1000_docs"] is not None else "Unknown usage"
        lines.append("| " + " | ".join([m["model"].rsplit("/",1)[-1], pct(m["typed"]["precision"]),
            pct(m["typed"]["recall"]), pct(m["typed"]["f1"]), pct(m["untyped_leak_rate"]),
            fmt(m["latency_seconds"]["p50"]), fmt(m["latency_seconds"]["p95"]), cost,
            f"{round(m['usable_output_rate'] * m['expected_rows'])}/{m['expected_rows']}"]) + " |")
    lines += ["", "## Reliability and cost completeness", "",
              "| Model | Hallucination | Empty-gold over-redaction | Exact-row match | Known-usage USD / 500 | Unknown-cost rows |",
              "| --- | --- | --- | --- | --- | --- |"]
    for m in report["models"]:
        lines.append(f"| {m['model'].rsplit('/',1)[-1]} | {pct(m['hallucination_rate'])} | {pct(m['zero_pii_overredaction_rate'])} | {pct(m['exact_row_match_rate'])} | ${m['known_usage_estimated_usd']:.6f} | {m['unknown_cost_rows']} |")
    lines += ["", "Unknown-cost rows have no reliable token accounting or require unpriced cache-write details. Known-usage costs are lower bounds when any costs are unknown. Failed requests are never assumed free. Estimates use dated Standard rates, not billing invoices.", "",
              "Request failures by model:"]
    for m in report["models"]:
        lines.append(f"- {m['model'].rsplit('/',1)[-1]}: " + json.dumps(m["errors"]))
    lines += ["", "Length failures mean the completion exceeded the fixed output budget. They are retained as empty predictions, with all gold counted as missed. Their reported token usage still contributes to cost.", "",
              "## Uncertainty", "", "95% row-bootstrap intervals, 1,000 replicates, seed 7. These are descriptive intervals, not a multiple-comparison-corrected significance test.", "",
              "| Model | Canonical F1 interval | Leak-rate interval |", "| --- | --- | --- |"]
    for m in report["models"]:
        c = confidence["models"][m["model"]]
        interval_text = lambda bounds: " to ".join(pct(v) for v in bounds) if bounds else "N/A"
        lines.append(f"| {m['model'].rsplit('/',1)[-1]} | {interval_text(c['f1_95ci'])} | {interval_text(c['leak_rate_95ci'])} |")
    lines += ["", "Paired difference intervals versus the prespecified Sol reference are in confidence_intervals.json. Small ranking differences should not be treated as certain wins.", "",
              "## Original preview excluded", "", "The following view excludes the ten fixed IDs from the original manual preview. The remaining 490 are not a newly unused test set: they were already scored for earlier open-model runs. No model prompts were tuned from these comparison results.", "",
              "| Model | Precision | Recall | Canonical F1 | Leak rate |", "| --- | --- | --- | --- | --- |"]
    for m in subset_report["models"]:
        lines.append(f"| {m['model'].rsplit('/',1)[-1]} | {pct(m['typed']['precision'])} | {pct(m['typed']['recall'])} | {pct(m['typed']['f1'])} | {pct(m['untyped_leak_rate'])} |")
    lines += ["", "## Definitions and limitations", "",
              "- Canonical F1 matches exact values after fixed edge punctuation/whitespace trimming and merged labels. Matching is case-sensitive, unique per value/label pair per row. Precision = TP/(TP+FP); recall = TP/(TP+FN).",
              "- Leak rate ignores labels and counts gold values not covered by a grounded predicted value. It is value-based, not a complete span-level privacy audit. Invalid, failed and missing outputs contribute empty predictions.",
              "- Every ranked model uses concurrency five, a 30-second timeout and a 2,048-token output cap. OpenAI uses low reasoning and omits temperature; Fireworks uses temperature 0 and the model-specific reasoning settings recorded in each source manifest.",
              "- Runs took place at different wall-clock times. Provider load, schema warmup, tokenizers and cache behavior differ; latency is descriptive and not proof of an infrastructure-only speedup.",
              "- The gold is synthetic and may contain annotation errors. CARDISSUER has only one instance. Per-label support and scores are in per_label.md. Strict numbered raw-label F1 is unavailable because those labels were merged in the frozen gold.",
              "- Nemotron was explicitly skipped after timeouts. Its earlier five-row training smoke returned 4/5 schema-valid outputs; a later ten-row batch was interrupted after four saved records. It has no matched 500-row score and is not ranked here.",
              "- Astra and Claude have not been tested and are not included. No fine-tuned model has been evaluated yet.", "",
              "## Source runs", ""]
    for source in report["manifest"]["source_runs"]:
        m = source["manifest"]
        lines.append(f"- {source['path']}: {', '.join(m['models'])}; concurrency {m['concurrency']}; retries {m['retries']}.")
    lines += ["", "Artifacts: metrics.json, metrics_preview_excluded.json, predictions.jsonl, row_metrics.jsonl, confidence_intervals.json, per_label.md, and manifest.json. The aggregation made no API calls.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold_path, labels_path = DATA_ROOT / "test_set.jsonl", DATA_ROOT / "labels.json"
    gold, records, manifest = load_matched(args.runs, gold_path, labels_path)
    taxonomy = json.loads(labels_path.read_text())
    prices = json.loads((ROOT / "pipeline/comparison_prices_2026-11-02.json").read_text())
    report, diagnostics = evaluate(gold, records, manifest, taxonomy, prices)
    protocol = json.loads((ROOT / "pipeline/evaluation_protocol.json").read_text())
    ids = [i for i in manifest["ids"] if i not in protocol["previewed_ids"]]
    subset_manifest = dict(manifest, ids=ids, rows=len(ids))
    subset, _ = evaluate(gold, [r for r in records if r["id"] in ids], subset_manifest, taxonomy, prices)
    confidence = bootstrap(diagnostics, manifest["models"], manifest["ids"])
    output = args.output or ROOT / "pipeline/outputs" / ("comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"evaluator_sha256": hashlib.sha256((ROOT / "scripts/evaluate.py").read_bytes()).hexdigest(),
                  "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "comparison_protocol_sha256": hashlib.sha256((ROOT / "pipeline/comparison_protocol.json").read_bytes()).hexdigest()}
    report["provenance"] = provenance
    for name, obj in [("manifest.json", manifest), ("metrics.json", report),
                      ("metrics_preview_excluded.json", subset), ("confidence_intervals.json", confidence)]:
        (output / name).write_text(json.dumps(obj, indent=2) + "\n")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "row_metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in diagnostics))
    write_markdown(report, output / "per_label.md")
    markdown = render_comparison(report, confidence, subset)
    (output / "comparison.md").write_text(markdown)
    print(f"Compared {len(manifest['models'])} models × {len(manifest['ids'])} rows; artifacts: {output}")
    print(markdown.split("## Reliability")[0])


if __name__ == "__main__":
    main()
