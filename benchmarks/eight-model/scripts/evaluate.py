"""Offline, gold-based PII evaluation. Never calls an inference API."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import string

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parents[1] / "data"
EDGE = string.whitespace + string.punctuation


def normalize(value):
    return value.strip(EDGE)


def ratio(a, b):
    return a / b if b else None


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def parse_entities(record, allowed):
    if record is None or not isinstance(record.get("raw_output"), str):
        return [], False
    try:
        data = json.loads(record["raw_output"])
        if not isinstance(data, dict) or set(data) != {"entities"} or not isinstance(data["entities"], list):
            return [], False
        for entity in data["entities"]:
            if (not isinstance(entity, dict) or set(entity) != {"value", "label"}
                    or not isinstance(entity["value"], str) or not entity["value"]
                    or not isinstance(entity["label"], str) or entity["label"] not in allowed):
                return [], False
        return data["entities"], True
    except (ValueError, TypeError):
        return [], False


def score_row(gold, record, labels, mapping):
    entities, json_valid = parse_entities(record, set(labels) | set(mapping))
    usable = (json_valid and not record.get("error") and record.get("finish_reason") == "stop")
    if not usable:
        entities = []
    def pairs(items):
        return {(normalize(e["value"]), mapping.get(e["label"], e["label"])) for e in items}
    expected, predicted = pairs(gold["entities"]), pairs(entities)
    matched = expected & predicted
    missed, extra = expected - predicted, predicted - expected
    gold_values = {value for value, _ in expected}
    pred_values = {value for value, _ in predicted}
    # A value can protect gold only if it really occurs in the source. A
    # hallucinated longer string containing gold cannot falsely hide a leak.
    grounded = {value for value in pred_values if value and value in gold["source_text"]}
    leaked = {value for value in gold_values if not value or not any(value in p for p in grounded)}
    hallucinated = {value for value in pred_values if not value or value not in gold["source_text"]}
    per_label = {}
    for label in labels:
        per_label[label] = {"tp": sum(l == label for _, l in matched),
                            "fp": sum(l == label for _, l in extra),
                            "fn": sum(l == label for _, l in missed),
                            "support": sum(l == label for _, l in expected)}
    return {"id": str(gold["id"]), "json_valid": json_valid, "usable_output": bool(usable),
            "exact_row_match": bool(usable and expected == predicted),
            "gold_pairs": len(expected), "predicted_pairs": len(predicted),
            "tp": len(matched), "fp": len(extra), "fn": len(missed),
            "gold_values": len(gold_values), "predicted_values": len(pred_values),
            "leaked_values": len(leaked), "hallucinated_values": len(hallucinated),
            "zero_pii": not expected, "overredacted_zero_pii": bool(not expected and predicted),
            "per_label": per_label,
            "missed_entities": [{"value": v, "label": l} for v, l in sorted(missed)],
            "extra_entities": [{"value": v, "label": l} for v, l in sorted(extra)],
            "leaked_entity_values": sorted(leaked),
            "error": record.get("error") if record else "Missing prediction"}


def cost_of(record, rates):
    usage = record.get("usage", {}) if record else {}
    if not rates or not isinstance(usage, dict):
        return None
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    # A separate cache-write charge needs its duration/rate to be known. Do not
    # silently price it as ordinary input when a provider reports such writes.
    if (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0):
        return None
    if (not all(isinstance(n, int) and n >= 0 for n in [prompt, output, cached]) or cached > prompt):
        return None
    return ((prompt - cached) * rates["input"] + cached * rates["cached_input"] + output * rates["output"]) / 1e6


def evaluate(gold_rows, predictions, manifest, taxonomy, pricebook):
    gold_by_id = {str(r["id"]): r for r in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Duplicate gold row ID")
    ids, models = manifest["ids"], manifest["models"]
    if not ids or len(ids) != len(set(ids)) or len(models) != len(set(models)):
        raise ValueError("Manifest must have unique IDs and models")
    if any(row_id not in gold_by_id for row_id in ids):
        raise ValueError("Manifest references missing gold rows")
    labels = taxonomy["canonical"] if isinstance(taxonomy, dict) else taxonomy
    mapping = taxonomy.get("raw_to_canonical", {}) if isinstance(taxonomy, dict) else {}
    for row_id in ids:
        for e in gold_by_id[row_id]["entities"]:
            if not isinstance(e.get("value"), str) or mapping.get(e["label"], e["label"]) not in labels:
                raise ValueError("Invalid gold entity")
    indexed = {}
    for record in predictions:
        key = record["model"], str(record["id"])
        if key in indexed:
            raise ValueError("Duplicate prediction for a model/row; do not mix retries")
        if key[0] not in models or key[1] not in ids:
            raise ValueError("Prediction outside the manifest")
        if record.get("source_text") != gold_by_id[key[1]]["source_text"]:
            raise ValueError("Prediction source text differs from frozen input")
        indexed[key] = record
    report, diagnostics = [], []
    for model in models:
        records = [indexed.get((model, row_id)) for row_id in ids]
        scored = [score_row(gold_by_id[row_id], record, labels, mapping) for row_id, record in zip(ids, records)]
        for score in scored:
            diagnostics.append(dict(score, model=model))
        total = Counter()
        for row in scored:
            total.update({k: int(row[k]) for k in ["tp", "fp", "fn", "gold_values", "predicted_values",
                         "leaked_values", "hallucinated_values", "json_valid", "usable_output", "exact_row_match",
                         "zero_pii", "overredacted_zero_pii"]})
        per_label = {}
        for label in labels:
            counts = Counter()
            for row in scored:
                counts.update(row["per_label"][label])
            per_label[label] = dict(prf(counts["tp"], counts["fp"], counts["fn"]), support=counts["support"])
        timings = sorted(r["elapsed_ms"] / 1000 for r in records if r and isinstance(r.get("elapsed_ms"), (int, float)))
        costs = [cost_of(r, pricebook.get("models", {}).get(model)) for r in records]
        known_cost = sum(c for c in costs if c is not None)
        unknown_cost = sum(c is None for c in costs)
        errors = Counter((r.get("error") or "Nonusable completion") if r else "Missing prediction"
                         for r, s in zip(records, scored) if not s["usable_output"])
        report.append({"model": model, "expected_rows": len(ids), "recorded_rows": sum(r is not None for r in records),
                       "typed": prf(total["tp"], total["fp"], total["fn"]),
                       "untyped_leak_rate": ratio(total["leaked_values"], total["gold_values"]),
                       "leaked_values": total["leaked_values"], "gold_values": total["gold_values"],
                       "hallucination_rate": ratio(total["hallucinated_values"], total["predicted_values"]),
                       "hallucinated_values": total["hallucinated_values"], "predicted_values": total["predicted_values"],
                       "json_validity_rate": total["json_valid"] / len(ids),
                       "usable_output_rate": total["usable_output"] / len(ids),
                       "exact_row_match_rate": total["exact_row_match"] / len(ids),
                       "zero_pii_rows": total["zero_pii"],
                       "zero_pii_overredaction_rate": ratio(total["overredacted_zero_pii"], total["zero_pii"]),
                       "latency_seconds": {"observations": len(timings),
                           "p50": statistics.median(timings) if timings else None,
                           "p95": timings[math.ceil(.95 * len(timings)) - 1] if timings else None,
                           "definition": "client elapsed including errors; nearest-rank p95"},
                       "known_usage_estimated_usd": known_cost, "unknown_cost_rows": unknown_cost,
                       "estimated_total_usd": None if unknown_cost else known_cost,
                       "estimated_usd_per_1000_docs": None if unknown_cost else known_cost / len(ids) * 1000,
                       "errors": dict(errors), "per_label": per_label})
    return {"models": report, "normalization": "strip Python ASCII string.punctuation and whitespace at both edges; case-sensitive; unique pairs per row",
            "strict_raw_label_f1": None, "strict_raw_label_note": "Frozen gold contains canonical labels only; original numbered labels cannot be recovered.",
            "pricebook": pricebook, "manifest": manifest}, diagnostics


def write_markdown(report, output):
    def pct(n):
        return f"{n * 100:.2f}%" if n is not None else "N/A"
    def num(n):
        return f"{n:.3f}" if n is not None else "N/A"
    lines = ["# Frozen-set PII evaluation", "", "All metrics use gold annotations. Invalid, failed, and missing outputs contribute empty predictions and missed gold entities.", "",
             "| Model | Typed F1 | Precision | Recall | Leak rate | JSON valid | Usable output | p50 s | p95 s | USD / 1,000 docs |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in report["models"]:
        cost = f"${m['estimated_usd_per_1000_docs']:.4f}" if m["estimated_usd_per_1000_docs"] is not None else "Unknown (missing usage)"
        lines.append("| " + " | ".join([m["model"].rsplit("/", 1)[-1], pct(m["typed"]["f1"]), pct(m["typed"]["precision"]),
            pct(m["typed"]["recall"]), pct(m["untyped_leak_rate"]), pct(m["json_validity_rate"]), pct(m["usable_output_rate"]),
            num(m["latency_seconds"]["p50"]), num(m["latency_seconds"]["p95"]), cost]) + " |")
    lines += ["", "Value normalization: " + report["normalization"] + ".", "",
              "Leak coverage ignores labels: a gold value is protected only when contained in a predicted value that actually occurs in the source. Full-value coverage is required. Hallucination counts normalized predicted values absent from the source. Both rates use unique values per row.", "",
              "Latency includes failed attempts with recorded elapsed times. Costs use reported tokens and dated standard prices; unknown usage is never treated as zero. Known-usage estimates are in metrics.json, including billed invalid completions.", "",
              report["strict_raw_label_note"], "", "## Per-label scores", ""]
    labels = list(report["models"][0]["per_label"])
    lines += ["| Label | Gold support | " + " | ".join(m["model"].rsplit("/", 1)[-1] + " F1" for m in report["models"]) + " |",
              "| --- | --- | " + " | ".join("---" for _ in report["models"]) + " |"]
    for label in labels:
        support = report["models"][0]["per_label"][label]["support"]
        lines.append(f"| {label} | {support} | " + " | ".join(pct(m["per_label"][label]["f1"]) for m in report["models"]) + " |")
    lines += ["", "## Reliability and annotation notes", ""]
    for m in report["models"]:
        lines += [f"- {m['model'].rsplit('/',1)[-1]}: {m['recorded_rows']}/{m['expected_rows']} recorded rows; hallucination rate {pct(m['hallucination_rate'])}; zero-PII over-redaction {pct(m['zero_pii_overredaction_rate'])}; exact-row match {pct(m['exact_row_match_rate'])}; {m['unknown_cost_rows']} rows with unknown cost.",
                  "  Errors: " + json.dumps(m["errors"])]
    lines += ["", "The first ten frozen rows were previewed earlier. Report the full 500-row score with that disclosure and report the remaining 490 separately; these are not 500 previously unseen cases. CARDISSUER has support 1, below the intended minimum of 10. Gold is synthetic and may have annotation errors; scores measure agreement with this frozen gold.", ""]
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DATA_ROOT / "test_set.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA_ROOT / "labels.json")
    parser.add_argument("--prices", type=Path, default=ROOT / "pipeline/prices_2026-11-02.json")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    for file, key in [(args.gold, "input_sha256"), (args.labels, "labels_sha256")]:
        if hashlib.sha256(file.read_bytes()).hexdigest() != manifest[key]:
            parser.error(f"Frozen file hash mismatch: {file.name}")
    gold = [json.loads(x) for x in args.gold.read_text().splitlines() if x.strip()]
    predictions = [json.loads(x) for x in (args.run / "predictions.jsonl").read_text().splitlines() if x.strip()]
    taxonomy, prices = json.loads(args.labels.read_text()), json.loads(args.prices.read_text())
    report, diagnostics = evaluate(gold, predictions, manifest, taxonomy, prices)
    report["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.run / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.run / "row_metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in diagnostics))
    write_markdown(report, args.run / "evaluation.md")
    # Fixed in the public protocol before the full run, not selected by score.
    protocol_path = ROOT / "pipeline/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    preview_ids = set(protocol["previewed_ids"])
    unseen_ids = [row_id for row_id in manifest["ids"] if row_id not in preview_ids]
    if preview_ids and unseen_ids and len(unseen_ids) < len(manifest["ids"]):
        subset_manifest = dict(manifest, ids=unseen_ids, rows=len(unseen_ids))
        unseen, _ = evaluate(gold, [r for r in predictions if str(r["id"]) in unseen_ids], subset_manifest, taxonomy, prices)
        (args.run / "metrics_unpreviewed.json").write_text(json.dumps(unseen, indent=2) + "\n")
    print((args.run / "evaluation.md").read_text().split("## Per-label")[0])


if __name__ == "__main__":
    main()
