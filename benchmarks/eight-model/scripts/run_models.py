"""Run text rows through Fireworks or direct OpenAI; no local models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from fireworks_client import ROOT, FireworksError, extract, load_env, model_id
import openai_client


def run_one(row, model, labels, provider="fireworks"):
    started = time.perf_counter()
    result = {"schema_version": 1, "id": row["id"], "provider": provider, "model": model,
              "source_text": row["source_text"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "entities": None, "raw_output": None, "usage": {}, "json_valid": False,
              "error": None, "attempts": 1}
    try:
        extractor = openai_client.extract if provider == "openai" else extract
        result.update(extractor(row["source_text"], labels, model=model))
    except (FireworksError, openai_client.OpenAIError) as exc:
        result["error"] = str(exc)
    except Exception as exc:
        # No exception body: it could contain source text or credentials.
        result["error"] = "Unexpected request failure: " + type(exc).__name__
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["fireworks", "openai"], default="fireworks")
    parser.add_argument("--input", type=Path, default=ROOT / "pipeline/data/smoke/inputs.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "pipeline/data/smoke/labels.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", action="append", help="Exact selected-provider model ID; repeat to compare models")
    parser.add_argument("--limit", type=int, default=5, help="Maximum input rows (default: 5)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print request count without any API calls")
    args = parser.parse_args()
    load_env()
    if args.limit < 1 or args.concurrency < 1 or args.progress_every < 1:
        parser.error("limit, concurrency, and progress-every must be positive")
    prefix = "OPENAI" if args.provider == "openai" else "FIREWORKS"
    models = args.model or [m.strip() for m in os.environ.get(prefix + "_MODELS", "").split(",") if m.strip()]
    if not models:
        models = [openai_client.model_id() if args.provider == "openai" else model_id()]
    if len(set(models)) != len(models):
        parser.error("Duplicate model IDs")
    labels = json.loads(args.labels.read_text())
    # The teammate's frozen taxonomy includes canonical labels plus a mapping.
    # Use only the shared canonical list; never send the row's gold entities.
    if isinstance(labels, dict):
        labels = labels.get("canonical")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
        parser.error("labels must be a nonempty list of strings or an object with a canonical list")
    rows, ids = [], set()
    with args.input.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("source_text"), str) or not row["source_text"] or "id" not in row:
                parser.error("Each row needs id and nonempty source_text")
            row["id"] = str(row["id"])
            if row["id"] in ids:
                parser.error("Duplicate input ID")
            ids.add(row["id"])
            rows.append(row)
            if len(rows) == args.limit:
                break
    if not rows:
        parser.error("Input contains no rows")
    print(f"{len(rows)} rows × {len(models)} models = {len(rows) * len(models)} requests; concurrency {args.concurrency}")
    if args.dry_run:
        print("Dry run: no API calls made.")
        return
    if not os.environ.get(prefix + "_API_KEY", "").strip():
        parser.error(f"Add {prefix}_API_KEY to .env first")
    output = args.output or ROOT / "pipeline/outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    client_file = ROOT / "scripts" / ("openai_client.py" if args.provider == "openai" else "fireworks_client.py")
    manifest = {"provider": args.provider,
                "input_path": str(args.input.resolve()),
                "labels_path": str(args.labels.resolve()),
                "label_taxonomy": labels,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                "client_sha256": hashlib.sha256(client_file.read_bytes()).hexdigest(),
                "shared_extraction_sha256": hashlib.sha256((ROOT / "scripts/fireworks_client.py").read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "evaluator_sha256": hashlib.sha256((ROOT / "scripts/evaluate.py").read_bytes()).hexdigest() if (ROOT / "scripts/evaluate.py").exists() else None,
                "evaluation_protocol_sha256": hashlib.sha256((ROOT / "pipeline/evaluation_protocol.json").read_bytes()).hexdigest() if (ROOT / "pipeline/evaluation_protocol.json").exists() else None,
                "models": models, "rows": len(rows), "ids": [r["id"] for r in rows],
                "concurrency": args.concurrency, "retries": 0,
                "max_tokens": int(os.environ.get(prefix + "_MAX_TOKENS", "2048")),
                "temperature": os.environ.get(prefix + "_TEMPERATURE", "" if args.provider == "openai" else "0"),
                "timeout_seconds": float(os.environ.get(prefix + "_TIMEOUT", "30" if args.provider == "openai" else "120")),
                "endpoint": openai_client.ENDPOINT if args.provider == "openai" else "https://api.fireworks.ai/inference/v1/chat/completions",
                "reasoning_profile": os.environ.get("OPENAI_REASONING_EFFORT", "") if args.provider == "openai" else os.environ.get("FIREWORKS_REASONING_PROFILE", "default"),
                "output_mode": "json_schema" if args.provider == "openai" else os.environ.get("FIREWORKS_OUTPUT_MODE", "json_schema"),
                "mode": "non-streaming; JSON object with entities array"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    errors = 0
    # Models run sequentially to avoid competing for concurrency during comparisons.
    completed = 0
    with (output / "predictions.jsonl").open("w") as target:
        for model in models:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run_one, row, model, labels, args.provider) for row in rows]
                for future in as_completed(futures):
                    result = future.result()
                    errors += result["error"] is not None
                    target.write(json.dumps(result, ensure_ascii=False) + "\n")
                    target.flush()
                    completed += 1
                    if completed % args.progress_every == 0 or completed == len(rows) * len(models):
                        print(f"{completed}/{len(rows) * len(models)} completed; {errors} errors; model {model.rsplit('/', 1)[-1]}", flush=True)
    print(f"Saved predictions to {output}; {errors} error(s)")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
