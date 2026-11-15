"""Download a small, revision-pinned training slice; never touch validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "ai4privacy/pii-masking-300k"
ROOT = Path(__file__).resolve().parents[1]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def prepare(output: Path, count: int, scan_rows: int, seed: int, max_chars: int):
    if count < 1 or scan_rows < count or max_chars < 1:
        raise ValueError("Require rows > 0, scan-rows >= rows, and max-chars > 0")
    if output.exists():
        raise ValueError("Output directory exists; use a new --output to preserve the previous sample")
    info = get_json("https://huggingface.co/api/datasets/" + DATASET)
    revision = info["sha"]
    files = sorted(item["rfilename"] for item in info["siblings"]
                   if item["rfilename"].startswith("data/train/")
                   and item["rfilename"].endswith(".jsonl"))
    if not files:
        raise ValueError("No training JSONL files found in dataset metadata")
    candidates, labels, scanned, seen_ids = [], set(), 0, set()
    for name in files:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{urllib.parse.quote(name)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            for line in response:
                if not line.strip():
                    continue
                row = json.loads(line)
                scanned += 1
                if str(row.get("language", "")).lower() in {"english", "en"}:
                    mask = row["privacy_mask"]
                    if isinstance(mask, str):
                        mask = json.loads(mask)
                    labels.update(entity["label"] for entity in mask)
                    text, row_id = row["source_text"], str(row["id"])
                    if 0 < len(text) <= max_chars and row_id not in seen_ids:
                        seen_ids.add(row_id)
                        candidates.append({"id": row_id, "source_text": text,
                                           "gold_entities": mask, "language": row["language"],
                                           "dataset": DATASET, "split": "train",
                                           "revision": revision})
                if scanned >= scan_rows:
                    break
        if scanned >= scan_rows:
            break
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible English rows; increase --scan-rows")
    sample = random.Random(seed).sample(candidates, count)
    output.mkdir(parents=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample)
    (output / "inputs.jsonl").write_text(data)
    (output / "labels.json").write_text(json.dumps(sorted(labels), indent=2) + "\n")
    manifest = {"dataset": DATASET, "revision": revision, "split": "train",
                "purpose": "smoke test only; not the locked evaluation set",
                "seed": seed, "rows": count, "rows_scanned": scanned,
                "max_chars": max_chars, "ids": [row["id"] for row in sample],
                "sha256": hashlib.sha256(data.encode()).hexdigest(),
                "label_scope": "English rows in the scanned training slice; not full taxonomy",
                "source": "https://huggingface.co/datasets/" + DATASET}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {count} English training rows; {len(labels)} labels; {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pipeline/data/smoke")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--scan-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-chars", type=int, default=512)
    args = parser.parse_args()
    prepare(args.output, args.rows, args.scan_rows, args.seed, args.max_chars)


if __name__ == "__main__":
    main()
