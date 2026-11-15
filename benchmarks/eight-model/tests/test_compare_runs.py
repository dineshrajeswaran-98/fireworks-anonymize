"""Offline checks for matched-run loading and paired bootstrap intervals."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import bootstrap, load_matched


def test_bootstrap_paired_differences_have_correct_direction_and_are_reproducible():
    rows = []
    for row_id in ["1", "2"]:
        rows.extend([{"model": "good", "id": row_id, "tp": 1, "fp": 0, "fn": 0, "leaked_values": 0, "gold_values": 1},
                     {"model": "ref", "id": row_id, "tp": 0, "fp": 0, "fn": 1, "leaked_values": 1, "gold_values": 1}])
    first = bootstrap(rows, ["good", "ref"], ["1", "2"], reps=100, reference="ref")
    assert first == bootstrap(rows, ["good", "ref"], ["1", "2"], reps=100, reference="ref")
    assert first["models"]["good"]["f1_difference_vs_reference_95ci"] == [1, 1]
    assert first["models"]["good"]["leak_rate_difference_vs_reference_95ci"] == [-1, -1]


def test_matched_loader_rejects_partial_and_duplicate_model_runs(tmp_path):
    gold = tmp_path / "gold.jsonl"
    labels = tmp_path / "labels.json"
    gold.write_text(json.dumps({"id": "1", "source_text": "Hello", "entities": []}) + "\n")
    labels.write_text('["EMAIL"]')
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"input_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "models": ["m"], "ids": ["1"], "rows": 1}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        load_matched([run], gold, labels, expected_rows=1)
    (run / "predictions.jsonl").write_text(json.dumps({"model": "m", "id": "1", "source_text": "Hello"}) + "\n")
    _, records, combined = load_matched([run], gold, labels, expected_rows=1)
    assert len(records) == 1 and combined["models"] == ["m"]
    with pytest.raises(ValueError, match="Duplicate model"):
        load_matched([run, run], gold, labels, expected_rows=1)
