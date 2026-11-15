"""Gold-based scoring fixtures; no network and no live model calls."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import score_row, evaluate, cost_of

LABELS = ["GIVENNAME", "EMAIL", "DATE"]
MAPPING = {"GIVENNAME1": "GIVENNAME"}


def prediction(entities, **kwargs):
    record = {"raw_output": json.dumps({"entities": entities}), "finish_reason": "stop", "error": None}
    record.update(kwargs)
    return record


def gold(text, entities):
    return {"id": "1", "source_text": text, "entities": entities}


def entity(value, label="GIVENNAME"):
    return {"value": value, "label": label}


def test_exact_match_duplicates_trim_and_label_merge():
    row = gold("Jane jane@example.com", [entity("Jane"), entity("jane@example.com", "EMAIL")])
    output = prediction([entity(" Jane,", "GIVENNAME1"), entity("Jane"), entity("jane@example.com", "EMAIL")])
    scored = score_row(row, output, LABELS, MAPPING)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (2, 0, 0)
    assert scored["exact_row_match"] and scored["leaked_values"] == 0


def test_wrong_type_is_fp_and_fn_but_does_not_leak():
    row = gold("Jane", [entity("Jane")])
    scored = score_row(row, prediction([entity("Jane", "DATE")]), LABELS, MAPPING)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (0, 1, 1)
    assert scored["leaked_values"] == 0


def test_partial_name_leaks_but_grounded_larger_span_covers():
    row = gold("Jane Smith wrote", [entity("Jane Smith")])
    assert score_row(row, prediction([entity("Jane")]), LABELS, MAPPING)["leaked_values"] == 1
    full = score_row(row, prediction([entity("Jane Smith wrote")]), LABELS, MAPPING)
    assert full["leaked_values"] == 0 and full["fp"] == 1


def test_hallucinated_longer_value_cannot_hide_gold():
    row = gold("Jane", [entity("Jane")])
    scored = score_row(row, prediction([entity("Jane Smith")]), LABELS, MAPPING)
    assert scored["leaked_values"] == 1 and scored["hallucinated_values"] == 1


@pytest.mark.parametrize("record", [None, prediction([], raw_output="broken"),
    prediction([entity("Jane")], error="timeout"), prediction([entity("Jane")], finish_reason="length")])
def test_bad_requests_remain_in_recall_and_leak_denominators(record):
    scored = score_row(gold("Jane", [entity("Jane")]), record, LABELS, MAPPING)
    assert scored["fn"] == 1 and scored["leaked_values"] == 1
    assert not scored["usable_output"]


def test_empty_gold_and_predictions_are_correct_but_failed_empty_is_not():
    row = gold("Hello", [])
    assert score_row(row, prediction([]), LABELS, MAPPING)["exact_row_match"]
    assert not score_row(row, None, LABELS, MAPPING)["exact_row_match"]
    assert score_row(row, prediction([entity("Hello")]), LABELS, MAPPING)["overredacted_zero_pii"]


def test_cost_includes_cache_and_reasoning_in_output_once():
    record = {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 80},
                        "completion_tokens_details": {"reasoning_tokens": 15}}}
    assert cost_of(record, {"input": 1, "cached_input": .1, "output": 2}) == pytest.approx(68 / 1e6)
    assert cost_of({}, {"input": 1, "cached_input": .1, "output": 2}) is None
    record["usage"]["prompt_tokens_details"]["cache_write_tokens"] = 10
    assert cost_of(record, {"input": 1, "cached_input": .1, "output": 2}) is None


def test_aggregate_missing_row_micro_f1_and_unknown_cost():
    rows = [gold("Jane", [entity("Jane")]), dict(gold("Bob", [entity("Bob")]), id="2")]
    manifest = {"ids": ["1", "2"], "models": ["m"]}
    pred = dict(prediction([entity("Jane")]), model="m", id="1", source_text="Jane", elapsed_ms=20)
    report, _ = evaluate(rows, [pred], manifest, LABELS, {})
    m = report["models"][0]
    assert m["typed"]["f1"] == pytest.approx(2 / 3)
    assert m["untyped_leak_rate"] == .5 and m["json_validity_rate"] == .5
    assert m["estimated_total_usd"] is None and m["unknown_cost_rows"] == 2
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate(rows, [pred, pred], manifest, LABELS, {})
    with pytest.raises(ValueError, match="source text"):
        evaluate(rows, [dict(pred, source_text="changed")], manifest, LABELS, {})
