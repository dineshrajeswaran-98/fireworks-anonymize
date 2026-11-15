"""Offline transport, output contract, and dataset sampling checks."""
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fireworks_client as client
import prepare_smoke
import run_models


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    monkeypatch.setattr(client, "load_env", lambda: None)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-test-key")
    monkeypatch.setenv("FIREWORKS_MODEL", "test-model")


def result(raw, finish="stop"):
    return {"raw_output": raw, "finish_reason": finish, "usage": {"prompt_tokens": 12}, "latency_ms": 1}


def test_fireworks_request_uses_remote_endpoint_and_records_usage(monkeypatch):
    class Response(io.BytesIO):
        headers = {"x-request-id": "test-request"}
    def respond(request, timeout):
        assert request.full_url == client.ENDPOINT
        assert request.get_header("Authorization") == "Bearer fake-test-key"
        body = json.loads(request.data)
        assert body["model"] == "test-model" and body["stream"] is False
        assert "prompt" not in body  # no Ollama request format
        return Response(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())
    monkeypatch.setattr(client.urllib.request, "urlopen", respond)
    output = client.complete([{"role": "user", "content": "Synthetic input"}])
    assert output["usage"]["prompt_tokens"] == 10
    assert output["request_id"] == "test-request"
    assert output["latency_ms"] >= 0


def test_invalid_json_is_not_silently_an_empty_prediction(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: result("not json"))
    output = client.extract("sample", ["EMAIL"])
    assert output["entities"] is None and output["error"] and not output["json_valid"]


@pytest.mark.parametrize("model,effort", [
    ("accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b", "none"),
    ("accounts/fireworks/models/gpt-oss-120b", "low"),
    ("accounts/fireworks/models/glm-5p3-flash", "low"),
    ("accounts/fireworks/models/minimax-m3", "low"),
    ("accounts/fireworks/models/qwen3p8-max", "none"),
    ("accounts/fireworks/models/deepseek-v4p1-flash-1017", "none"),
])
def test_schema_and_extraction_reasoning_profile(monkeypatch, model, effort):
    monkeypatch.setenv("FIREWORKS_REASONING_PROFILE", "extraction")
    monkeypatch.setenv("FIREWORKS_OUTPUT_MODE", "json_schema")
    class Response(io.BytesIO):
        headers = {}
    def respond(request, timeout):
        body = json.loads(request.data)
        assert body["reasoning_effort"] == effort
        fmt = body["response_format"]
        assert fmt["type"] == "json_schema"
        schema = fmt["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["entities"]["items"]["properties"]["label"]["enum"] == ["EMAIL"]
        return Response(json.dumps({"choices": [{"message": {"content": '{"entities":[]}'}, "finish_reason": "stop"}]}).encode())
    monkeypatch.setattr(client.urllib.request, "urlopen", respond)
    output = client.extract("Hello", ["EMAIL"], model=model)
    assert output["entities"] == [] and output["error"] is None


def test_hallucinated_values_are_preserved_for_evaluator(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: result('{"entities":[{"value":"invented","label":"EMAIL"}]}'))
    output = client.extract("sample", ["EMAIL"])
    assert output["entities"][0]["value"] == "invented"


def test_truncated_json_is_flagged_even_if_parseable(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: result('{"entities":[]}', "length"))
    assert client.extract("sample", ["EMAIL"])["error"]


def test_gold_never_enters_model_request(monkeypatch):
    def check(text, labels, **kwargs):
        assert text == "Hello" and labels == ["EMAIL"]
        assert "gold_entities" not in kwargs
        return dict(result('{"entities":[]}'), entities=[], json_valid=True, error=None)
    monkeypatch.setattr(run_models, "extract", check)
    output = run_models.run_one({"id": "1", "source_text": "Hello", "gold_entities": ["SECRET-GOLD"]}, "m", ["EMAIL"])
    assert "SECRET-GOLD" not in json.dumps(output)


def test_http_error_does_not_expose_provider_body(monkeypatch):
    def fail(*a, **k):
        raise urllib.error.HTTPError(client.ENDPOINT, 401, "private detail", {}, io.BytesIO(b"sensitive provider body"))
    monkeypatch.setattr(client.urllib.request, "urlopen", fail)
    with pytest.raises(client.FireworksError, match="HTTP 401") as error:
        client.complete([])
    assert "private" not in str(error.value) and "sensitive" not in str(error.value)


@pytest.mark.parametrize("model,expected", [
    ("accounts/fireworks/models/gpt-oss-120b", None),
    ("gpt-oss-120b", None),
    ("claude-sonnet-5-5", "X-anthropic-api-key"),
])
def test_provider_keys_only_forwarded_to_matching_pinned_model(monkeypatch, model, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    class Response(io.BytesIO):
        headers = {}
    def respond(request, timeout):
        actual = {key for key in request.headers if key.lower() in {"x-anthropic-api-key", "x-openai-api-key"}}
        assert actual == ({expected} if expected else set())
        assert request.full_url == client.ENDPOINT
        return Response(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())
    monkeypatch.setattr(client.urllib.request, "urlopen", respond)
    client.complete([], model=model)


def test_proprietary_gpt_cannot_be_routed_to_fireworks(monkeypatch):
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Must not call Fireworks"))
    with pytest.raises(client.FireworksError, match="provider openai"):
        client.complete([], model="gpt-6-astra")


def test_sample_is_pinned_english_train_and_gold_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_smoke, "get_json", lambda url: {"sha": "revision123", "siblings": [
        {"rfilename": "data/train/a.jsonl"}, {"rfilename": "data/validation/a.jsonl"}]})
    rows = [{"id": "1A", "source_text": "Hi Jane", "language": "English",
             "privacy_mask": json.dumps([{"value": "Jane", "label": "GIVENNAME", "start": 3, "end": 7}])},
            {"id": "2A", "source_text": "Bonjour", "language": "French", "privacy_mask": "[]"}]
    def respond(url, timeout):
        assert "/revision123/data/train/" in url
        return io.BytesIO(("\n".join(json.dumps(r) for r in rows) + "\n").encode())
    monkeypatch.setattr(prepare_smoke.urllib.request, "urlopen", respond)
    output = tmp_path / "smoke"
    prepare_smoke.prepare(output, 1, 2, 7, 512)
    row = json.loads((output / "inputs.jsonl").read_text())
    assert row["id"] == "1A" and row["split"] == "train"
    assert row["gold_entities"][0]["start"] == 3
    assert json.loads((output / "labels.json").read_text()) == ["GIVENNAME"]


def test_frozen_taxonomy_format_is_accepted_without_inference(tmp_path, monkeypatch, capsys):
    source = tmp_path / "inputs.jsonl"
    source.write_text(json.dumps({"id": "frozen-1", "source_text": "Hello", "entities": []}) + "\n")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"canonical": ["EMAIL"], "raw_to_canonical": {"EMAIL": "EMAIL"}}))
    monkeypatch.setattr(sys, "argv", ["run_models", "--input", str(source), "--labels", str(labels),
                                     "--model", "test-model", "--dry-run"])
    monkeypatch.setattr(run_models, "extract", lambda *a, **k: pytest.fail("Dry run must not make inference calls"))
    run_models.main()
    assert "1 rows × 1 models = 1 requests" in capsys.readouterr().out
