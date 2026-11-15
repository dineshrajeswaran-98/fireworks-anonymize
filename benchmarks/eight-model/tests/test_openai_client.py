"""Direct OpenAI routing and shared evaluation contract; no API calls."""
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import openai_client as client
import fireworks_client
import run_models


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(client, "load_env", lambda: None)
    monkeypatch.setattr(fireworks_client, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-fireworks-key")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "")


def mock_response(monkeypatch, *, status="completed", content=None, usage=True, details=None):
    class Response(io.BytesIO):
        headers = {"x-request-id": "openai-request"}
    def respond(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_header("Authorization") == "Bearer fake-openai-key"
        assert not any(k.lower().startswith("x-") for k in request.headers)
        body = json.loads(request.data)
        assert body["store"] is False and body["stream"] is False
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["reasoning"]["effort"] == "low"
        assert "temperature" not in body
        assert "gold" not in json.dumps(body).lower()
        payload = {"id": "response-1", "model": "test-openai-model", "status": status,
                   "output": [{"type": "reasoning", "summary": []},
                              {"type": "message", "content": content if content is not None else [
                                  {"type": "output_text", "text": '{"entities":[]}'}]}],
                   "incomplete_details": details}
        if usage:
            payload["usage"] = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 80}, "output_tokens_details": {"reasoning_tokens": 10}}
        return Response(json.dumps(payload).encode())
    monkeypatch.setattr(client.urllib.request, "urlopen", respond)


def test_direct_request_shared_contract_and_usage_translation(monkeypatch):
    mock_response(monkeypatch)
    output = client.extract("Hello", ["EMAIL"], model="test-openai-model")
    assert output["entities"] == [] and output["json_valid"] and output["error"] is None
    assert output["usage"]["prompt_tokens"] == 100
    assert output["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    assert output["usage"]["completion_tokens"] == 20
    assert output["usage"]["completion_tokens_details"]["reasoning_tokens"] == 10
    assert output["request_id"] == "openai-request"


@pytest.mark.parametrize("status,content,details", [
    ("incomplete", [{"type": "output_text", "text": '{"entities":[]}'}], {"reason": "max_output_tokens"}),
    ("completed", [{"type": "refusal", "refusal": "Cannot comply"}], None),
    ("completed", [], None),
])
def test_refusal_incomplete_and_empty_output_are_failures(monkeypatch, status, content, details):
    mock_response(monkeypatch, status=status, content=content, details=details)
    assert client.extract("Hello", ["EMAIL"], model="test-openai-model")["error"]


def test_missing_usage_is_not_zero(monkeypatch):
    mock_response(monkeypatch, usage=False)
    assert client.extract("Hello", ["EMAIL"], model="test-openai-model")["usage"] == {}


def test_safe_error_does_not_echo_key_or_provider_body(monkeypatch):
    def fail(*a, **k):
        raise urllib.error.HTTPError(client.ENDPOINT, 401, "fake-openai-key", {}, io.BytesIO(b"private text"))
    monkeypatch.setattr(client.urllib.request, "urlopen", fail)
    with pytest.raises(client.OpenAIError, match="HTTP 401") as exc:
        client.extract("Hello", ["EMAIL"], model="test-model")
    assert "fake-openai-key" not in str(exc.value) and "private text" not in str(exc.value)


def test_harness_provider_choice_does_not_use_fireworks_transport(monkeypatch):
    def fake_extract(text, labels, model):
        return {"raw_output": '{"entities":[]}', "entities": [], "json_valid": True, "error": None}
    monkeypatch.setattr(client, "extract", fake_extract)
    monkeypatch.setattr(run_models, "extract", lambda *a, **k: pytest.fail("Fireworks extractor must not run"))
    output = run_models.run_one({"id": "1", "source_text": "Hello", "entities": ["gold"]}, "model", ["EMAIL"], "openai")
    assert output["provider"] == "openai" and output["error"] is None
    assert "gold" not in json.dumps(output)
