"""Direct OpenAI Responses API adapter. No requests are routed to Fireworks."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from fireworks_client import extract as shared_extract, load_env

ENDPOINT = "https://api.openai.com/v1/responses"


class OpenAIError(RuntimeError):
    """Error text excludes provider bodies, request text and credentials."""


def model_id():
    load_env()
    value = os.environ.get("OPENAI_MODEL", "").strip()
    if not value:
        raise OpenAIError("Set OPENAI_MODEL in .env or pass --model.")
    return value


def complete(messages, *, model=None, max_tokens=None, json_mode=False, schema=None):
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise OpenAIError("Set OPENAI_API_KEY in .env.")
    model = model or model_id()
    body = {"model": model, "input": messages, "store": False, "stream": False,
            "max_output_tokens": max_tokens or int(os.environ.get("OPENAI_MAX_TOKENS", "2048"))}
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if effort:
        body["reasoning"] = {"effort": effort}
    temperature = os.environ.get("OPENAI_TEMPERATURE", "").strip()
    if temperature:
        body["temperature"] = float(temperature)
    if schema is not None:
        body["text"] = {"format": {"type": "json_schema", "name": "pii_entities", "strict": True, "schema": schema}}
    elif json_mode:
        body["text"] = {"format": {"type": "json_object"}}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("OPENAI_TIMEOUT", "30"))) as response:
            payload = json.loads(response.read())
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as exc:
        raise OpenAIError(f"OpenAI HTTP {exc.code}; check model access, supported settings, key or quota.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise OpenAIError("OpenAI network request failed or timed out.") from None
    except (ValueError, UnicodeDecodeError):
        raise OpenAIError("OpenAI returned invalid JSON.") from None
    latency_ms = (time.perf_counter() - started) * 1000
    if not isinstance(payload, dict) or not isinstance(payload.get("output"), list):
        raise OpenAIError("OpenAI response did not contain an output list.")
    parts, refused = [], False
    for item in payload["output"]:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content["text"])
            elif content.get("type") == "refusal":
                refused = True
    status = payload.get("status")
    if refused:
        finish = "refusal"
    elif status == "completed" and parts:
        finish = "stop"
    elif status == "incomplete":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        finish = "length" if reason == "max_output_tokens" else "incomplete"
    else:
        finish = status if status != "completed" else "empty_output"
    provider_usage = payload.get("usage") or {}
    # Translate field names, not token counts. Missing usage remains unknown.
    usage = {}
    for source, target in [("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                           ("total_tokens", "total_tokens"), ("input_tokens_details", "prompt_tokens_details"),
                           ("output_tokens_details", "completion_tokens_details")]:
        if source in provider_usage:
            usage[target] = provider_usage[source]
    return {"raw_output": "".join(parts), "usage": usage, "provider_usage": provider_usage,
            "latency_ms": latency_ms, "finish_reason": finish, "request_id": request_id,
            "response_model": payload.get("model"), "response_id": payload.get("id"),
            "raw_response": payload,
            "request_parameters": {k: v for k, v in body.items() if k not in {"input", "model"}}}


def extract(text, labels, *, model=None):
    # Reuse the exact baseline prompt, schema and parser; only transport changes.
    return shared_extract(text, labels, model=model, completion_fn=complete,
                          output_mode="json_schema")
