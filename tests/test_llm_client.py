import io
import json
import urllib.error

from llm.client import RealLLMClient


class FakeResponse:
    def __init__(self, body):
        self.body = io.BytesIO(json.dumps(body).encode())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return self.body.read(*args)


def test_deepseek_json_mode(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = RealLLMClient()
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        return FakeResponse({"choices": [{"message": {"content": '{"tasks":[]}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert client.complete_json("system", "user") == {"tasks": []}
    assert len(calls) == 1
    assert calls[0].full_url == "https://api.deepseek.com/chat/completions"
    payload = json.loads(calls[0].data)
    assert payload["model"] == "deepseek-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 2048


def test_auth_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = RealLLMClient()
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = client.complete_json("system", "user")
    assert result["error"] == "LLM_ERROR"
    assert "401" in result["message"]
    assert len(calls) == 1


def test_length_error_reports_token_budget(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = RealLLMClient()

    def urlopen(request, **kwargs):
        return FakeResponse({
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 1200},
        })

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = client.complete_json("system", "user", max_tokens=1200)
    assert result["error"] == "LLM_ERROR"
    assert "completion_tokens=1200" in result["message"]
