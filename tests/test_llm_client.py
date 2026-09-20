import io
import json
import urllib.error

from llm.client import RealLLMClient, LLMServiceUnavailable


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
        return FakeResponse({"choices": [{"message": {"content": '{"tasks":[]}'}, "finish_reason": "stop"}],
                             "usage": {"total_tokens": 37}})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert client.complete_json("system", "user") == {"tasks": []}
    assert len(calls) == 1
    assert client.total_tokens == 37
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


def test_truncated_json_retries_with_larger_budget(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = RealLLMClient()
    budgets = []

    def urlopen(request, **kwargs):
        budgets.append(json.loads(request.data)["max_tokens"])
        if len(budgets) == 1:
            return FakeResponse({
                "choices": [{"message": {"content": '{"instructions":['}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 1024},
            })
        return FakeResponse({
            "choices": [{"message": {"content": '{"instructions":[]}'}, "finish_reason": "stop"}]
        })

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert client.complete_json("system", "user", max_tokens=1024) == {"instructions": []}
    assert budgets == [1024, 2048]


def test_markdown_base_url_is_cleaned_before_request(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "[https://api.deepseek.com/v1](https://api.deepseek.com/v1)")
    client = RealLLMClient()
    assert client.base_url == "https://api.deepseek.com/v1"


def test_invalid_base_url_is_rejected_early(monkeypatch):
    import pytest
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "[https://api.deepseek.com/v1")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        RealLLMClient()


def test_connection_preflight_fails_on_401_without_exposing_key(monkeypatch):
    import pytest
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    client = RealLLMClient()

    def urlopen(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error":"secret-test-key invalid"}'))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(ValueError, match="HTTP 401") as exc:
        client.check_connection()
    assert "secret-test-key" not in str(exc.value)


def test_service_busy_is_retried_and_marked_retryable(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    client = RealLLMClient()
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.HTTPError(request.full_url, 503, "Busy", {}, io.BytesIO(b'{"error":"busy"}'))
        return FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    assert client.complete_json("system", "user") == {"ok": True}
    assert len(calls) == 3

    calls.clear()
    def always_busy(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 503, "Busy", {}, io.BytesIO(b'{"error":"busy"}'))

    monkeypatch.setattr("urllib.request.urlopen", always_busy)
    result = client.complete_json("system", "user")
    assert result["retryable"] is True
    assert len(calls) == 3


def test_preflight_503_reports_temporary_outage(monkeypatch):
    import pytest
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = RealLLMClient()

    def urlopen(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 503, "Busy", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(LLMServiceUnavailable, match="HTTP 503"):
        client.check_connection()
