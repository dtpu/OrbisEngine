import io
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "stages"))
import vlm_judge


@pytest.fixture
def budget(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "schema": "wander.vlm-budget/1",
                "maxCalls": 2,
                "maxCompletionTokens": 4096,
                "allowedModels": ["gpt-6-astra"],
                "calls": [],
            }
        )
    )
    monkeypatch.setenv("WANDER_VLM_BUDGET", str(path))
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    return path


def response(content='{"ok": true}'):
    return io.BytesIO(
        json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "private": "secret-test-key",
                },
            }
        ).encode()
    )


def ask(prompt="private prompt", **kwargs):
    return vlm_judge.ask_images("gpt-6-astra", [], prompt, **kwargs)


def test_success_quota_duplicate_and_token_bound(budget, monkeypatch):
    network = Mock(side_effect=lambda *a, **kw: response())
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert ask() == {"ok": True}
    assert "duplicate" in ask()["error"]
    assert ask("second prompt") == {"ok": True}
    assert "exhausted" in ask("third prompt")["error"]
    assert network.call_count == 2
    body = json.loads(network.call_args.args[0].data)
    assert body["max_completion_tokens"] == 4096
    data = json.loads(budget.read_text())
    assert data["calls"][0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    assert all(c["status"] == "completed" for c in data["calls"])
    assert "secret-test-key" not in budget.read_text()
    assert "private prompt" not in budget.read_text()


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, ValueError])
def test_failed_or_ambiguous_request_consumes_claim(budget, monkeypatch, failure):
    def fail(*args, **kwargs):
        assert json.loads(budget.read_text())["calls"][0]["status"] == "claimed"
        raise failure("secret-test-key private prompt")

    network = Mock(side_effect=fail)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "error" in ask(retries=20)
    assert "duplicate" in ask()["error"]
    assert network.call_count == 1
    assert json.loads(budget.read_text())["calls"][0]["status"] == "error"
    assert "secret-test-key" not in budget.read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "wrong"},
        {"maxCalls": 0},
        {"maxCalls": True},
        {"maxCompletionTokens": -1},
        {"allowedModels": ["other"]},
        {"calls": [{}]},
    ],
)
def test_bad_config_denies_network(budget, monkeypatch, change):
    data = json.loads(budget.read_text())
    data.update(change)
    budget.write_text(json.dumps(data))
    network = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "error" in ask()
    network.assert_not_called()


def test_missing_or_empty_config_denies_network(budget, monkeypatch):
    budget.unlink()
    network = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "error" in ask()
    monkeypatch.setenv("WANDER_VLM_BUDGET", "")
    assert "error" in ask()
    network.assert_not_called()


def test_parallel_calls_share_allowance(budget, monkeypatch):
    network = Mock(side_effect=lambda *a, **kw: response())
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: ask(str(i)), range(8)))
    assert sum(r == {"ok": True} for r in results) == 2
    assert network.call_count == 2
    assert len(json.loads(budget.read_text())["calls"]) == 2


def test_unconfigured_retains_retries(budget, monkeypatch):
    monkeypatch.delenv("WANDER_VLM_BUDGET")
    network = Mock(side_effect=[TimeoutError(), response()])
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    monkeypatch.setattr(vlm_judge.time, "sleep", lambda _: None)
    assert ask() == {"ok": True}
    assert network.call_count == 2
    assert "max_completion_tokens" not in json.loads(network.call_args.args[0].data)


def test_configured_bounds_and_model_are_used(budget, monkeypatch):
    data = json.loads(budget.read_text())
    data.update(maxCalls=1, maxCompletionTokens=123, allowedModels=["configured-model"])
    budget.write_text(json.dumps(data))
    network = Mock(side_effect=lambda *a, **kw: response())
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "not allowed" in ask()["error"]
    assert vlm_judge.ask_images("configured-model", [], "prompt") == {"ok": True}
    assert "exhausted" in vlm_judge.ask_images("configured-model", [], "other")["error"]
    assert network.call_count == 1
    assert json.loads(network.call_args.args[0].data)["max_completion_tokens"] == 123


def test_malformed_result_keeps_usage_and_consumes_claim(budget, monkeypatch):
    network = Mock(side_effect=lambda *a, **kw: response("invalid json"))
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "error" in ask()
    assert "duplicate" in ask()["error"]
    call = json.loads(budget.read_text())["calls"][0]
    assert call["status"] == "error"
    assert call["usage"]["total_tokens"] == 30
    assert network.call_count == 1


def test_claim_write_failure_prevents_network(budget, monkeypatch):
    monkeypatch.setattr(vlm_judge.vlm_budget, "_save", Mock(side_effect=OSError("disk")))
    network = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    assert "error" in ask()
    network.assert_not_called()


class ErrorBody(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


@pytest.mark.parametrize(
    ("status", "code", "param"),
    [
        (401, "invalid_api_key", None),
        (429, "insufficient_quota", None),
        (429, "credit_balance_exhausted", None),
        (429, "organization_spend_limit_exceeded", None),
        (429, "project_spend_limit_exceeded", None),
        (429, "organization_usage_limit_exceeded", None),
        (429, "slow_down", None),
        (400, "unsupported_parameter", "max_completion_tokens"),
    ],
)
def test_http_error_keeps_only_safe_diagnostics_without_retry(
    budget, monkeypatch, status, code, param
):
    raw = json.dumps(
        {
            "error": {
                "code": code,
                "param": param,
                "message": "secret-test-key private prompt private image",
                "extra": "private provider body",
            }
        }
    ).encode()
    body = ErrorBody(raw)
    error = vlm_judge.urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions",
        status,
        "private error reason",
        {"private-header": "secret-test-key"},
        body,
    )
    network = Mock(side_effect=error)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    result = ask(retries=20)
    expected = {"httpStatus": status, "providerCode": code}
    if param:
        expected["providerParam"] = param
    assert {key: value for key, value in result.items() if key != "errorGuidance"} == {
        "error": "bounded VLM request stopped: HTTPError",
        **expected,
    }
    assert isinstance(result["errorGuidance"], str)
    call = json.loads(budget.read_text())["calls"][0]
    assert call["status"] == "error"
    assert call["errorType"] == "HTTPError"
    assert {key: call[key] for key in expected} == expected
    assert body.read_sizes == [16 * 1024]
    assert body.closed
    assert "duplicate" in ask()["error"]
    assert network.call_count == 1
    combined = budget.read_text() + json.dumps(result)
    for secret in [
        "secret-test-key",
        "private prompt",
        "private image",
        "private provider body",
        "private error reason",
        "private-header",
    ]:
        assert secret not in combined


@pytest.mark.parametrize(
    "payload",
    [
        {
            "error": {
                "code": "untrusted-secret-code",
                "param": "private-prompt-param",
                "type": "private-secret-type",
            }
        },
        {"error": {"code": {"secret": "value"}, "param": ["private"]}},
        {"error": "private provider message"},
        ["private provider message"],
    ],
)
def test_unknown_provider_fields_are_omitted(budget, monkeypatch, payload):
    error = vlm_judge.urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions",
        400,
        "private reason",
        {},
        io.BytesIO(json.dumps(payload).encode()),
    )
    network = Mock(side_effect=error)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    result = ask()
    assert {key: value for key, value in result.items() if key != "errorGuidance"} == {
        "error": "bounded VLM request stopped: HTTPError",
        "httpStatus": 400,
    }
    call = json.loads(budget.read_text())["calls"][0]
    assert call["httpStatus"] == 400
    assert "providerCode" not in call
    assert "providerParam" not in call
    assert "providerType" not in call
    assert network.call_count == 1


def test_http_error_body_is_read_once_with_a_finite_limit(budget, monkeypatch):
    body = ErrorBody(b" " * (16 * 1024) + b'{"error":{"code":"insufficient_quota"}}')
    error = vlm_judge.urllib.error.HTTPError("https://api.openai.com", 429, "private", {}, body)
    network = Mock(side_effect=error)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    result = ask(retries=20)
    assert result["error"] == "bounded VLM request stopped: HTTPError"
    assert result["httpStatus"] == 429
    assert "providerCode" not in result
    assert "cause is unknown" in result["errorGuidance"]
    assert body.read_sizes == [16 * 1024]
    assert network.call_count == 1


def test_receipt_revalidates_diagnostics_instead_of_storing_arbitrary_fields(budget):
    receipt = vlm_judge.vlm_budget.claim(budget, {"model": "gpt-6-astra"})
    vlm_judge.vlm_budget.finish(
        receipt,
        "error",
        error=ValueError("private prompt"),
        error_details={
            "httpStatus": True,
            "providerCode": "private-code",
            "providerParam": "private-param",
            "providerType": "private-type",
            "body": "secret-test-key",
            "headers": {"Authorization": "secret-test-key"},
        },
    )
    call = json.loads(budget.read_text())["calls"][0]
    assert call["errorType"] == "ValueError"
    assert (
        not {"httpStatus", "providerCode", "providerParam", "providerType", "body", "headers"}
        & call.keys()
    )
    assert "private" not in budget.read_text()
    assert "secret-test-key" not in budget.read_text()


@pytest.mark.parametrize(
    "code,expected_guidance",
    [
        ("credit_balance_exhausted", "credit balance"),
        ("organization_spend_limit_exceeded", "organization spending limit"),
        ("project_spend_limit_exceeded", "project spending limit"),
        ("organization_usage_limit_exceeded", "organization usage limit"),
        ("slow_down", "Reduce request rate"),
        ("unrecognized-private-code", "cause is unknown"),
    ],
)
def test_429_guidance_preserves_known_causes_without_guessing_unknowns(
    budget, monkeypatch, code, expected_guidance
):
    body = ErrorBody(json.dumps({"error": {"code": code, "message": "secret-test-key"}}).encode())
    error = vlm_judge.urllib.error.HTTPError("https://api.openai.com", 429, "private", {}, body)
    network = Mock(side_effect=error)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    result = ask(retries=20)
    assert expected_guidance in result["errorGuidance"]
    call = json.loads(budget.read_text())["calls"][0]
    if code == "unrecognized-private-code":
        assert "providerCode" not in result and "providerCode" not in call
    else:
        assert result["providerCode"] == call["providerCode"] == code
    assert call["status"] == "error"
    assert len(json.loads(budget.read_text())["calls"]) == 1
    assert "secret-test-key" not in json.dumps(result) + budget.read_text()
    assert body.read_sizes == [16 * 1024]
    assert network.call_count == 1


@pytest.mark.parametrize("kind", ["insufficient_quota", "invalid_request_error"])
def test_known_error_type_is_retained_separately_when_code_is_missing(budget, monkeypatch, kind):
    error = vlm_judge.urllib.error.HTTPError(
        "https://api.openai.com",
        429,
        "private",
        {},
        io.BytesIO(
            json.dumps(
                {"error": {"type": kind, "code": None, "message": "secret-test-key"}}
            ).encode()
        ),
    )
    network = Mock(side_effect=error)
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    result = ask()
    call = json.loads(budget.read_text())["calls"][0]
    assert result["providerType"] == call["providerType"] == kind
    assert "providerCode" not in result and "providerCode" not in call
    assert network.call_count == 1
    assert "secret-test-key" not in json.dumps(result) + budget.read_text()


@pytest.mark.parametrize(
    "code",
    [
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "unknown-private-code",
        None,
    ],
)
def test_unbounded_billing_or_unknown_429_stops_without_retry(budget, monkeypatch, code):
    monkeypatch.delenv("WANDER_VLM_BUDGET")
    body = ErrorBody(json.dumps({"error": {"code": code, "message": "secret-test-key"}}).encode())
    error = vlm_judge.urllib.error.HTTPError("https://api.openai.com", 429, "private", {}, body)
    network = Mock(side_effect=[error, response()])
    sleep = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    monkeypatch.setattr(vlm_judge.time, "sleep", sleep)
    result = ask(retries=20)
    assert result["error"] == "VLM request stopped: HTTPError"
    assert result["httpStatus"] == 429
    assert network.call_count == 1
    sleep.assert_not_called()
    assert body.read_sizes == [16 * 1024]
    assert "secret-test-key" not in json.dumps(result)
    assert json.loads(budget.read_text())["calls"] == []


@pytest.mark.parametrize("code", ["rate_limit_exceeded", "slow_down"])
def test_unbounded_recognized_rate_limit_retains_finite_retry_and_retry_after(
    budget, monkeypatch, code
):
    monkeypatch.delenv("WANDER_VLM_BUDGET")
    error = vlm_judge.urllib.error.HTTPError(
        "https://api.openai.com",
        429,
        "private",
        {"Retry-After": "20"},
        io.BytesIO(json.dumps({"error": {"code": code}}).encode()),
    )
    network = Mock(side_effect=[error, response()])
    sleep = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    monkeypatch.setattr(vlm_judge.time, "sleep", sleep)
    assert ask(retries=2) == {"ok": True}
    assert network.call_count == 2
    sleep.assert_called_once_with(20)


def test_unbounded_recognized_rate_limit_stops_at_caller_retry_allowance(budget, monkeypatch):
    monkeypatch.delenv("WANDER_VLM_BUDGET")

    def fail(*args, **kwargs):
        return_error = vlm_judge.urllib.error.HTTPError(
            "https://api.openai.com",
            429,
            "secret-test-key",
            {},
            io.BytesIO(b'{"error":{"code":"slow_down","message":"private prompt"}}'),
        )
        raise return_error

    network = Mock(side_effect=fail)
    sleep = Mock()
    monkeypatch.setattr(vlm_judge.urllib.request, "urlopen", network)
    monkeypatch.setattr(vlm_judge.time, "sleep", sleep)
    result = ask(retries=2)
    assert result["error"] == "VLM request stopped: HTTPError"
    assert result["providerCode"] == "slow_down"
    assert "Reduce request rate" in result["errorGuidance"]
    assert network.call_count == 2
    sleep.assert_called_once_with(15)
    assert "private prompt" not in json.dumps(result)
    assert "secret-test-key" not in json.dumps(result)
