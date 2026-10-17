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
