import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline


def test_submission_links_to_the_unified_dashboard(monkeypatch, capsys):
    client = Mock()
    client.submit.return_value = {"id": "accepted-run", "sourceSha256": "source-digest"}
    monkeypatch.setattr(run_pipeline, "PipelineClient", Mock(return_value=client))
    monkeypatch.setenv("WANDER_API_TOKEN", "test-placeholder")

    assert run_pipeline.main(["--clip", "source.mp4", "--no-follow"]) == 0
    assert "dashboard: http://127.0.0.1:5399/runs/accepted-run" in capsys.readouterr().out
