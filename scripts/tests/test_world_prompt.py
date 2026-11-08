import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import world_prompt
import vlm_judge


@pytest.mark.parametrize(
    ("model_args", "model", "effort", "response", "description"),
    [
        ([], "gpt-6-astra", "low", {"text_prompt": "A forest path."}, "A forest path."),
        (
            [],
            "gpt-6-astra",
            "low",
            {"error": "bounded VLM request stopped: TimeoutError"},
            None,
        ),
        ([], "gpt-6-astra", "low", [], None),
        (["--model", "gpt-4.1"], "gpt-4.1", None, {"text_prompt": "A room."}, "A room."),
    ],
)
def test_source_description_records_request_settings(
    tmp_path, monkeypatch, model_args, model, effort, response, description
):
    clip = tmp_path / "source.mp4"
    output = tmp_path / "prompt.json"
    frames = [0, 10]
    paths = ["frame_0000.png", "frame_0010.png"]
    sample = Mock(return_value=(frames, paths))
    ask_images = Mock(return_value=response)
    monkeypatch.setattr(world_prompt, "sample", sample)
    monkeypatch.setattr(vlm_judge, "ask_images", ask_images)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setattr(
        sys,
        "argv",
        ["world_prompt.py", "--clip", str(clip), "--out", str(output), *model_args],
    )

    world_prompt.main()

    sample.assert_called_once_with(clip, None, 6, tmp_path / "prompt-frames")
    ask_images.assert_called_once_with(
        model,
        paths,
        world_prompt.BRIEF,
        detail="high",
        retries=1,
        timeout=300,
        reasoning_effort=effort,
    )
    receipt = json.loads(output.read_text())
    assert receipt["model"] == model
    assert receipt["reasoningEffort"] == effort
    assert receipt["timeoutSeconds"] == 300
    assert receipt["frames"] == frames
    assert receipt["framePaths"] == paths
    assert receipt.get("text_prompt") == description
    if description is None:
        assert receipt["error"]
