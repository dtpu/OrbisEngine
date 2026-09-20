"""Compatibility client for submitting a source and following durable run events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"succeeded", "failed", "blocked", "canceled"}


class PipelineClient:
    def __init__(self, endpoint: str, token: str, timeout: float = 120):
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout = timeout

    def submit(
        self,
        source: Path,
        *,
        run_id: str | None,
        options: dict[str, Any],
        budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = Path(source)
        if not source.is_file():
            raise ValueError(f"source clip does not exist: {source}")
        data = {
            "options": json.dumps(options, separators=(",", ":")),
            "budget": json.dumps(budget or {}, separators=(",", ":")),
        }
        if run_id:
            data["requested_id"] = run_id
        with source.open("rb") as stream:
            response = httpx.post(
                f"{self.endpoint}/api/pipeline/runs/upload",
                headers=self.headers,
                data=data,
                files={"source": (source.name, stream, "video/mp4")},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def events(self, run_id: str, after: int = 0):
        headers = {**self.headers, "Accept": "text/event-stream"}
        if after:
            headers["Last-Event-ID"] = str(after)
        with httpx.stream(
            "GET",
            f"{self.endpoint}/api/pipeline/runs/{run_id}/events",
            headers=headers,
            timeout=None,
        ) as response:
            response.raise_for_status()
            event: dict[str, Any] = {"data": []}
            for line in response.iter_lines():
                if not line:
                    if event["data"]:
                        yield {
                            "id": int(event["id"]) if event.get("id") else None,
                            "event": event.get("event", "message"),
                            "data": json.loads("\n".join(event["data"])),
                        }
                    event = {"data": []}
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                value = value.removeprefix(" ")
                if field == "data":
                    event["data"].append(value)
                else:
                    event[field] = value

    def follow(self, run_id: str, output=print) -> str:
        cursor = 0
        for event in self.events(run_id, cursor):
            if event["id"]:
                cursor = event["id"]
            data = event["data"]
            payload = data.get("payload", {})
            subject = data.get("subject_id") or data.get("subjectId") or run_id
            status = payload.get("status")
            output(
                f"[{event['id']}] {event['event']} {subject}" + (f": {status}" if status else "")
            )
            if event["event"] == "run" and status in TERMINAL:
                return status
        return "disconnected"


def token_from_environment(name: str = "WANDER_API_TOKEN") -> str:
    token = os.environ.get(name)
    if not token:
        raise ValueError(f"{name} is required; do not pass API tokens on the command line")
    return token
