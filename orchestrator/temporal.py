"""Temporal client adapter used by the API and compatibility CLI."""

from __future__ import annotations

from typing import Any

from temporalio.client import Client

from orchestrator.workflows.run import GenerationWorkflow, GenerationWorkflowInput


class TemporalWorkflowControl:
    def __init__(self, client: Client, task_queue: str = "pipeline-control"):
        self.client = client
        self.task_queue = task_queue

    async def start(self, request: GenerationWorkflowInput) -> None:
        await self.client.start_workflow(
            GenerationWorkflow.run,
            request,
            id=request.run_id,
            task_queue=self.task_queue,
        )

    async def signal(self, run_id: str, name: str, value: Any = None) -> None:
        handle = self.client.get_workflow_handle(run_id)
        if value is None:
            await handle.signal(name)
        else:
            await handle.signal(name, value)

    async def state(self, run_id: str) -> dict[str, Any]:
        handle = self.client.get_workflow_handle(run_id)
        return await handle.query(GenerationWorkflow.state)
