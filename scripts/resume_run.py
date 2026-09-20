#!/usr/bin/env python3
"""Start a fresh workflow for a run that already exists, resuming from its rows.

A run's progress lives in the database, so losing its scheduler is not losing the run. This
terminates whatever workflow is attached to a run and starts a new one on current code; the
workflow's first act is to load the stored state, so stages already finished stay finished.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from temporalio.client import Client

from orchestrator.database import RunRecord
from orchestrator.workflows.run import GenerationWorkflow, GenerationWorkflowInput


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="runs to resume")
    parser.add_argument("--address", default=os.environ.get("WANDER_TEMPORAL_ADDRESS"))
    parser.add_argument(
        "--queue", default=os.environ.get("WANDER_CONTROL_QUEUE", "pipeline-control")
    )
    args = parser.parse_args()

    database_url = os.environ["WANDER_DATABASE_URL"].replace(
        "postgresql+asyncpg://", "postgresql+psycopg://"
    )
    sessions = sessionmaker(create_engine(database_url), expire_on_commit=False)
    client = await Client.connect(args.address)

    for run_id in args.run_ids:
        with sessions() as session:
            run = session.scalars(select(RunRecord).where(RunRecord.id == run_id)).one_or_none()
            if run is None:
                print(f"{run_id}: no such run", file=sys.stderr)
                return 1
            options = (run.configuration or {}).get("options", {})
            source_artifact_id = run.source_artifact_id
            source_sha256 = run.source_sha256
            parent_run_id = run.parent_run_id
            branch_key = run.branch_key
        try:
            handle = client.get_workflow_handle(run_id)
            await handle.terminate(reason="resuming from stored run state")
            print(f"{run_id}: terminated previous workflow")
        except Exception as error:  # noqa: BLE001 - absent or already closed is fine
            print(f"{run_id}: no running workflow to terminate ({type(error).__name__})")
        await client.start_workflow(
            GenerationWorkflow.run,
            GenerationWorkflowInput(
                run_id=run_id,
                source_sha256=source_sha256,
                source_artifact_id=source_artifact_id,
                options=options,
                parent_run_id=parent_run_id,
                branch_key=branch_key,
            ),
            id=run_id,
            task_queue=args.queue,
        )
        print(f"{run_id}: resumed  options={json.dumps(options, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
