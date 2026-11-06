"""Opt-in resource ceilings for the single-person inference workers, not a spend cap."""

import os
import re

TIMEOUT_ENV = "WANDER_WORKER_TIMEOUT_SECONDS"


def inference_resources(
    *, timeout: int, memory: int, max_containers: int = 1, environment=None
) -> dict:
    """Return existing defaults unless a strict, bounded integer deadline is configured.

    Used at registration, before any provider calls. The environment is forwarded to
    containers because Modal imports the defining module again during startup.
    """
    options = {
        "cpu": 4,
        "memory": memory,
        "timeout": timeout,
        "max_containers": max_containers,
        "retries": 0,
        "scaledown_window": 2,
    }
    value = (os.environ if environment is None else environment).get(TIMEOUT_ENV)
    if value is None:
        return options
    if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= timeout:
        raise ValueError(f"{TIMEOUT_ENV} must be an integer from 1 to {timeout} seconds")
    deadline = int(value)
    options.update(
        cpu=(4, 4),
        memory=(memory, memory),
        timeout=deadline,
        startup_timeout=deadline,
        max_containers=1,
        env={TIMEOUT_ENV: value},
    )
    return options
