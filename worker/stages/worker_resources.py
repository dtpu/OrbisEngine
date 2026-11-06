"""Opt-in resource ceilings for the single-person inference workers, not a spend cap."""

import os
import re

TIMEOUT_ENV = "WANDER_WORKER_TIMEOUT_SECONDS"
LHM_GPU_ENV = "WANDER_LHM_GPU"
LHM_GPU_RATES = {"L4": 0.000222, "L40S": 0.000542}


def lhm_gpu(*, environment=None):
    """One explicit SM8.9 GPU choice; never a fallback list or implicit retry."""
    value = (os.environ if environment is None else environment).get(LHM_GPU_ENV, "L4")
    if not isinstance(value, str) or value not in LHM_GPU_RATES:
        raise ValueError(f"{LHM_GPU_ENV} must be exactly L4 or L40S")
    return value


def lhm_compute_rate(gpu):
    """Requested GPU + four physical CPU cores + 64GiB, in USD per worker second."""
    selected = lhm_gpu(environment={LHM_GPU_ENV: gpu})
    return LHM_GPU_RATES[selected] + 4 * 0.0000131 + 64 * 0.00000222


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
