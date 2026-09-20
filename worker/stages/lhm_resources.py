"""Bounded native LHM resources and estimated Modal compute rates."""

GPU_USD_PER_SECOND = {"L4": 0.000222, "H100": 0.001097}
CPU_USD_PER_SECOND = 4 * 0.0000131
MEMORY_USD_PER_SECOND = 64 * 0.00000222


def execution_options(gpu="L4", execution_timeout=1800):
    if gpu not in GPU_USD_PER_SECOND:
        raise ValueError("LHM GPU must be L4 or H100")
    if type(execution_timeout) is not int or not 1 <= execution_timeout <= 1800:
        raise ValueError("LHM execution timeout must be an integer from 1 to 1800")
    return {
        "gpu": gpu,
        "timeout": execution_timeout,
        "cpu": (4, 4),
        "memory": (65536, 65536),
        "retries": 0,
        "max_containers": 1,
    }


def compute_rate(gpu):
    execution_options(gpu)
    return GPU_USD_PER_SECOND[gpu] + CPU_USD_PER_SECOND + MEMORY_USD_PER_SECOND
