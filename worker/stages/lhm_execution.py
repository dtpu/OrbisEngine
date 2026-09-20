"""Local validation and bounded subprocess execution for saved LHM animation."""

import math
import subprocess
import time
from pathlib import Path


def animation_options(fixed_world_scale=None, execution_timeout=0):
    if fixed_world_scale is not None and (
        isinstance(fixed_world_scale, bool)
        or not math.isfinite(fixed_world_scale)
        or not 0 < fixed_world_scale < 10
    ):
        raise ValueError("Fixed world scale must be finite and strictly between 0 and 10")
    if type(execution_timeout) is not int or not 0 <= execution_timeout <= 3600:
        raise ValueError("Execution timeout must be an integer from 1 to 3600, or 0 for default")
    flags = [] if fixed_world_scale is None else ["--fixed-world-scale", str(fixed_world_scale)]
    options = (
        {
            "timeout": execution_timeout,
            "cpu": (4, 4),
            "memory": (65536, 65536),
            "retries": 0,
            "max_containers": 1,
        }
        if execution_timeout
        else {}
    )
    return flags, options


def run_logged_inference(command, log_path, *, started, execution_timeout=0, cwd=None):
    """Leave finalization time before the provider deadline and retain partial logs/files.

    subprocess.run kills and waits for its own child on timeout. Consequently no child
    continues writing output files while the caller hashes and commits the checkpoint.
    """
    remaining = None
    if execution_timeout:
        reserve = min(60.0, execution_timeout * 0.1)
        remaining = execution_timeout - reserve - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Animation setup exhausted the bounded inference deadline")
    with Path(log_path).open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=True, cwd=cwd
        )
    with Path(log_path).open() as log:
        for line in log:
            print(line, end="", flush=True)


def link_model_caches(repo=Path("/opt/lhm"), cache_root=Path("/cache"), gfpgan_package=None):
    """Expose only the staged public model trees; refuse conflicting runtime directories."""
    repo, cache_root = Path(repo), Path(cache_root)
    links = []
    for relative, name in (
        ("data/pretrained_models", "pretrained_models"),
        ("data/gfpgan", "gfpgan"),
    ):
        source, target = cache_root / relative, repo / name
        if not source.is_dir() or not source.resolve().is_relative_to(cache_root.resolve()):
            raise RuntimeError(f"Stage the public model cache before inference: {relative}")
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Runtime model path conflicts with the staged cache: {name}")
        else:
            target.symlink_to(source, target_is_directory=True)
        links.append({"runtimePath": str(target), "cachePath": str(source)})
    if gfpgan_package is None:
        import importlib.util

        package = importlib.util.find_spec("gfpgan")
        if package is None or package.origin is None:
            raise RuntimeError("GFPGAN must be installed in the native worker image")
        gfpgan_package = Path(package.origin).parent
    source = cache_root / "gfpgan/weights/GFPGANv1.3.pth"
    target = Path(gfpgan_package) / "weights/GFPGANv1.3.pth"
    if not source.is_file() or not source.resolve().is_relative_to(cache_root.resolve()):
        raise RuntimeError("Stage the public GFPGANv1.3 checkpoint before inference")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError("GFPGAN package weight conflicts with the staged cache")
    else:
        target.symlink_to(source)
    links.append({"runtimePath": str(target), "cachePath": str(source)})
    return links
