"""Portable environment metadata; no CUDA dependency required."""
import contextlib
import io
import os
import platform
from importlib.metadata import PackageNotFoundError, version
import numpy as np
from threadpoolctl import threadpool_info


def collect_environment(torch=None):
    versions = {}
    for package in ("parallelbench", "numpy", "pandas", "matplotlib", "torch", "threadpoolctl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    cpu = platform.processor()
    if not cpu and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            cpu = next((line.split(":", 1)[1].strip() for line in source if line.startswith("model name")), "unknown")
    return {
        "os": platform.platform(), "machine": platform.machine(),
        "cpu": cpu or "unknown", "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(), "libraries": versions,
        "numpy_config": config.getvalue(), "threadpools": threadpool_info(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "gpu": None if torch is None else {
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "cuda_runtime": torch.version.cuda,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }
