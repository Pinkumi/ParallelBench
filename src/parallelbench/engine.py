"""Benchmark orchestration, validation, and descriptive statistics."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import numpy as np
from .backends import CPUBackend, CUDABackend, discover_cuda, measure
from .hardware import collect_environment


@dataclass(frozen=True)
class BenchmarkConfig:
    sizes: tuple[int, ...] = (256, 512, 1024)
    warmup: int = 3
    repetitions: int = 10
    seed: int = 42
    rtol: float = 1e-3
    atol: float = 1e-3
    device: str = "auto"
    precision: str = "float32"

    def __post_init__(self):
        if not self.sizes or any(type(n) is not int or n <= 0 for n in self.sizes):
            raise ValueError("Matrix sizes must be positive integers")
        if type(self.warmup) is not int or self.warmup < 1:
            raise ValueError("At least one warm-up iteration is required")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("Repetitions must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if any(not math.isfinite(x) or x < 0 for x in (self.rtol, self.atol)):
            raise ValueError("Tolerances must be finite and nonnegative")
        if self.device not in ("auto", "cpu", "cuda") or self.precision != "float32":
            raise ValueError("Supported devices: auto, cpu, cuda; precision: float32")


def summarize(samples):
    values = np.asarray(samples)
    median = float(np.median(values))
    return {"median_ms": median, "std_ms": float(np.std(values)),
            "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "mad_ms": float(np.median(np.abs(values - median))), "samples_ms": samples}


def validate(actual, reference, config):
    np.testing.assert_allclose(actual, reference, rtol=config.rtol, atol=config.atol)


def run(config):
    torch, reason = (None, "CPU explicitly requested") if config.device == "cpu" else discover_cuda()
    if config.device == "cuda" and torch is None:
        raise RuntimeError(reason)
    rows = []
    old_tf32 = None
    if torch is not None:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        environment = collect_environment(torch)
        for size in config.sizes:
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, size]))
            a = rng.uniform(-1, 1, (size, size)).astype(np.float32)
            b = rng.uniform(-1, 1, (size, size)).astype(np.float32)
            # Float64 oracle computed outside all measured operations.
            reference = a.astype(np.float64) @ b.astype(np.float64)
            cpu = CPUBackend(a, b)
            for _ in range(config.warmup):
                cpu.compute()
            cpu_stats = summarize(measure(cpu.compute, config.repetitions))
            validate(cpu.output, reference, config)
            rows.append(dict(size=size, backend="cpu", operation="compute", valid=True,
                             speedup=1.0, **cpu_stats))
            if torch is not None:
                gpu = CUDABackend(torch, a, b)
                for _ in range(config.warmup):
                    gpu.upload()
                    gpu.compute()
                    gpu.download()
                    gpu.synchronize()
                for operation, function in (("host_to_device", gpu.upload),
                                            ("compute", gpu.compute),
                                            ("device_to_host", gpu.download)):
                    stats = summarize(measure(function, config.repetitions, gpu.synchronize))
                    rows.append(dict(size=size, backend="cuda", operation=operation, valid=True,
                        speedup=cpu_stats["median_ms"] / stats["median_ms"] if operation == "compute" else None,
                        **stats))
                validate(gpu.download(), reference, config)
                validate(gpu.host_output.numpy(), cpu.output, config)
        return {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "settings": asdict(config), "environment": environment,
                "cuda_skip_reason": reason, "results": rows}
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
