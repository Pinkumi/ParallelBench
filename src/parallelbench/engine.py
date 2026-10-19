"""Benchmark orchestration, independent trials, and descriptive statistics."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import random
import numpy as np
from .backends import CPUBackend, CUDABackend, calibrate_cpu, discover_cuda, load_torch, measure
from .hardware import collect_environment, cpu_thread_context


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
    cpu_threads: int | None = None
    trials: int = 1
    randomize_order: bool = False
    min_cpu_sample_ms: float = 5.0
    instability_threshold: float = 0.10

    def __post_init__(self):
        if not self.sizes or any(type(n) is not int or n <= 0 for n in self.sizes):
            raise ValueError("Matrix sizes must be positive integers")
        if type(self.warmup) is not int or self.warmup < 1:
            raise ValueError("At least one warm-up iteration is required")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("Repetitions must be positive")
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("Trials must be positive")
        if self.cpu_threads is not None and (type(self.cpu_threads) is not int or self.cpu_threads < 1):
            raise ValueError("CPU threads must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if any(not math.isfinite(x) or x < 0 for x in (self.rtol, self.atol, self.min_cpu_sample_ms)):
            raise ValueError("Tolerances and minimum CPU sample time must be finite and nonnegative")
        if not math.isfinite(self.instability_threshold) or self.instability_threshold <= 0:
            raise ValueError("Instability threshold must be finite and positive")
        if type(self.randomize_order) is not bool:
            raise ValueError("Randomize order must be a boolean")
        if self.device not in ("auto", "cpu", "cuda") or self.precision != "float32":
            raise ValueError("Supported devices: auto, cpu, cuda; precision: float32")


def summarize(samples):
    values = np.asarray(samples)
    median = float(np.median(values))
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"median_ms": median, "mean_ms": mean, "std_ms": std,
            "min_ms": float(np.min(values)), "max_ms": float(np.max(values)),
            "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "mad_ms": float(np.median(np.abs(values - median))),
            "cv": std / mean if mean else 0.0, "samples_ms": samples}


def validate(actual, reference, config):
    np.testing.assert_allclose(actual, reference, rtol=config.rtol, atol=config.atol)


def _measure_backend(backend, functions, config, trial, size, name):
    rows = []
    sync = backend.synchronize if name == "cuda" else lambda: None
    for operation, function in functions:
        for _ in range(config.warmup):
            function()
            sync()
        batch_size = calibrate_cpu(function, config.min_cpu_sample_ms) if name == "cpu" else 1
        elapsed = measure(function, config.repetitions, sync, iterations_per_sample=batch_size)
        samples = [value / batch_size for value in elapsed]
        stats = summarize(samples)
        rows.append({"trial": trial, "size": size, "backend": name,
                     "operation": operation, "valid": True,
                     "iterations_per_sample": batch_size, "elapsed_samples_ms": elapsed,
                     "unstable": stats["cv"] > config.instability_threshold, **stats})
    return rows


def _attach_speedups(rows):
    for size in dict.fromkeys(row["size"] for row in rows):
        group = [row for row in rows if row["size"] == size]
        cpu_ms = next(row["median_ms"] for row in group if row["backend"] == "cpu")
        gpu = {row["operation"]: row["median_ms"] for row in group if row["backend"] == "cuda"}
        compute = cpu_ms / gpu["compute"] if gpu.get("compute", 0) > 0 else None
        end_to_end = cpu_ms / gpu["end_to_end"] if gpu.get("end_to_end", 0) > 0 else None
        for row in group:
            row["compute_speedup"] = 1.0 if row["backend"] == "cpu" else compute
            row["end_to_end_speedup"] = 1.0 if row["backend"] == "cpu" else end_to_end
            row["speedup"] = (row["compute_speedup"] if row["operation"] == "compute"
                              else row["end_to_end_speedup"] if row["operation"] == "end_to_end"
                              else None)


def _aggregate(trial_rows, config):
    rows, warnings = [], []
    keys = dict.fromkeys((r["size"], r["backend"], r["operation"]) for r in trial_rows)
    for size, backend, operation in keys:
        trials = [r for r in trial_rows if (r["size"], r["backend"], r["operation"]) ==
                  (size, backend, operation)]
        samples = [sample for row in trials for sample in row["samples_ms"]]
        stats = summarize(samples)
        trial_cv = summarize([r["median_ms"] for r in trials])["cv"]
        max_trial_cv = max(r["cv"] for r in trials)
        unstable = max(stats["cv"], trial_cv, max_trial_cv) > config.instability_threshold
        row = {"size": size, "backend": backend, "operation": operation,
               "valid": True, "trial_count": len(trials), "trial_median_cv": trial_cv,
               "max_trial_cv": max_trial_cv, "unstable": unstable, **stats}
        rows.append(row)
        if unstable:
            warnings.append(f"Unstable measurements: n={size} {backend} {operation}; "
                            f"sample CV={stats['cv']:.1%}, trial-median CV={trial_cv:.1%}, "
                            f"maximum within-trial CV={max_trial_cv:.1%} "
                            f"(threshold={config.instability_threshold:.1%}). Raw samples retained.")
    _attach_speedups(rows)
    return rows, warnings


def _run_trials(config, torch, pytorch, reason):
    environment = collect_environment(torch, pytorch=pytorch)
    trial_rows, execution_order = [], []
    order_rng = random.Random(config.seed)
    for trial in range(config.trials):
        current_rows = []
        sizes = list(config.sizes)
        if config.randomize_order:
            order_rng.shuffle(sizes)
        for size in sizes:
            # Trial zero preserves Phase 1 inputs. Later trials use fresh deterministic inputs.
            seed = [config.seed, size] if trial == 0 else [config.seed, size, trial]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            a = rng.uniform(-1, 1, (size, size)).astype(np.float32)
            b = rng.uniform(-1, 1, (size, size)).astype(np.float32)
            reference = a.astype(np.float64) @ b.astype(np.float64)
            cpu = CPUBackend(a, b)
            gpu = CUDABackend(torch, a, b) if torch is not None else None
            names = ["cpu", "cuda"] if gpu is not None else ["cpu"]
            if config.randomize_order:
                order_rng.shuffle(names)
            for name in names:
                execution_order.append({"trial": trial + 1, "size": size, "backend": name})
                if name == "cpu":
                    current_rows.extend(_measure_backend(cpu, [("compute", cpu.compute)],
                                                         config, trial + 1, size, name))
                    validate(cpu.output, reference, config)
                else:
                    # Prime GEMM before timing even the standalone output transfer.
                    gpu.end_to_end()
                    gpu.synchronize()
                    functions = [("host_to_device", gpu.upload), ("compute", gpu.compute),
                                 ("device_to_host", gpu.download), ("end_to_end", gpu.end_to_end)]
                    current_rows.extend(_measure_backend(gpu, functions, config, trial + 1, size, name))
                    del functions
                    validate(gpu.download(), reference, config)
            if gpu is not None:
                validate(gpu.host_output.numpy(), cpu.output, config)
            # Release buffers before allocating the next size (important on small GPUs).
            del cpu, gpu, a, b, reference
        _attach_speedups(current_rows)
        trial_rows.extend(current_rows)
    rows, warnings = _aggregate(trial_rows, config)
    if config.repetitions < 2 or config.trials < 2:
        warnings.append("Limited variability assessment: use at least two repetitions and two trials.")
    return {"schema_version": 2, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "settings": asdict(config), "environment": environment,
            "cuda_skip_reason": reason, "execution_order": execution_order,
            "warnings": warnings, "results": rows, "trial_results": trial_rows}


def run(config):
    torch, reason = (None, "CPU explicitly requested") if config.device == "cpu" else discover_cuda()
    if config.device == "cuda" and torch is None:
        raise RuntimeError(reason)
    pytorch = torch if torch is not None else load_torch()[0]
    old_tf32 = None
    if torch is not None:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with cpu_thread_context(pytorch, config.cpu_threads):
            return _run_trials(config, torch, pytorch, reason)
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
