"""Allocation-free compute paths and explicit CUDA transfer measurements."""
import gc
from time import perf_counter_ns
import numpy as np


class CPUBackend:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.output = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)

    def compute(self):
        np.matmul(self.a, self.b, out=self.output)
        return self.output


class TorchCPUBackend:
    """Optional optimized PyTorch CPU comparison using the same host inputs."""

    def __init__(self, torch, a, b):
        self.torch = torch
        self.a, self.b = torch.from_numpy(a), torch.from_numpy(b)
        self.output = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device="cpu")

    def compute(self):
        self.torch.mm(self.a, self.b, out=self.output)
        return self.output


class CUDABackend:
    def __init__(self, torch, a, b):
        self.torch = torch
        self.device = torch.device("cuda")
        # Initialize CUDA and allocate all buffers before timing.
        torch.cuda.init()
        self.host_a, self.host_b = torch.from_numpy(a), torch.from_numpy(b)
        self.a = torch.empty(a.shape, dtype=torch.float32, device=self.device)
        self.b = torch.empty(b.shape, dtype=torch.float32, device=self.device)
        self.output = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=self.device)
        self.host_output = torch.empty(self.output.shape, dtype=torch.float32, device='cpu')
        self.upload()

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def upload(self):
        self.a.copy_(self.host_a)
        self.b.copy_(self.host_b)

    def compute(self):
        self.torch.mm(self.a, self.b, out=self.output)
        return self.output

    def download(self):
        self.host_output.copy_(self.output)
        return self.host_output.numpy()

    def end_to_end(self):
        self.upload()
        self.compute()
        return self.download()


def measure(operation, repetitions, synchronize=lambda: None, *, iterations_per_sample=1):
    """Return raw wall-clock milliseconds per sample (one sample may be a batch)."""
    samples = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            synchronize()
            start = perf_counter_ns()
            for _ in range(iterations_per_sample):
                operation()
            synchronize()
            samples.append((perf_counter_ns() - start) / 1_000_000)
    finally:
        if gc_enabled:
            gc.enable()
    return samples


def calibrate_cpu(operation, minimum_ms):
    """Choose a fixed batch size outside timing samples; cap calibration work."""
    iterations = 1
    if minimum_ms == 0:
        return iterations
    while True:
        elapsed = measure(operation, 1, iterations_per_sample=iterations)[0]
        if elapsed >= minimum_ms or iterations >= 1_048_576:
            return iterations
        iterations = min(iterations * 2, 1_048_576)


def load_torch():
    try:
        import torch
    except ImportError as exc:
        return None, f"PyTorch unavailable: {exc}"
    return torch, None


def discover_cuda():
    torch, reason = load_torch()
    if torch is None:
        return None, reason
    if not torch.cuda.is_available():
        return None, "CUDA unavailable according to PyTorch"
    return torch, None
