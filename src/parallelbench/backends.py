"""Allocation-free compute paths and explicit CUDA transfer measurements."""
from time import perf_counter
import numpy as np


class CPUBackend:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.output = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)

    def compute(self):
        np.matmul(self.a, self.b, out=self.output)
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
        self.host_output = torch.empty(self.output.shape, dtype=torch.float32)
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


def measure(operation, repetitions, synchronize=lambda: None):
    samples = []
    for _ in range(repetitions):
        synchronize()
        start = perf_counter()
        operation()
        synchronize()
        samples.append((perf_counter() - start) * 1000)
    return samples


def discover_cuda():
    try:
        import torch
    except ImportError as exc:
        return None, f"PyTorch unavailable: {exc}"
    if not torch.cuda.is_available():
        return None, "CUDA unavailable according to PyTorch"
    return torch, None
