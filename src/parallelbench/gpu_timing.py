"""CUDA event instrumentation for the existing PyTorch backend."""
from .backends import CUDABackend


class MeasuredTorchCUDA(CUDABackend):
    def __init__(self, torch, a, b):
        super().__init__(torch, a, b)
        self.metadata = dict(input_dtype='float32', multiplication_dtype='float32',
            accumulation_dtype='float32', output_dtype='float32',
            compute_contract='PyTorch FP32 with TF32 disabled',
            host_buffer_bytes=a.nbytes + b.nbytes + self.host_output.numel() * 4,
            device_buffer_bytes=(self.a.numel() + self.b.numel() + self.output.numel()) * 4,
            memory_scope='owned buffers; excludes runtime/library workspaces')

    def event_samples(self, repetitions):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()
        samples = []
        for _ in range(repetitions):
            self.synchronize()
            start.record()
            self.compute()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return samples
