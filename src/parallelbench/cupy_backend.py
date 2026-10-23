"""Optional CUDA C/NVRTC and cuBLAS backends; no system nvcc is required."""
from importlib.resources import files
import numpy as np


def discover_cupy():
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        return None, f'CuPy unavailable: {exc}'
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None, 'CuPy found no CUDA devices'
    except cp.cuda.runtime.CUDARuntimeError as exc:
        return None, f'CuPy CUDA unavailable: {exc}'
    return cp, None


def validate_block(block, properties, shared_bytes=0):
    if (len(block) != 2 or any(type(x) is not int or x < 1 for x in block) or
            block[0] * block[1] > properties['maxThreadsPerBlock'] or
            any(block[i] > properties['maxThreadsDim'][i] for i in range(2)) or
            shared_bytes > properties['sharedMemPerBlock']):
        raise ValueError(f'Unsupported CUDA block {block} or shared memory {shared_bytes} bytes')


class CuPyBackend:
    """Preallocated pageable host buffers and device buffers shared by all GPU cases."""
    def __init__(self, cp, a, b, *, output_dtype=None, max_memory_fraction=0.5):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('CUDA matmul requires compatible two-dimensional arrays')
        if max(*a.shape, *b.shape) > np.iinfo(np.int32).max:
            raise ValueError('Matrix dimensions exceed CUDA kernel int32 limits')
        self.cp, self.stream = cp, cp.cuda.get_current_stream()
        self.host_a, self.host_b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        dtype = np.dtype(output_dtype or a.dtype)
        shape = (a.shape[0], b.shape[1])
        needed = a.nbytes + b.nbytes + int(np.prod(shape)) * dtype.itemsize
        free, _ = cp.cuda.runtime.memGetInfo()
        if needed > free * max_memory_fraction:
            raise MemoryError(f'GPU buffers need {needed} bytes; exceeds configured fraction of {free} free bytes')
        self.a, self.b = cp.empty(a.shape, dtype=a.dtype), cp.empty(b.shape, dtype=b.dtype)
        self.output = cp.empty(shape, dtype=dtype)
        self.host_output = np.empty(shape, dtype=dtype)
        self.metadata = dict(host_buffer_bytes=self.host_a.nbytes + self.host_b.nbytes + self.host_output.nbytes,
                             device_buffer_bytes=needed, memory_scope='owned buffers; excludes runtime/library workspaces')
        self.upload()

    def synchronize(self):
        self.stream.synchronize()

    def upload(self):
        self.a.set(self.host_a, stream=self.stream)
        self.b.set(self.host_b, stream=self.stream)

    def download(self):
        self.output.get(out=self.host_output, stream=self.stream, blocking=True)
        return self.host_output

    def end_to_end(self):
        self.upload()
        self.compute()
        return self.download()

    def event_samples(self, repetitions):
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        # Initialize event resources outside the samples.
        start.record(self.stream)
        end.record(self.stream)
        end.synchronize()
        samples = []
        for _ in range(repetitions):
            self.synchronize()
            start.record(self.stream)
            self.compute()
            end.record(self.stream)
            end.synchronize()
            samples.append(float(self.cp.cuda.get_elapsed_time(start, end)))
        return samples


class RawMatmulBackend(CuPyBackend):
    def __init__(self, cp, a, b, *, tile=None, block=(16, 16), max_memory_fraction=0.5):
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError('Custom CUDA kernels support float32 only')
        if tile is not None and tile not in (8, 16, 32):
            raise ValueError('Supported CUDA tile sizes: 8, 16, 32')
        block = (tile, tile) if tile is not None else block
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        shared = 2 * tile * tile * 4 if tile is not None else 0
        validate_block(block, properties, shared)
        source = files('parallelbench').joinpath('kernels/matmul.cu').read_text(encoding='utf-8')
        options = ('-std=c++11', f'-DTILE_SIZE={tile or 16}')
        kernel = cp.RawKernel(source, 'matmul_tiled' if tile else 'matmul_naive', options=options)
        try:
            kernel.compile()  # Compilation never enters warm-up, calibration, or timed iterations.
        except cp.cuda.compiler.CompileException as exc:
            raise RuntimeError(f'NVRTC compilation failed: {exc}') from exc
        if block[0] * block[1] > kernel.max_threads_per_block:
            raise ValueError('Requested block exceeds the compiled kernel thread limit')
        super().__init__(cp, a, b, max_memory_fraction=max_memory_fraction)
        self.kernel, self.block = kernel, (*block, 1)
        m, k = a.shape
        n = b.shape[1]
        self.grid = ((n + block[0] - 1) // block[0], (m + block[1] - 1) // block[1], 1)
        self.arguments = (self.a, self.b, self.output, np.int32(m), np.int32(n), np.int32(k))
        self.metadata.update(block=list(self.block), grid=list(self.grid), tile=tile,
                             shared_memory_bytes=shared, registers_per_thread=kernel.num_regs,
                             multiplication_dtype='float32', accumulation_dtype='float32',
                             input_dtype='float32', output_dtype='float32', fused_multiply_add=True,
                             compiler='NVRTC', compiler_options=list(options))

    def compute(self):
        self.kernel(self.grid, self.block, self.arguments, stream=self.stream)
        return self.output


class CuBLASBackend(CuPyBackend):
    """Row-major cuBLAS GEMM with an explicit precision contract."""
    def __init__(self, cp, a, b, *, precision='float32', max_memory_fraction=0.5):
        from cupy.cuda import cublas
        contracts = {
            'float16': ('float16', 'float16', 'float16', cublas.CUBLAS_COMPUTE_16F_PEDANTIC, 2),
            'float32': ('float32', 'float32', 'float32', cublas.CUBLAS_COMPUTE_32F_PEDANTIC, 0),
            'float64': ('float64', 'float64', 'float64', cublas.CUBLAS_COMPUTE_64F_PEDANTIC, 1),
            'mixed': ('float16', 'float32', 'float32', cublas.CUBLAS_COMPUTE_32F_PEDANTIC, 2),
        }
        if precision not in contracts:
            raise ValueError('Unknown cuBLAS precision')
        storage, accumulation, output, self.compute_type, self.input_type = contracts[precision]
        a, b = a.astype(storage), b.astype(storage)
        self.output_type = {'float16': 2, 'float32': 0, 'float64': 1}[output]
        super().__init__(cp, a, b, output_dtype=output, max_memory_fraction=max_memory_fraction)
        self.cublas, self.handle = cublas, cp.cuda.device.get_cublas_handle()
        cublas.setStream(self.handle, self.stream.ptr)
        cublas.setPointerMode(self.handle, cublas.CUBLAS_POINTER_MODE_HOST)
        self.alpha, self.beta = np.array(1, dtype=accumulation), np.array(0, dtype=accumulation)
        self.m, self.k = a.shape
        self.n = b.shape[1]
        self.metadata.update(input_dtype=storage, multiplication_dtype=accumulation,
                             accumulation_dtype=accumulation, output_dtype=output,
                             compute_contract=f'cuBLAS GEMMEx pedantic compute type {self.compute_type}')

    def compute(self):
        cb = self.cublas
        cb.gemmEx(self.handle, cb.CUBLAS_OP_N, cb.CUBLAS_OP_N, self.n, self.m, self.k,
                  self.alpha.ctypes.data, self.b.data.ptr, self.input_type, self.n,
                  self.a.data.ptr, self.input_type, self.k, self.beta.ctypes.data,
                  self.output.data.ptr, self.output_type, self.n, self.compute_type,
                  cb.CUBLAS_GEMM_DEFAULT)
        return self.output
