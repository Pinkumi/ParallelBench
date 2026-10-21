"""Explicit float32 row kernels. Imported only for the optional scalability mode."""
from contextlib import contextmanager
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=False, fastmath=False)
def matmul_sequential(a, b, output):
    """i-k-j traversal with contiguous inner loops and no BLAS calls."""
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            output[i, j] = np.float32(0)
        for k in range(a.shape[1]):
            value = a[i, k]
            for j in range(b.shape[1]):
                output[i, j] += value * b[k, j]


@njit(parallel=True, fastmath=False)
def matmul_parallel(a, b, output):
    """The same arithmetic as sequential; each worker owns independent rows."""
    for i in prange(a.shape[0]):
        for j in range(b.shape[1]):
            output[i, j] = np.float32(0)
        for k in range(a.shape[1]):
            value = a[i, k]
            for j in range(b.shape[1]):
                output[i, j] += value * b[k, j]


class NumbaBackend:
    def __init__(self, a, b, *, parallel=False):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("Numba matmul requires compatible two-dimensional inputs")
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError("Numba matmul requires float32 inputs")
        self.a, self.b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        self.output = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
        self.kernel = matmul_parallel if parallel else matmul_sequential

    def compute(self):
        self.kernel(self.a, self.b, self.output)
        return self.output


def validate_thread_counts(counts):
    capacity = numba.config.NUMBA_NUM_THREADS
    if any(type(count) is not int or not 1 <= count <= capacity for count in counts):
        raise ValueError(f"Requested Numba threads must be between 1 and {capacity}; "
                         "choose --numba-threads within that capacity")


@contextmanager
def numba_thread_context(threads):
    validate_thread_counts((threads,))
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        yield
    finally:
        numba.set_num_threads(previous)


def prepare_numba():
    """Compile both C-contiguous float32 signatures and initialize workers untimed."""
    a = np.ones((2, 2), dtype=np.float32)
    with numba_thread_context(1):
        NumbaBackend(a, a).compute()
        NumbaBackend(a, a, parallel=True).compute()
