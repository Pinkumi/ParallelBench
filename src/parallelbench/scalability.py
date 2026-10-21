"""CPU scalability experiments reusing Phase 2 timing, statistics, and validation."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import random
import numpy as np
from threadpoolctl import threadpool_info
from .backends import CPUBackend, CUDABackend, TorchCPUBackend, discover_cuda, load_torch
from .engine import BenchmarkConfig, _aggregate, _attach_speedups, _make_inputs, _measure_backend, validate
from .hardware import collect_environment, cpu_thread_context


@dataclass(frozen=True)
class ScalabilityConfig(BenchmarkConfig):
    sizes: tuple[int, ...] = (128, 256, 512, 1024)
    numba_threads: tuple[int, ...] = (1, 2, 4, 8, 16)
    include_torch_cpu: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (not self.numba_threads or
                any(type(p) is not int or p < 1 for p in self.numba_threads) or
                len(set(self.numba_threads)) != len(self.numba_threads)):
            raise ValueError("Numba thread counts must be distinct positive integers")
        if 1 not in self.numba_threads:
            raise ValueError("Include 1 in --numba-threads for the one-thread parallel baseline")
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("Scalability matrix sizes must be distinct")
        if type(self.include_torch_cpu) is not bool:
            raise ValueError("Include PyTorch CPU must be a boolean")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _attach_scalability_metrics(rows):
    for row in rows:
        row.update(speedup=None, compute_speedup=None, end_to_end_speedup=None,
                   speedup_vs_sequential=None, parallel_efficiency=None,
                   speedup_vs_parallel_one_thread=None, efficiency_vs_parallel_one_thread=None)
    _attach_speedups([r for r in rows if r['backend'] in ('cpu', 'cuda')])
    for size in dict.fromkeys(r['size'] for r in rows):
        group = [r for r in rows if r['size'] == size]
        sequential = next(r['median_ms'] for r in group if r['backend'] == 'numba_sequential')
        parallel_one = next(r['median_ms'] for r in group
                            if r['backend'] == 'numba_parallel' and r['actual_numba_threads'] == 1)
        for row in group:
            if row['operation'] != 'compute':
                continue
            row['speedup_vs_sequential'] = _ratio(sequential, row['median_ms'])
            if row['backend'] in ('numba_sequential', 'numba_parallel', 'torch_cpu'):
                row['speedup'] = row['speedup_vs_sequential']
            if row['backend'] == 'numba_parallel':
                threads = row['actual_numba_threads']
                row['parallel_efficiency'] = (row['speedup'] / threads
                                              if row['speedup'] is not None else None)
                relative = _ratio(parallel_one, row['median_ms'])
                row['speedup_vs_parallel_one_thread'] = relative
                row['efficiency_vs_parallel_one_thread'] = relative / threads if relative is not None else None


def _measure_configuration(config, nb, pytorch, backend, name, threads, trial, size,
                           reference, numpy_reference):
    is_numba = name.startswith('numba_')
    context = (nb.numba_thread_context(threads) if is_numba
               else cpu_thread_context(pytorch, config.cpu_threads))
    with context:
        actual = nb.numba.get_num_threads() if is_numba else None
        if is_numba and actual != threads:
            raise RuntimeError(f"Numba requested {threads} threads but configured {actual}")
        pools = threadpool_info()
        if name == 'cuda':
            backend.end_to_end()
            backend.synchronize()
            functions = [('host_to_device', backend.upload), ('compute', backend.compute),
                         ('device_to_host', backend.download), ('end_to_end', backend.end_to_end)]
        else:
            functions = [('compute', backend.compute)]
        rows = _measure_backend(backend, functions, config, trial, size, name)
        if name == 'cuda':
            output = backend.download()
            backend.synchronize()
        elif name == 'torch_cpu':
            output = backend.output.numpy()
        else:
            output = backend.output
        validate(output, reference, config)
        validate(output, numpy_reference, config)
        for row in rows:
            row.update(requested_threads=threads, actual_numba_threads=actual,
                       numba_threading_layer=nb.numba.threading_layer() if is_numba else None,
                       matrix_shape=[size, size], dtype='float32', threadpools=pools)
        return rows


def _run_trials(config, nb, torch, pytorch, reason):
    with cpu_thread_context(pytorch, config.cpu_threads):
        environment = collect_environment(torch, pytorch=pytorch)
    environment['numba'] = {
        'version': nb.numba.__version__, 'max_threads': nb.numba.config.NUMBA_NUM_THREADS,
        'initial_threads': nb.numba.get_num_threads(), 'threading_layer': nb.numba.threading_layer(),
        'requested_threads': list(config.numba_threads), 'loop_order': 'i-k-j', 'fastmath': False,
    }
    trial_rows, execution_order = [], []
    order_rng = random.Random(config.seed)
    for trial in range(config.trials):
        current_rows = []
        sizes = list(config.sizes)
        if config.randomize_order:
            order_rng.shuffle(sizes)
        for size in sizes:
            with cpu_thread_context(pytorch, config.cpu_threads):
                a, b, reference = _make_inputs(config, size, trial)
                numpy_reference = a @ b  # Untimed float32 oracle shared by every backend.
            jobs = [('numba_sequential', 1)]
            jobs.extend(('numba_parallel', p) for p in config.numba_threads)
            jobs.append(('cpu', config.cpu_threads))
            if config.include_torch_cpu:
                jobs.append(('torch_cpu', config.cpu_threads))
            if torch is not None:
                jobs.append(('cuda', None))
            if config.randomize_order:
                order_rng.shuffle(jobs)
            for name, threads in jobs:
                # Allocation and dispatch happen outside measured operations.
                if name.startswith('numba_'):
                    backend = nb.NumbaBackend(a, b, parallel=name == 'numba_parallel')
                elif name == 'cpu':
                    backend = CPUBackend(a, b)
                elif name == 'torch_cpu':
                    backend = TorchCPUBackend(pytorch, a, b)
                else:
                    backend = CUDABackend(torch, a, b)
                execution_order.append(dict(trial=trial + 1, size=size, backend=name,
                                            requested_threads=threads))
                current_rows.extend(_measure_configuration(config, nb, pytorch, backend, name,
                    threads, trial + 1, size, reference, numpy_reference))
                del backend
            del a, b, reference, numpy_reference
        _attach_scalability_metrics(current_rows)
        trial_rows.extend(current_rows)
    rows, warnings = _aggregate(trial_rows, config,
        group_fields=('size', 'backend', 'operation', 'requested_threads', 'actual_numba_threads'),
        attach_speedups=False)
    _attach_scalability_metrics(rows)
    for row in rows:
        row.update(matrix_shape=[row['size'], row['size']], dtype='float32',
                   numba_threading_layer=(environment['numba']['threading_layer']
                                         if row['actual_numba_threads'] is not None else None))
    if config.trials < 2 or config.repetitions < 2:
        warnings.append('Limited variability assessment: use at least two repetitions and two trials.')
    physical = environment['physical_cpu_count']
    if physical is not None and max(config.numba_threads) > physical:
        warnings.append(f'Thread sweep exceeds {physical} physical cores; logical threads share core resources.')
    return dict(schema_version=3, experiment='cpu_scalability',
                timestamp_utc=datetime.now(timezone.utc).isoformat(), settings=asdict(config),
                environment=environment, cuda_skip_reason=reason, execution_order=execution_order,
                warnings=warnings, results=rows, trial_results=trial_rows)


def run_scalability(config):
    try:
        from . import numba_backend as nb
    except ImportError as exc:
        raise RuntimeError("Numba unavailable or incompatible; install 'parallelbench[scalability]'. "
                           f"Details: {exc}") from exc
    nb.validate_thread_counts(config.numba_threads)
    # Compile and start the Numba runtime before timing any backend, including calibration.
    nb.prepare_numba()
    torch, reason = (None, 'CPU explicitly requested') if config.device == 'cpu' else discover_cuda()
    if config.device == 'cuda' and torch is None:
        raise RuntimeError(reason)
    pytorch = torch if torch is not None else load_torch()[0]
    if config.include_torch_cpu and pytorch is None:
        raise RuntimeError('PyTorch CPU requested but PyTorch is not installed')
    old_tf32 = None
    if torch is not None:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _run_trials(config, nb, torch, pytorch, reason)
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
