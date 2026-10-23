"""Optional CUDA experiments using the existing CPU timing and trial statistics."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import random
from .backends import CPUBackend, discover_cuda, load_torch
from .cupy_backend import CuBLASBackend, RawMatmulBackend, discover_cupy
from .engine import BenchmarkConfig, _aggregate, _make_inputs, _measure_backend, summarize, validate
from .gpu_timing import MeasuredTorchCUDA
from .hardware import collect_environment, cpu_thread_context


@dataclass(frozen=True)
class CUDAConfig(BenchmarkConfig):
    sizes: tuple[int, ...] = (128, 256, 512)
    tiles: tuple[int, ...] = (8, 16, 32)
    blocks: tuple[tuple[int, int], ...] = ((16, 16),)
    max_memory_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not self.tiles or any(type(t) is not int or t not in (8, 16, 32) for t in self.tiles):
            raise ValueError('Tile sizes must be 8, 16, or 32')
        if (not self.blocks or any(len(b) != 2 or any(type(x) is not int or x < 1 for x in b)
                                   for b in self.blocks)):
            raise ValueError('Blocks must contain two positive integer dimensions')
        if not 0 < self.max_memory_fraction <= 1:
            raise ValueError('GPU memory fraction must be in (0, 1]')
        if len(set(self.sizes)) != len(self.sizes) or len(set(self.tiles)) != len(self.tiles) or len(set(self.blocks)) != len(self.blocks):
            raise ValueError('Experiment sizes, tiles, and blocks must be distinct')


def _measure_case(backend, case, config, size, trial, reference, *, validate_result=True):
    gpu = case['gpu']
    if gpu:
        functions = [('host_to_device', backend.upload), ('compute', backend.compute),
                     ('device_to_host', backend.download), ('end_to_end', backend.end_to_end)]
        backend.end_to_end()
        backend.synchronize()
    else:
        functions = [('compute', backend.compute)]
    rows = _measure_backend(backend, functions, config, trial, size, 'cuda' if gpu else 'cpu')
    for row in rows:
        row['timer'] = 'host_wall'
    if gpu:
        samples = backend.event_samples(config.repetitions)
        rows.append(dict(trial=trial, size=size, operation='compute', timer='cuda_event',
                         iterations_per_sample=1, elapsed_samples_ms=samples,
                         valid=True, unstable=summarize(samples)['cv'] > config.instability_threshold,
                         **summarize(samples)))
        output = backend.download()
        backend.synchronize()
    else:
        output = backend.output
    if validate_result:
        validate(output, reference, config)
    metadata = getattr(backend, 'metadata', dict(input_dtype='float32', multiplication_dtype='float32',
        accumulation_dtype='float32', output_dtype='float32',
        host_buffer_bytes=backend.a.nbytes + backend.b.nbytes + backend.output.nbytes,
        device_buffer_bytes=0, memory_scope='owned buffers; excludes library workspaces')) if not gpu else backend.metadata
    for row in rows:
        row.update(backend=case['backend'], variant=case['variant'], precision=case.get('precision', 'float32'),
                   dtype=metadata['output_dtype'], matrix_shape=[size, size], metadata=metadata)
    return rows


def _speedups(rows, baseline):
    for row in rows:
        matching = [r for r in rows if r['size'] == row['size'] and r['operation'] == row['operation']
                    and r['timer'] == row['timer']]
        naive = next((r for r in matching if r['backend'] == 'cuda_naive' and r['variant'] == baseline), None)
        sequential = next((r for r in rows if r['size'] == row['size'] and
                           r['backend'] == 'numba_sequential' and r['timer'] == 'host_wall'), None)
        row['speedup_vs_naive'] = naive['median_ms'] / row['median_ms'] if naive and row['median_ms'] > 0 else None
        row['speedup_vs_sequential'] = (sequential['median_ms'] / row['median_ms']
            if sequential and row['timer'] == 'host_wall' and row['operation'] in ('compute', 'end_to_end')
            and row['median_ms'] > 0 else None)


def _run_cuda(config, cp, torch, pytorch, reason):
    try:
        from . import numba_backend as nb
        nb.prepare_numba()
    except ImportError:
        nb = None
    with cpu_thread_context(pytorch, config.cpu_threads):
        environment = collect_environment(torch, pytorch=pytorch)
    if cp is not None:
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        environment['cupy'] = dict(version=cp.__version__, runtime_version=cp.cuda.runtime.runtimeGetVersion(),
            driver_api_version=cp.cuda.runtime.driverGetVersion(), device_name=p['name'].decode() if isinstance(p['name'], bytes) else p['name'])
        if environment['gpu'] is None:
            environment['gpu'] = dict(name=environment['cupy']['device_name'], memory_bytes=p['totalGlobalMem'],
                compute_capability=[p['major'], p['minor']])
    trial_rows, order, skipped = [], [], []
    if cp is None:
        skipped.append(dict(backend='custom_cuda', reason=reason))
    if nb is None:
        skipped.append(dict(backend='numba_sequential', reason='Numba optional dependency unavailable'))
    rng = random.Random(config.seed)
    baseline = f'block_{config.blocks[0][0]}x{config.blocks[0][1]}'
    for trial in range(config.trials):
        current = []
        sizes = list(config.sizes)
        if config.randomize_order:
            rng.shuffle(sizes)
        for size in sizes:
            with cpu_thread_context(pytorch, config.cpu_threads):
                a, b, reference = _make_inputs(config, size, trial)
            cases = [dict(backend='cpu', variant='numpy', gpu=False)]
            if nb is not None:
                cases.append(dict(backend='numba_sequential', variant='sequential', gpu=False))
            if cp is not None:
                cases.extend(dict(backend='cuda_naive', variant=f'block_{x}x{y}', block=(x, y), gpu=True)
                             for x, y in config.blocks)
                cases.extend(dict(backend='cuda_tiled', variant=f'tile_{t}', tile=t, gpu=True) for t in config.tiles)
                cases.append(dict(backend='cupy_cublas', variant='library', gpu=True))
            if torch is not None:
                cases.append(dict(backend='cuda', variant='pytorch', gpu=True))
            if config.randomize_order:
                rng.shuffle(cases)
            for case in cases:
                name = case['backend']
                context = nb.numba_thread_context(1) if name == 'numba_sequential' else cpu_thread_context(pytorch, config.cpu_threads)
                with context:
                    if name == 'cpu':
                        backend = CPUBackend(a, b)
                    elif name == 'numba_sequential':
                        backend = nb.NumbaBackend(a, b)
                    elif name == 'cuda':
                        backend = MeasuredTorchCUDA(torch, a, b)
                    elif name == 'cupy_cublas':
                        backend = CuBLASBackend(cp, a, b, max_memory_fraction=config.max_memory_fraction)
                    else:
                        backend = RawMatmulBackend(cp, a, b, tile=case.get('tile'),
                            block=case.get('block', (16, 16)), max_memory_fraction=config.max_memory_fraction)
                    order.append(dict(trial=trial + 1, size=size, backend=name, variant=case['variant']))
                    current.extend(_measure_case(backend, case, config, size, trial + 1, reference))
                    del backend
                if cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
                if torch is not None:
                    torch.cuda.empty_cache()
        _speedups(current, baseline)
        trial_rows.extend(current)
    fields = ('size', 'backend', 'operation', 'variant', 'precision', 'timer')
    rows, warnings = _aggregate(trial_rows, config, group_fields=fields, attach_speedups=False)
    for row in rows:
        example = next(r for r in trial_rows if all(r[f] == row[f] for f in fields))
        row.update(metadata=example['metadata'], matrix_shape=example['matrix_shape'], dtype=example['dtype'])
    _speedups(rows, baseline)
    if config.trials < 2 or config.repetitions < 2:
        warnings.append('Limited variability assessment: use at least two repetitions and two trials.')
    return dict(schema_version=4, experiment='cuda_kernels', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        settings=asdict(config), environment=environment, results=rows, trial_results=trial_rows,
        execution_order=order, skipped=skipped, warnings=warnings,
        cuda_skip_reason=reason, naive_baseline=baseline)


def run_cuda_kernels(config):
    cp, reason = (None, 'CPU explicitly requested') if config.device == 'cpu' else discover_cupy()
    torch, _ = (None, None) if config.device == 'cpu' else discover_cuda()
    if config.device == 'cuda' and cp is None:
        raise RuntimeError(reason)
    pytorch = torch if torch is not None else load_torch()[0]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch is not None else None
    try:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
        return _run_cuda(config, cp, torch, pytorch, reason)
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
