"""Explicit precision contracts, accuracy metrics, and numerical experiments."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import random
import numpy as np
from .backends import load_torch
from .cupy_backend import CuBLASBackend, discover_cupy
from .engine import BenchmarkConfig, _aggregate, _make_inputs
from .experiments import _measure_case
from .hardware import collect_environment, cpu_thread_context


@dataclass(frozen=True)
class PrecisionConfig(BenchmarkConfig):
    sizes: tuple[int, ...] = (128, 256, 512)
    precisions: tuple[str, ...] = ('float32', 'float16', 'float64', 'mixed')
    max_memory_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if (not self.precisions or any(p not in ('float16', 'float32', 'float64', 'mixed') for p in self.precisions)
                or len(set(self.precisions)) != len(self.precisions)):
            raise ValueError('Choose distinct precisions: float16, float32, float64, mixed')
        if not 0 < self.max_memory_fraction <= 1:
            raise ValueError('GPU memory fraction must be in (0, 1]')
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError('Precision experiment sizes must be distinct')


def accuracy_metrics(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError('Accuracy metrics require nonempty arrays of equal shape')
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError('Accuracy metrics require finite values')
    difference = actual - reference
    norm = float(np.linalg.norm(reference))
    error_norm = float(np.linalg.norm(difference))
    return dict(max_absolute_error=float(np.max(np.abs(difference))),
                mean_absolute_error=float(np.mean(np.abs(difference))),
                relative_frobenius_error=error_norm / norm if norm else (0.0 if not error_norm else None),
                relative_error_defined=bool(norm or not error_norm),
                rmse=float(np.sqrt(np.mean(difference ** 2))))


class PrecisionCPUBackend:
    """NumPy native FP32/64 and explicitly emulated FP16 storage contracts."""
    def __init__(self, a, b, precision):
        if precision not in ('float16', 'float32', 'float64', 'mixed'):
            raise ValueError('Unsupported CPU precision')
        storage = 'float16' if precision in ('float16', 'mixed') else precision
        compute = 'float64' if precision == 'float64' else 'float32'
        output = 'float16' if precision == 'float16' else compute
        self.host_a, self.host_b = np.array(a, dtype=storage, order='C'), np.array(b, dtype=storage, order='C')
        self.a, self.b = self.host_a.astype(compute, copy=False), self.host_b.astype(compute, copy=False)
        self.output = np.empty((a.shape[0], b.shape[1]), dtype=output)
        self.accumulator = self.output if output == compute else np.empty(self.output.shape, dtype=compute)
        arrays = {id(x): x for x in (self.host_a, self.host_b, self.a, self.b, self.output, self.accumulator)}
        self.metadata = dict(input_dtype=storage, multiplication_dtype=compute, accumulation_dtype=compute,
            output_dtype=output, compute_contract=('NumPy FP32 GEMM, preconverted FP16 inputs; timed output rounding'
                if storage == 'float16' else f'NumPy {compute} GEMM'),
            emulated=storage == 'float16', host_buffer_bytes=sum(x.nbytes for x in arrays.values()),
            device_buffer_bytes=0, memory_scope='owned buffers, including conversion scratch; excludes library workspaces')

    def compute(self):
        np.matmul(self.a, self.b, out=self.accumulator)
        if self.output is not self.accumulator:
            np.copyto(self.output, self.accumulator, casting='unsafe')
        return self.output


def validate_rounding_contract(output, a, b, metadata):
    """Check an entrywise standard rounding-error bound on the stored input values."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    exact = a64 @ b64
    eps = np.finfo(metadata['accumulation_dtype']).eps
    product = a.shape[1] * eps
    if product >= 1:
        if not np.all(np.isfinite(output)):
            raise AssertionError('Nonfinite precision output')
        return False  # The usual gamma_k bound is not informative in this regime.
    gamma = product / (1 - product)
    absolute_product = np.abs(a64) @ np.abs(b64)
    bound = gamma * absolute_product
    bound += np.finfo(metadata['output_dtype']).eps * np.abs(exact)
    bound += np.finfo(metadata['output_dtype']).tiny
    # Include float64 reference rounding, including when the target is itself FP64.
    bound += np.finfo(np.float64).eps * a.shape[1] * absolute_product
    if not np.all(np.isfinite(output)) or not np.all(np.abs(output.astype(np.float64) - exact) <= bound):
        raise AssertionError('Output exceeds the declared precision rounding-error bound')
    return True


def _precision_speedups(rows):
    for row in rows:
        base = next((r for r in rows if r['size'] == row['size'] and r['backend'] == row['backend']
                     and r['operation'] == row['operation'] and r['timer'] == row['timer']
                     and r['precision'] == 'float32'), None)
        row['speedup_vs_fp32'] = base['median_ms'] / row['median_ms'] if base and row['median_ms'] > 0 else None


def run_precision(config):
    cp, reason = (None, 'CPU explicitly requested') if config.device == 'cpu' else discover_cupy()
    if config.device == 'cuda' and cp is None:
        raise RuntimeError(reason)
    pytorch = load_torch()[0]
    with cpu_thread_context(pytorch, config.cpu_threads):
        environment = collect_environment(pytorch=pytorch)
        if cp is not None:
            p = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
            environment['gpu'] = dict(name=p['name'].decode() if isinstance(p['name'], bytes) else p['name'],
                memory_bytes=p['totalGlobalMem'], compute_capability=[p['major'], p['minor']],
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
        all_rows, order, skipped = [], [], []
        if cp is None:
            skipped.append(dict(backend='cupy_cublas', reason=reason))
        rng = random.Random(config.seed)
        for trial in range(config.trials):
            current = []
            sizes = list(config.sizes)
            if config.randomize_order:
                rng.shuffle(sizes)
            for size in sizes:
                a, b, reference = _make_inputs(config, size, trial)
                cases = [dict(backend=name, variant='precision_contract', gpu=name != 'numpy', precision=precision)
                         for name in (('numpy', 'cupy_cublas') if cp is not None else ('numpy',))
                         for precision in config.precisions]
                if config.randomize_order:
                    rng.shuffle(cases)
                for case in cases:
                    mode = case['precision']
                    if case['gpu']:
                        try:
                            backend = CuBLASBackend(cp, a, b, precision=mode,
                                max_memory_fraction=config.max_memory_fraction)
                            backend.compute()  # Real capability probe; no silent dtype fallback.
                            backend.synchronize()
                        except cp.cuda.cublas.CUBLASError as exc:
                            if exc.status not in (8, 15):  # ARCH_MISMATCH / NOT_SUPPORTED only.
                                raise
                            skipped.append(dict(size=size, trial=trial + 1, precision=mode,
                                                backend=case['backend'], reason=str(exc)))
                            continue
                    else:
                        backend = PrecisionCPUBackend(a, b, mode)
                    order.append(dict(trial=trial + 1, size=size, backend=case['backend'], precision=mode))
                    measured = _measure_case(backend, case, config, size, trial + 1, reference, validate_result=False)
                    output = backend.host_output if case['gpu'] else backend.output
                    bound_checked = validate_rounding_contract(output, backend.host_a, backend.host_b, backend.metadata)
                    accuracy = accuracy_metrics(output, reference)
                    for row in measured:
                        row.update(accuracy=accuracy, rounding_bound_checked=bound_checked,
                            accuracy_within_tolerance=bool(np.allclose(output, reference, rtol=config.rtol, atol=config.atol)))
                    current.extend(measured)
                    del backend
                    if cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
            _precision_speedups(current)
            all_rows.extend(current)
        fields = ('size', 'backend', 'operation', 'variant', 'precision', 'timer')
        rows, warnings = _aggregate(all_rows, config, group_fields=fields, attach_speedups=False)
        if any(p in ('float16', 'mixed') for p in config.precisions):
            warnings.append('CPU FP16 storage modes use explicit FP32 emulation and conversion buffers; not native FP16 execution')
        for row in rows:
            group = [r for r in all_rows if all(r[f] == row[f] for f in fields)]
            accuracy = {}
            for field in ('max_absolute_error', 'mean_absolute_error', 'relative_frobenius_error', 'rmse'):
                values = [r['accuracy'][field] for r in group if r['accuracy'][field] is not None]
                accuracy[field] = (max(values) if field == 'max_absolute_error' else
                    float(np.sqrt(np.mean(np.square(values)))) if field == 'rmse' else float(np.mean(values))) if values else None
            accuracy['relative_error_defined'] = all(r['accuracy']['relative_error_defined'] for r in group)
            row.update(accuracy=accuracy, metadata=group[0]['metadata'], dtype=group[0]['dtype'],
                matrix_shape=[row['size'], row['size']],
                rounding_bound_checked=all(r['rounding_bound_checked'] for r in group),
                accuracy_within_tolerance=all(r['accuracy_within_tolerance'] for r in group))
            if not row['accuracy_within_tolerance'] and row['operation'] == 'compute' and row['timer'] == 'host_wall':
                warnings.append(f"Accuracy target exceeded: {row['backend']} {row['precision']} n={row['size']}; samples retained")
            if not row['rounding_bound_checked'] and row['operation'] == 'compute' and row['timer'] == 'host_wall':
                warnings.append(f"No useful gamma_k rounding bound: {row['backend']} {row['precision']} n={row['size']}; finiteness checked")
        _precision_speedups(rows)
        if config.trials < 2 or config.repetitions < 2:
            warnings.append('Limited variability assessment: use at least two repetitions and two trials.')
        return dict(schema_version=5, experiment='precision', timestamp_utc=datetime.now(timezone.utc).isoformat(),
            settings=asdict(config), environment=environment, results=rows, trial_results=all_rows,
            execution_order=order, skipped=skipped, warnings=warnings, cuda_skip_reason=reason)
