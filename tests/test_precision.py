import json
import numpy as np
import pytest
from parallelbench.precision import accuracy_metrics, PrecisionCPUBackend, PrecisionConfig, run_precision, validate_rounding_contract
from parallelbench.cupy_backend import CuBLASBackend, discover_cupy
from parallelbench.export import export_results
from parallelbench.visualization import plot_results


def test_accuracy_metrics_and_zero_reference():
    metrics = accuracy_metrics(np.array([2., 4.]), np.array([1., 2.]))
    assert metrics['max_absolute_error'] == 2
    assert metrics['mean_absolute_error'] == 1.5
    assert metrics['relative_frobenius_error'] == 1
    assert metrics['rmse'] == pytest.approx(np.sqrt(2.5))
    assert accuracy_metrics(np.zeros(2), np.zeros(2))['relative_frobenius_error'] == 0
    result = accuracy_metrics(np.ones(2), np.zeros(2))
    assert result['relative_frobenius_error'] is None and not result['relative_error_defined']
    with pytest.raises(ValueError):
        accuracy_metrics([float('nan')], [0])


@pytest.mark.parametrize('mode', ['float16', 'float32', 'float64', 'mixed'])
def test_cpu_precision_contracts(mode):
    a = np.arange(15, dtype=np.float32).reshape(3, 5) / 19
    b = np.arange(35, dtype=np.float32).reshape(5, 7) / 23
    backend = PrecisionCPUBackend(a, b, mode)
    output = backend.compute()
    assert str(output.dtype) == backend.metadata['output_dtype']
    assert backend.compute() is output
    assert validate_rounding_contract(output, backend.host_a, backend.host_b, backend.metadata)
    assert backend.metadata['emulated'] == (mode in ('float16', 'mixed'))
    backend.output.fill(10000)
    with pytest.raises(AssertionError):
        validate_rounding_contract(backend.output, backend.host_a, backend.host_b, backend.metadata)


def test_precision_cpu_only_exports(monkeypatch, tmp_path):
    monkeypatch.setattr('parallelbench.precision.discover_cupy', lambda: pytest.fail('Must not probe CUDA'))
    report = run_precision(PrecisionConfig(sizes=(4,), trials=2, repetitions=2, warmup=1,
                                         device='cpu', cpu_threads=1, min_cpu_sample_ms=0))
    assert len(report['results']) == 4
    assert all(len(r['samples_ms']) == 4 for r in report['results'])
    export_results(report, tmp_path)
    restored = json.loads((tmp_path / 'results.json').read_text())
    assert restored['trial_results'] == report['trial_results']
    assert plot_results(report, tmp_path).exists()


@pytest.mark.parametrize('mode', ['float16', 'float32', 'float64', 'mixed'])
def test_real_gpu_precision_contract(mode):
    cp, reason = discover_cupy()
    if cp is None:
        pytest.skip(reason)
    rng = np.random.default_rng(42)
    a, b = rng.uniform(-1, 1, (17, 9)).astype(np.float32), rng.uniform(-1, 1, (9, 13)).astype(np.float32)
    try:
        backend = CuBLASBackend(cp, a, b, precision=mode)
        backend.compute()
        result = backend.download()
    except cp.cuda.cublas.CUBLASError as exc:
        if exc.status in (8, 15):
            pytest.skip(f'Hardware/library precision not supported: {exc}')
        raise
    assert validate_rounding_contract(result, backend.host_a, backend.host_b, backend.metadata)
    assert str(result.dtype) == backend.metadata['output_dtype']
    assert accuracy_metrics(result, a.astype(np.float64) @ b.astype(np.float64))['max_absolute_error'] >= 0
