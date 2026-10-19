import json
import numpy as np
import pandas as pd
import pytest
from parallelbench import engine
from parallelbench.backends import CPUBackend, discover_cuda, measure
from parallelbench.cli import main
from parallelbench.engine import BenchmarkConfig, run, summarize, validate
from parallelbench.export import export_results
from parallelbench.visualization import plot_results


@pytest.mark.parametrize('kwargs', [dict(sizes=(0,)), dict(sizes=()), dict(warmup=0),
    dict(repetitions=0), dict(seed=-1), dict(rtol=-1), dict(atol=float('nan')),
    dict(device='other'), dict(precision='float16')])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_cpu_correctness_and_buffer_reuse():
    a = np.array([[1, 2], [-3, 4]], dtype=np.float32)
    backend = CPUBackend(a, a)
    output = backend.compute()
    np.testing.assert_array_equal(output, [[-5, 10], [-15, 10]])
    assert backend.compute() is output


def test_validation_rejects_incorrect_results():
    with pytest.raises(AssertionError):
        validate(np.zeros((2, 2)), np.ones((2, 2)), BenchmarkConfig())


def test_statistics():
    result = summarize([1., 2., 3., 4., 5.])
    assert result['median_ms'] == 3
    assert result['mad_ms'] == 1
    assert result['iqr_ms'] == 2
    assert result['std_ms'] == pytest.approx(np.sqrt(2))
    assert result['mean_ms'] == 3
    assert result['min_ms'] == 1
    assert result['max_ms'] == 5


def test_measure_synchronizes_every_iteration():
    calls = []
    samples = measure(lambda: calls.append('compute'), 2, lambda: calls.append('sync'))
    assert calls == ['sync', 'compute', 'sync'] * 2
    assert len(samples) == 2 and all(value >= 0 for value in samples)


def test_cpu_run_exports_and_plot(tmp_path):
    report = run(BenchmarkConfig(sizes=(8, 16), warmup=1, repetitions=2, device='cpu'))
    assert len(report['results']) == 2
    assert all(row['valid'] and len(row['samples_ms']) == 2 for row in report['results'])
    assert report['environment']['libraries']['numpy']
    assert 'threadpools' in report['environment']
    csv_path, json_path = export_results(report, tmp_path)
    assert json.loads(json_path.read_text())['settings']['seed'] == 42
    frame = pd.read_csv(csv_path)
    assert len(frame) == 2
    assert json.loads(frame.iloc[0]['environment'])['cpu']
    assert plot_results(report, tmp_path).stat().st_size > 0


def test_no_cuda_fallback_and_explicit_failure(monkeypatch):
    monkeypatch.setattr(engine, 'discover_cuda', lambda: (None, 'Test: CUDA unavailable'))
    assert run(BenchmarkConfig(sizes=(4,), repetitions=1))['cuda_skip_reason'] == 'Test: CUDA unavailable'
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        run(BenchmarkConfig(sizes=(4,), device='cuda'))


def test_cli(tmp_path, capsys):
    assert main(['--device', 'cpu', '--sizes', '4', '--warmup', '1',
                 '--repetitions', '1', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'results.json').exists()
    assert main(['--sizes', '-1']) == 1
    assert 'Benchmark failed' in capsys.readouterr().err


def test_real_cuda():
    torch, reason = discover_cuda()
    if torch is None:
        pytest.skip(reason)
    original = torch.backends.cuda.matmul.allow_tf32
    report = run(BenchmarkConfig(sizes=(16,), warmup=1, repetitions=2, device='cuda'))
    assert torch.backends.cuda.matmul.allow_tf32 == original
    rows = report['results']
    assert len(rows) == 5
    assert {r['operation'] for r in rows} == {'compute', 'host_to_device', 'device_to_host', 'end_to_end'}
    assert all(r['valid'] for r in rows)
    assert next(r for r in rows if r['backend'] == 'cuda' and r['operation'] == 'compute')['speedup'] > 0
    cpu = next(r for r in rows if r['backend'] == 'cpu')
    e2e = next(r for r in rows if r['operation'] == 'end_to_end')
    assert e2e['end_to_end_speedup'] == pytest.approx(cpu['median_ms'] / e2e['median_ms'])


def test_seed_inputs_are_independent_of_size_order(monkeypatch):
    captured = []
    original = engine.CPUBackend

    def capture(a, b):
        captured.append((a.copy(), b.copy()))
        return original(a, b)

    monkeypatch.setattr(engine, 'CPUBackend', capture)
    run(BenchmarkConfig(sizes=(4, 8), seed=7, warmup=1, repetitions=1, device='cpu'))
    run(BenchmarkConfig(sizes=(8, 4), seed=7, warmup=1, repetitions=1, device='cpu'))
    for first, second in ((captured[0], captured[3]), (captured[1], captured[2])):
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            assert left.dtype == np.float32
