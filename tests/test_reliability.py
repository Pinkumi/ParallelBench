"""Reliability checks; deterministic timing fixtures are not experimental results."""
import gc
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info
from parallelbench import backends, engine
from parallelbench.backends import CUDABackend, calibrate_cpu, measure
from parallelbench.cli import main
from parallelbench.engine import BenchmarkConfig, run, summarize
from parallelbench.export import export_results
from parallelbench.visualization import plot_results


@pytest.mark.parametrize('kwargs', [dict(cpu_threads=0), dict(cpu_threads=True),
    dict(trials=0), dict(trials=1.5), dict(min_cpu_sample_ms=-1),
    dict(min_cpu_sample_ms=float('inf')), dict(instability_threshold=0),
    dict(instability_threshold=float('nan')), dict(randomize_order='yes')])
def test_invalid_reliability_config(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_batch_measurement_preserves_elapsed_and_gc(monkeypatch):
    # Deliberate timer fixture to verify units, not a measured benchmark.
    ticks = iter([0, 12_000_000, 20_000_000, 35_000_000])
    monkeypatch.setattr(backends, 'perf_counter_ns', lambda: next(ticks))
    calls = []
    was_enabled = gc.isenabled()

    def operation():
        assert not gc.isenabled()
        calls.append(1)

    elapsed = measure(operation, 2, iterations_per_sample=3)
    assert elapsed == [12.0, 15.0]
    assert len(calls) == 6
    assert gc.isenabled() == was_enabled


def test_gc_restored_after_failure():
    before = gc.isenabled()

    def fail():
        raise RuntimeError('timed operation failed')

    with pytest.raises(RuntimeError):
        measure(fail, 1)
    assert gc.isenabled() == before
    gc.disable()
    try:
        measure(lambda: None, 1)
        assert not gc.isenabled()
    finally:
        if before:
            gc.enable()


def test_cpu_calibration_is_separate_and_fixed(monkeypatch):
    batches = []

    def fixture_measure(operation, repetitions, *, iterations_per_sample):
        batches.append(iterations_per_sample)
        return [float(iterations_per_sample)]

    monkeypatch.setattr(backends, 'measure', fixture_measure)
    assert calibrate_cpu(lambda: None, 5) == 8
    assert batches == [1, 2, 4, 8]
    batches.clear()
    assert calibrate_cpu(lambda: None, 0) == 1
    assert batches == []


def test_cpu_normalization(monkeypatch):
    monkeypatch.setattr(engine, 'calibrate_cpu', lambda operation, target: 4)
    monkeypatch.setattr(engine, 'measure', lambda *args, **kwargs: [8.0, 12.0])
    backend = SimpleNamespace(compute=lambda: None)
    rows = engine._measure_backend(backend, [('compute', backend.compute)],
        BenchmarkConfig(repetitions=2), 1, 8, 'cpu')
    assert rows[0]['elapsed_samples_ms'] == [8.0, 12.0]
    assert rows[0]['samples_ms'] == [2.0, 3.0]
    assert rows[0]['iterations_per_sample'] == 4


def test_end_to_end_invokes_all_operations_in_order():
    calls = []
    fixture = SimpleNamespace(upload=lambda: calls.append('upload'),
        compute=lambda: calls.append('compute'), download=lambda: calls.append('download'))
    CUDABackend.end_to_end(fixture)
    assert calls == ['upload', 'compute', 'download']


def test_gpu_measurement_boundaries(monkeypatch):
    # Exercise the synchronized measurement path with a recording backend.
    calls = []
    fixture = SimpleNamespace(synchronize=lambda: calls.append('sync'))
    monkeypatch.setattr(engine, 'calibrate_cpu', lambda *args: pytest.fail('GPU must not use CPU calibration'))
    rows = engine._measure_backend(fixture, [('end_to_end', lambda: calls.append('pipeline'))],
        BenchmarkConfig(warmup=1, repetitions=2), 1, 8, 'cuda')
    assert calls == ['pipeline', 'sync', 'sync', 'pipeline', 'sync', 'sync', 'pipeline', 'sync']
    assert rows[0]['iterations_per_sample'] == 1
    assert rows[0]['elapsed_samples_ms'] == rows[0]['samples_ms']


def test_trials_inputs_order_and_exports(monkeypatch, tmp_path):
    captured = []
    original = engine.CPUBackend

    def capture(a, b):
        captured.append((a.copy(), b.copy()))
        return original(a, b)

    monkeypatch.setattr(engine, 'CPUBackend', capture)
    config = BenchmarkConfig(sizes=(4, 8), repetitions=3, warmup=1, trials=3,
                             cpu_threads=1, device='cpu', randomize_order=True, min_cpu_sample_ms=0)
    report = run(config)
    second = run(config)
    assert report['execution_order'] == second['execution_order']
    assert [row['size'] for row in report['execution_order'][:2]] != [4, 8]
    for first_inputs, second_inputs in zip(captured[:6], captured[6:]):
        for left, right in zip(first_inputs, second_inputs):
            np.testing.assert_array_equal(left, right)
    size4 = [inputs[0] for inputs in captured[:6] if inputs[0].shape == (4, 4)]
    assert not np.array_equal(size4[0], size4[1])
    assert len(report['results']) == 2
    assert len(report['trial_results']) == 6
    assert all(len(row['samples_ms']) == 9 for row in report['results'])
    export_results(report, tmp_path)
    restored = json.loads((tmp_path / 'results.json').read_text())
    assert restored['trial_results'] == report['trial_results']
    trials_csv = pd.read_csv(tmp_path / 'trial_results.csv')
    assert len(trials_csv) == 6
    assert json.loads(trials_csv.iloc[0]['samples_ms']) == report['trial_results'][0]['samples_ms']
    assert json.loads(trials_csv.iloc[0]['elapsed_samples_ms']) == report['trial_results'][0]['elapsed_samples_ms']
    assert plot_results(report, tmp_path).stat().st_size > 0


def test_thread_limits_are_effective_and_restored(monkeypatch):
    torch = engine.load_torch()[0]
    before = [(p['filepath'], p['num_threads']) for p in threadpool_info()]
    torch_before = torch.get_num_threads() if torch is not None else None
    interop_before = torch.get_num_interop_threads() if torch is not None else None
    config = BenchmarkConfig(sizes=(4,), warmup=1, repetitions=2, trials=2,
                             device='cpu', cpu_threads=1, min_cpu_sample_ms=0)
    report = run(config)
    assert all(pool['num_threads'] == 1 for pool in report['environment']['threadpools'])
    if torch is not None:
        assert report['environment']['pytorch']['intraop_threads'] == 1
        assert report['environment']['pytorch']['interop_threads'] == interop_before
        assert torch.get_num_threads() == torch_before
    assert [(p['filepath'], p['num_threads']) for p in threadpool_info()] == before

    def fail(*args):
        raise AssertionError('validation failure')

    monkeypatch.setattr(engine, 'validate', fail)
    with pytest.raises(AssertionError):
        run(config)
    assert [(p['filepath'], p['num_threads']) for p in threadpool_info()] == before
    if torch is not None:
        assert torch.get_num_threads() == torch_before


def test_instability_keeps_outliers_and_detects_trial_drift():
    config = BenchmarkConfig(trials=2, repetitions=3)
    fixture = [{"trial": trial, "size": 4, "backend": "cpu", "operation": "compute",
                **summarize(samples)} for trial, samples in
               [(1, [1., 1., 1.]), (2, [10., 10., 100.])]]
    rows, warnings = engine._aggregate(fixture, config)
    assert rows[0]['samples_ms'] == [1., 1., 1., 10., 10., 100.]
    assert rows[0]['unstable']
    assert rows[0]['trial_median_cv'] > config.instability_threshold
    assert warnings and 'Raw samples retained' in warnings[0]
    stable = [{**row, **summarize([1., 1., 1.])} for row in fixture]
    assert engine._aggregate(stable, config)[1] == []


def test_speedups_use_measured_pipeline_not_sum_of_medians():
    fixture = [dict(size=4, backend=backend, operation=operation, median_ms=latency)
               for backend, operation, latency in [('cpu', 'compute', 20.),
                   ('cuda', 'compute', 2.), ('cuda', 'host_to_device', 3.),
                   ('cuda', 'device_to_host', 4.), ('cuda', 'end_to_end', 8.)]]
    engine._attach_speedups(fixture)
    assert fixture[1]['compute_speedup'] == 10.
    assert fixture[-1]['end_to_end_speedup'] == 2.5
    assert fixture[-1]['speedup'] == 2.5
    assert fixture[2]['speedup'] is None


def test_existing_results_cannot_be_overwritten(tmp_path, monkeypatch, capsys):
    sentinel = tmp_path / 'results.json'
    sentinel.write_text('existing experimental data')
    with pytest.raises(FileExistsError):
        export_results({}, tmp_path)
    monkeypatch.setattr('parallelbench.cli.run', lambda config: pytest.fail('Must refuse before benchmarking'))
    assert main(['--output', str(tmp_path)]) == 1
    assert 'Refusing to overwrite' in capsys.readouterr().err
    assert sentinel.read_text() == 'existing experimental data'


def test_cli_reliability_options(tmp_path, capsys):
    assert main(['--device', 'cpu', '--sizes', '4', '--repetitions', '2', '--warmup', '1',
                 '--trials', '2', '--cpu-threads', '1', '--randomize-order',
                 '--min-cpu-sample-ms', '0', '--instability-threshold', '0.01',
                 '--output', str(tmp_path)]) == 0
    report = json.loads((tmp_path / 'results.json').read_text())
    assert report['settings']['trials'] == 2
    assert report['settings']['cpu_threads'] == 1
    assert report['settings']['randomize_order'] is True
    assert 'mean=' in capsys.readouterr().out


def test_real_cuda_multiple_trials():
    torch, reason = backends.discover_cuda()
    if torch is None:
        pytest.skip(reason)
    config = BenchmarkConfig(sizes=(16, 32), warmup=1, repetitions=2, trials=2,
        cpu_threads=1, device='cuda', randomize_order=True, min_cpu_sample_ms=0)
    original_threads = torch.get_num_threads()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    report = run(config)
    assert len(report['trial_results']) == 20
    assert len(report['results']) == 10
    assert all(row['valid'] and len(row['samples_ms']) == 4 for row in report['results'])
    assert report['environment']['pytorch']['intraop_threads'] == 1
    assert torch.get_num_threads() == original_threads
    assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
    assert any(row['backend'] == 'cuda' for row in report['execution_order'])
