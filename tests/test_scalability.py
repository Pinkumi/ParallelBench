"""Actual kernel checks and analytical fixtures for scalability metrics."""
import json
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

numba = pytest.importorskip('numba', reason='Install parallelbench[scalability] for Phase 3 tests')
from parallelbench import engine, numba_backend as nb, scalability
from parallelbench.cli import main
from parallelbench.engine import summarize
from parallelbench.export import export_results, ensure_output_available
from parallelbench.scalability import ScalabilityConfig, run_scalability
from parallelbench.visualization import plot_results


def small_config(**changes):
    options = dict(sizes=(8, 16), numba_threads=tuple(p for p in (1, 2) if p <= numba.config.NUMBA_NUM_THREADS),
                   device='cpu', cpu_threads=1, warmup=1, trials=2,
                   repetitions=2, min_cpu_sample_ms=0)
    options.update(changes)
    return ScalabilityConfig(**options)


@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 5, 2), (17, 17, 17), (32, 32, 32), (128, 128, 128)])
@pytest.mark.parametrize('threads', [1, 2, 4, 8, 16])
def test_custom_kernels_correct_and_reuse_output(shape, threads):
    if threads > numba.config.NUMBA_NUM_THREADS:
        pytest.skip('Thread count exceeds available Numba capacity')
    n, k, m = shape
    rng = np.random.default_rng(17)
    a = rng.uniform(-1, 1, (n, k)).astype(np.float32)
    b = rng.uniform(-1, 1, (k, m)).astype(np.float32)
    a_before, b_before = a.copy(), b.copy()
    reference = a @ b
    sequential = nb.NumbaBackend(a, b)
    parallel = nb.NumbaBackend(a, b, parallel=True)
    with nb.numba_thread_context(threads):
        seq_out, par_out = sequential.compute(), parallel.compute()
        np.testing.assert_allclose(seq_out, reference, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(seq_out, par_out)
        # Poison the output to ensure each new call initializes it correctly.
        par_out.fill(np.nan)
        assert parallel.compute() is par_out
        assert sequential.compute() is seq_out
        np.testing.assert_array_equal(seq_out, par_out)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    assert seq_out.dtype == np.float32 and par_out.dtype == np.float32


def test_noncontiguous_inputs_are_prepared_outside_kernel():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]
    b = np.arange(16, dtype=np.float32).reshape(4, 4)[::2, :]
    backend = nb.NumbaBackend(a, b)
    assert backend.a.flags.c_contiguous and backend.b.flags.c_contiguous
    np.testing.assert_allclose(backend.compute(), a @ b)


@pytest.mark.parametrize('a,b', [(np.ones((2, 2)), np.ones((2, 2))),
    (np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),
    (np.ones(2, dtype=np.float32), np.ones((2, 2), dtype=np.float32))])
def test_kernel_rejects_invalid_inputs(a, b):
    with pytest.raises(ValueError):
        nb.NumbaBackend(a, b)


@pytest.mark.parametrize('changes', [dict(numba_threads=()), dict(numba_threads=(2, 4)),
    dict(numba_threads=(1, 1)), dict(numba_threads=(1, 0)), dict(numba_threads=(1, True)),
    dict(sizes=(8, 8)), dict(include_torch_cpu='yes')])
def test_invalid_scalability_settings(changes):
    with pytest.raises(ValueError):
        small_config(**changes)


def test_capacity_is_checked_before_compilation(monkeypatch):
    monkeypatch.setattr(nb, 'prepare_numba', lambda: pytest.fail('Must validate capacity first'))
    with pytest.raises(ValueError, match='capacity'):
        run_scalability(small_config(numba_threads=(1, numba.config.NUMBA_NUM_THREADS + 1)))


def test_thread_context_restores_on_exception():
    previous = numba.get_num_threads()
    with pytest.raises(RuntimeError):
        with nb.numba_thread_context(1):
            assert numba.get_num_threads() == 1
            raise RuntimeError('fixture failure')
    assert numba.get_num_threads() == previous


def test_scalability_metrics_have_two_distinct_baselines():
    # Analytical fixture only; these numbers are not experimental measurements.
    rows = [dict(size=8, backend=backend, operation='compute',
                 actual_numba_threads=threads, median_ms=latency)
            for backend, threads, latency in [('numba_sequential', 1, 20.),
                ('numba_parallel', 1, 25.), ('numba_parallel', 4, 5.), ('cpu', None, 2.)]]
    scalability._attach_scalability_metrics(rows)
    assert rows[2]['speedup_vs_sequential'] == 4.
    assert rows[2]['parallel_efficiency'] == 1.
    assert rows[2]['speedup_vs_parallel_one_thread'] == 5.
    assert rows[2]['efficiency_vs_parallel_one_thread'] == 1.25
    assert rows[1]['speedup_vs_parallel_one_thread'] == 1.
    assert rows[1]['speedup_vs_sequential'] == 0.8
    assert rows[-1]['parallel_efficiency'] is None


def test_aggregation_keeps_thread_configurations_separate():
    fixtures = [dict(trial=trial, size=8, backend='numba_parallel', operation='compute',
                     requested_threads=p, actual_numba_threads=p, **summarize([float(p)] * 2))
                for trial in (1, 2) for p in (1, 2)]
    rows, warnings = engine._aggregate(fixtures, small_config(),
        group_fields=('size', 'backend', 'operation', 'requested_threads', 'actual_numba_threads'),
        attach_speedups=False)
    assert len(rows) == 2 and not warnings
    assert rows[0]['samples_ms'] == [1.] * 4
    assert rows[1]['samples_ms'] == [2.] * 4
    assert all(row['trial_count'] == 2 for row in rows)


def test_cpu_only_run_serialization_plots_and_jit_exclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(scalability, 'discover_cuda', lambda: pytest.fail('CPU-only must not probe CUDA'))
    original_measure = engine.measure
    original_prepare = nb.prepare_numba
    events = []

    def prepare():
        original_prepare()
        events.append('compiled')

    def checked_measure(*args, **kwargs):
        assert events and events[0] == 'compiled'
        assert nb.matmul_sequential.nopython_signatures
        assert nb.matmul_parallel.nopython_signatures
        events.append('measured')
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(nb, 'prepare_numba', prepare)
    monkeypatch.setattr(engine, 'measure', checked_measure)
    previous = numba.get_num_threads()
    config = small_config()
    report = run_scalability(config)
    assert numba.get_num_threads() == previous
    assert report['schema_version'] == 3
    assert report['environment']['numba']['version'] == numba.__version__
    assert 'physical_cpu_count' in report['environment']
    count = len(config.numba_threads) + 2
    assert len(report['results']) == count * len(config.sizes)
    assert len(report['trial_results']) == count * len(config.sizes) * config.trials
    for row in report['results']:
        assert row['valid'] and len(row['samples_ms']) == 4
        if row['backend'].startswith('numba_'):
            assert row['requested_threads'] == row['actual_numba_threads']
            assert row['numba_threading_layer']
    csv_path, json_path = export_results(report, tmp_path)
    restored = json.loads(json_path.read_text())
    assert restored['trial_results'] == report['trial_results']
    csv = pd.read_csv(csv_path)
    assert len(csv) == len(report['results'])
    assert json.loads(csv.iloc[0]['samples_ms']) == report['results'][0]['samples_ms']
    trials = pd.read_csv(tmp_path / 'trial_results.csv')
    assert json.loads(trials.iloc[0]['elapsed_samples_ms']) == report['trial_results'][0]['elapsed_samples_ms']
    png = plot_results(report, tmp_path)
    assert png.stat().st_size > 0
    assert png.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    with pytest.raises(FileExistsError):
        plot_results(report, tmp_path)


def test_randomization_reproducible_and_inputs_equivalent(monkeypatch):
    captured = []
    original = nb.NumbaBackend

    class Capture(original):
        def __init__(self, a, b, **kwargs):
            captured.append((a.copy(), b.copy(), kwargs.get('parallel', False)))
            super().__init__(a, b, **kwargs)

    monkeypatch.setattr(nb, 'NumbaBackend', Capture)
    config = small_config(randomize_order=True)
    first = run_scalability(config)
    split = len(captured)
    second = run_scalability(config)
    assert first['execution_order'] == second['execution_order']
    # Ignore tiny 2x2 preparation inputs and compare the actual configuration inputs.
    real = [entry for entry in captured[:split] if entry[0].shape != (2, 2)]
    repeated = [entry for entry in captured[split:] if entry[0].shape != (2, 2)]
    for left, right in zip(real, repeated):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
    # All implementations for a size and trial use identical values.
    per_size = len(config.numba_threads) + 1
    for start in range(0, len(real), per_size):
        group = real[start:start + per_size]
        for entry in group[1:]:
            np.testing.assert_array_equal(group[0][0], entry[0])
            np.testing.assert_array_equal(group[0][1], entry[1])
    same_size = [entry[0] for entry in real if entry[0].shape == real[0][0].shape]
    assert not np.array_equal(same_size[0], same_size[per_size])


def test_thread_state_restored_when_validation_fails(monkeypatch):
    nb.prepare_numba()
    torch = scalability.load_torch()[0]
    previous = numba.get_num_threads()
    torch_threads = torch.get_num_threads() if torch is not None else None
    pools = [(p['filepath'], p['num_threads']) for p in threadpool_info()]

    def fail(*args):
        raise AssertionError('deliberate validation failure')

    monkeypatch.setattr(scalability, 'validate', fail)
    with pytest.raises(AssertionError):
        run_scalability(small_config())
    assert numba.get_num_threads() == previous
    assert [(p['filepath'], p['num_threads']) for p in threadpool_info()] == pools
    if torch is not None:
        assert torch.get_num_threads() == torch_threads


def test_scalability_without_pytorch(monkeypatch):
    monkeypatch.setattr(scalability, 'load_torch', lambda: (None, 'Not installed'))
    report = run_scalability(small_config(sizes=(4,)))
    assert report['environment']['pytorch'] is None
    assert not any(r['backend'] in ('torch_cpu', 'cuda') for r in report['results'])
    with pytest.raises(RuntimeError, match='PyTorch CPU'):
        run_scalability(small_config(sizes=(4,), include_torch_cpu=True))


def test_optional_torch_cpu():
    torch = pytest.importorskip('torch')
    original = torch.get_num_threads()
    report = run_scalability(small_config(sizes=(4,), include_torch_cpu=True))
    assert any(row['backend'] == 'torch_cpu' and row['valid'] for row in report['results'])
    assert torch.get_num_threads() == original


def test_cli_scalability_and_old_defaults(tmp_path, monkeypatch):
    assert main(['--scalability', '--device', 'cpu', '--sizes', '8', '--numba-threads', '1',
                 '--cpu-threads', '1', '--warmup', '1', '--trials', '2', '--repetitions', '2',
                 '--min-cpu-sample-ms', '0', '--output', str(tmp_path)]) == 0
    report = json.loads((tmp_path / 'results.json').read_text())
    assert report['experiment'] == 'cpu_scalability'
    assert report['settings']['numba_threads'] == [1]
    assert (tmp_path / 'comparison.pdf').exists()
    captured = []

    def capture(config):
        captured.append(config)
        raise RuntimeError('Configuration inspected without executing benchmark')

    monkeypatch.setattr('parallelbench.cli.run', capture)
    assert main(['--output', str(tmp_path / 'legacy')]) == 1
    assert captured[0].sizes == (256, 512, 1024)
    monkeypatch.setattr(scalability, 'run_scalability', capture)
    assert main(['--scalability', '--output', str(tmp_path / 'defaults')]) == 1
    assert captured[-1].sizes == (128, 256, 512, 1024)


def test_cli_requires_scalability_for_new_options():
    with pytest.raises(SystemExit) as error:
        main(['--numba-threads', '1', '2'])
    assert error.value.code == 2


def test_pdf_is_protected_from_overwrite(tmp_path):
    pdf = tmp_path / 'comparison.pdf'
    pdf.write_bytes(b'existing data')
    with pytest.raises(FileExistsError):
        ensure_output_available(tmp_path)
    assert pdf.read_bytes() == b'existing data'


def test_real_cuda_scalability():
    torch, reason = scalability.discover_cuda()
    if torch is None:
        pytest.skip(reason)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    report = run_scalability(small_config(sizes=(16,), device='cuda'))
    cuda = [row for row in report['results'] if row['backend'] == 'cuda']
    assert len(cuda) == 4 and all(row['valid'] for row in cuda)
    assert next(row for row in cuda if row['operation'] == 'compute')['compute_speedup'] > 0
    assert torch.backends.cuda.matmul.allow_tf32 == previous_tf32


@pytest.mark.parametrize('size', [128, 256, 512, 1024])
def test_phase3_requested_sizes(size):
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (size, size)).astype(np.float32)
    b = rng.uniform(-1, 1, (size, size)).astype(np.float32)
    reference = a @ b
    sequential = nb.NumbaBackend(a, b)
    parallel = nb.NumbaBackend(a, b, parallel=True)
    with nb.numba_thread_context(min(4, numba.config.NUMBA_NUM_THREADS)):
        np.testing.assert_allclose(sequential.compute(), reference, rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(parallel.compute(), sequential.output)
