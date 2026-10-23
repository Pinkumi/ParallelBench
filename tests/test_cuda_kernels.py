import numpy as np
import pytest
from parallelbench.cupy_backend import RawMatmulBackend, CuBLASBackend, discover_cupy, validate_block
from parallelbench.experiments import CUDAConfig, run_cuda_kernels, _speedups


@pytest.mark.parametrize('settings', [dict(tiles=(7,)), dict(blocks=((0, 1),)),
    dict(max_memory_fraction=0), dict(tiles=(16, 16))])
def test_bad_cuda_settings(settings):
    with pytest.raises(ValueError):
        CUDAConfig(**settings)


def test_invalid_device_blocks():
    props = dict(maxThreadsPerBlock=1024, maxThreadsDim=(1024, 1024, 64), sharedMemPerBlock=49152)
    validate_block((32, 32), props, 8192)
    with pytest.raises(ValueError):
        validate_block((64, 32), props)
    with pytest.raises(ValueError):
        validate_block((16, 16), props, 100000)


def test_speedups_do_not_mix_host_and_event_timers():
    fixture = [dict(size=8, backend=name, variant=variant, operation='compute', timer=timer, median_ms=ms)
        for name, variant, timer, ms in [('numba_sequential', 'sequential', 'host_wall', 20.),
            ('cuda_naive', 'block_16x16', 'host_wall', 10.),
            ('cuda_naive', 'block_16x16', 'cuda_event', 2.),
            ('cuda_tiled', 'tile_16', 'cuda_event', 1.)]]
    _speedups(fixture, 'block_16x16')
    assert fixture[-1]['speedup_vs_naive'] == 2
    assert fixture[-1]['speedup_vs_sequential'] is None
    assert fixture[1]['speedup_vs_sequential'] == 2


@pytest.mark.parametrize('shape', [(1, 1, 1), (17, 9, 33), (32, 32, 32)])
@pytest.mark.parametrize('tile', [None, 8, 16, 32])
def test_real_custom_cuda_correctness(shape, tile):
    cp, reason = discover_cupy()
    if cp is None:
        pytest.skip(reason)
    m, k, n = shape
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
    backend = RawMatmulBackend(cp, a, b, tile=tile)
    backend.compute()
    result = backend.download()
    np.testing.assert_allclose(result, a.astype(np.float64) @ b.astype(np.float64), rtol=1e-3, atol=1e-3)
    assert len(backend.event_samples(2)) == 2
    backend.output.fill(cp.nan)
    backend.end_to_end()
    np.testing.assert_allclose(backend.download(), a @ b, rtol=1e-3, atol=1e-3)


def test_real_cublas_reference():
    cp, reason = discover_cupy()
    if cp is None:
        pytest.skip(reason)
    a = np.arange(15, dtype=np.float32).reshape(3, 5)
    b = np.arange(35, dtype=np.float32).reshape(5, 7)
    backend = CuBLASBackend(cp, a, b)
    backend.compute()
    np.testing.assert_allclose(backend.download(), a @ b)


def test_custom_cuda_cpu_fallback(monkeypatch, tmp_path):
    from parallelbench import experiments
    from parallelbench.export import export_results
    monkeypatch.setattr(experiments, 'discover_cupy', lambda: (None, 'CuPy unavailable fixture'))
    monkeypatch.setattr(experiments, 'discover_cuda', lambda: (None, 'CUDA unavailable fixture'))
    report = run_cuda_kernels(CUDAConfig(sizes=(4,), device='auto', repetitions=1, trials=1,
        warmup=1, cpu_threads=1, min_cpu_sample_ms=0))
    assert all(r['backend'] in ('cpu', 'numba_sequential') for r in report['results'])
    assert report['skipped'][0]['reason'] == 'CuPy unavailable fixture'
    export_results(report, tmp_path)


def test_custom_cuda_cli_cpu(tmp_path):
    from parallelbench.cli import main
    assert main(['--custom-cuda', '--device', 'cpu', '--profile', 'quick', '--sizes', '4',
        '--repetitions', '1', '--trials', '1', '--output', str(tmp_path)]) == 0
