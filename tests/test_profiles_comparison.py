"""Analytical comparison fixtures are unit tests, not experimental datasets."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from parallelbench.cli import main
from parallelbench.comparison import compare_reports, compare_files, load_report
from parallelbench.engine import BenchmarkConfig, run
from parallelbench.export import export_results
from parallelbench.hardware import collect_environment, gpu_inventory
from parallelbench.profiles import profile_settings
from parallelbench.visualization import plot_results


def fixture_report():
    return dict(schema_version=2, timestamp_utc='fixture-only', settings=dict(warmup=2, repetitions=5,
        trials=2, seed=42, cpu_threads=1, min_cpu_sample_ms=1, randomize_order=True),
        environment={'cpu': 'unit-test fixture'}, warnings=[], cuda_skip_reason='test',
        results=[dict(size=32, backend='cpu', operation='compute', median_ms=4., samples_ms=[4.],
                      valid=True, unstable=False)], trial_results=[])


@pytest.mark.parametrize('name', ['quick', 'standard', 'research'])
def test_profiles_are_reproducible_copies(name):
    first = profile_settings(name)
    first['trials'] = -1
    assert profile_settings(name)['trials'] > 0
    with pytest.raises(ValueError):
        profile_settings('unknown')


def test_relative_performance_and_incompatible_conditions():
    base = fixture_report()
    other = deepcopy(base)
    other['results'][0]['median_ms'] = 2.
    report = compare_reports(base, other)
    assert report['results'][0]['relative_speedup'] == 2
    assert report['results'][0]['latency_change_percent'] == -50
    other['settings']['seed'] = 99
    incompatible = compare_reports(base, other)
    assert not incompatible['results']
    assert any('Incompatible' in warning for warning in incompatible['warnings'])


@pytest.mark.parametrize('key,value', [('timer', 'cuda_event'), ('precision', 'float16'),
    ('size', 64), ('metadata', {'accumulation_dtype': 'float16'})])
def test_non_equivalent_measurements_are_not_compared(key, value):
    base = fixture_report()
    other = deepcopy(base)
    other['results'][0][key] = value
    assert not compare_reports(base, other)['results']


def test_json_csv_roundtrip_and_comparison_plot(tmp_path):
    source = tmp_path / 'source'
    report = fixture_report()
    csv, js = export_results(report, source)
    loaded = load_report(csv)
    assert loaded['results'][0]['unstable'] is False
    assert loaded['results'][0]['valid'] is True
    comparison = compare_files(js, csv, labels=('format JSON', 'format CSV'))
    assert comparison['results'][0]['relative_speedup'] == 1
    assert any('same run' in warning for warning in comparison['warnings'])
    export_results(comparison, tmp_path / 'comparison')
    assert plot_results(comparison, tmp_path / 'comparison').exists()
    assert json.loads((tmp_path / 'comparison/results.json').read_text())['sources'][0]['sha256']


def test_invalid_medians_and_trial_csv_are_rejected(tmp_path):
    report = fixture_report()
    report['results'][0]['median_ms'] = -1
    source = tmp_path / 'bad.json'
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='positive'):
        load_report(source)
    report = fixture_report()
    report['results'].append(deepcopy(report['results'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        compare_reports(report, fixture_report())


def test_profile_explicit_overrides_and_cpu_cli(tmp_path):
    assert main(['--profile', 'quick', '--device', 'cpu', '--sizes', '4', '--repetitions', '2',
                 '--trials', '1', '--output', str(tmp_path)]) == 0
    report = load_report(tmp_path / 'results.json')
    assert report['settings']['profile'] == 'quick'
    assert report['settings']['sizes'] == [4]
    assert report['settings']['repetitions'] == 2
    assert report['settings']['trials'] == 1
    assert report['settings']['warmup'] == 2
    assert report['settings']['cpu_threads'] == 1


def test_compare_cli(tmp_path):
    source = tmp_path / 'source'
    csv, js = export_results(fixture_report(), source)
    assert main(['--compare', str(js), str(csv), '--labels', 'A', 'B',
                 '--output', str(tmp_path / 'comparison')]) == 0


def test_driver_inventory_and_metadata(monkeypatch):
    gpu_inventory.cache_clear()
    monkeypatch.setattr('parallelbench.hardware.subprocess.run', lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='0, Test GPU, 123.45, 4096\n'))
    try:
        devices, status = gpu_inventory()
        assert status == 'available'
        assert devices[0]['memory_bytes'] == 4096 * 1024 ** 2
        assert devices[0]['driver_version'] == '123.45'
        environment = collect_environment()
        assert environment['system_memory']['total_bytes'] > 0
        assert 'is_wsl' in environment
        assert 'physical_cpu_count' in environment
    finally:
        gpu_inventory.cache_clear()


def test_version_metadata_is_consistent():
    import re
    from pathlib import Path
    from parallelbench import __version__
    text = (Path(__file__).parents[1] / 'pyproject.toml').read_text()
    assert __version__ == re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1) == '1.0.0'


def test_compare_cli_refuses_methodology_override(tmp_path):
    with pytest.raises(SystemExit):
        main(['--compare', 'A.json', 'B.json', '--repetitions', '1'])
