"""Compare real exported runs only after checking equivalent experimental conditions."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

METADATA = {'schema_version', 'timestamp_utc', 'settings', 'environment', 'experiment', 'warnings',
            'cuda_skip_reason', 'execution_order', 'skipped', 'naive_baseline'}


def _decode(value):
    if value in ('True', 'False'):
        return value == 'True'
    if value == '':
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def load_report(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        report = json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8') as source:
            raw = list(csv.DictReader(source))
        if not raw:
            raise ValueError('CSV report is empty')
        metadata = {key: raw[0][key] for key in METADATA if key in raw[0]}
        if any(any(row.get(key) != value for key, value in metadata.items()) for row in raw):
            raise ValueError('CSV mixes metadata from different runs')
        report = {key: _decode(value) for key, value in metadata.items()}
        report['results'] = [{key: _decode(value) for key, value in row.items() if key not in METADATA}
                             for row in raw]
    else:
        raise ValueError('Comparison inputs must be exported JSON or CSV files')
    if (not isinstance(report, dict) or not isinstance(report.get('settings'), dict)
            or not isinstance(report.get('environment'), dict) or not isinstance(report.get('results'), list)):
        raise ValueError('Invalid ParallelBench report structure')
    if report.get('experiment') == 'cross_device':
        raise ValueError('Load original benchmark runs, not an existing comparison')
    for row in report['results']:
        if not isinstance(row, dict) or any(key not in row for key in ('size', 'backend', 'operation', 'median_ms')):
            raise ValueError('Report contains an incomplete measurement')
        if not isinstance(row['median_ms'], (int, float)) or not math.isfinite(row['median_ms']) or row['median_ms'] <= 0:
            raise ValueError('Comparison requires finite, positive measured medians')
    return report


def _method(report):
    settings = report['settings']
    defaults = dict(warmup=3, repetitions=10, trials=1, seed=42, rtol=1e-3, atol=1e-3,
                    min_cpu_sample_ms=0 if report.get('schema_version') == 1 else 5.0,
                    randomize_order=False, cpu_threads=None)
    return {**{key: settings.get(key, default) for key, default in defaults.items()},
            'wall_timer_method': 'phase1-unbatched' if report.get('schema_version') == 1 else 'phase2-batched-v1'}


def _identity(row, report):
    backend = row['backend']
    variant = row.get('variant') or {'cpu': 'numpy', 'cuda': 'pytorch',
        'numba_sequential': 'sequential', 'numba_parallel': 'parallel'}.get(backend, 'default')
    precision = row.get('precision') or row.get('dtype') or report['settings'].get('precision', 'float32')
    threads = row.get('actual_numba_threads') if backend.startswith('numba_') else None
    if threads is None:
        threads = row.get('requested_threads')
    details = row.get('metadata') or {}
    contract = {key: details.get(key) for key in ('input_dtype', 'multiplication_dtype',
                'accumulation_dtype', 'output_dtype', 'compute_contract', 'emulated',
                'fused_multiply_add', 'compiler_options', 'tile', 'block') if key in details}
    # Library versions belong in environment warnings, not implicit equivalence claims.
    return (row['size'], backend, row['operation'], variant, precision,
            row.get('timer') or 'host_wall', threads, json.dumps(contract, sort_keys=True))


def compare_reports(baseline, candidate, labels=('baseline', 'candidate')):
    if len(labels) != 2 or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError('Provide two nonempty comparison labels')
    warnings, results = [], []
    if _method(baseline) != _method(candidate):
        conflicts = [key for key in _method(baseline) if _method(baseline)[key] != _method(candidate)[key]]
        warnings.append('Incompatible experimental settings; no ratios calculated: ' + ', '.join(conflicts))
    else:
        left = {}
        for row in baseline['results']:
            key = _identity(row, baseline)
            if key in left:
                raise ValueError('Duplicate baseline configuration; use aggregate results.csv, not trial_results.csv')
            left[key] = row
        seen = set()
        for row in candidate['results']:
            key = _identity(row, candidate)
            if key in seen:
                raise ValueError('Duplicate candidate configuration')
            seen.add(key)
            previous = left.get(key)
            if previous is None:
                warnings.append(f"No equivalent baseline: {row['backend']} n={row['size']} {row['operation']}")
                continue
            if not previous.get('valid', True) or not row.get('valid', True):
                warnings.append('Invalid numerical configuration excluded')
                continue
            base, current = previous['median_ms'], row['median_ms']
            results.append(dict(size=row['size'], backend=row['backend'], operation=row['operation'],
                variant=key[3], precision=key[4], timer=key[5], threads=key[6],
                baseline_median_ms=base, candidate_median_ms=current,
                relative_speedup=base / current, latency_change_percent=(current / base - 1) * 100,
                baseline_unstable=previous.get('unstable', False), candidate_unstable=row.get('unstable', False)))
        unmatched = set(left) - seen
        if unmatched:
            warnings.append(f'{len(unmatched)} baseline configurations have no equivalent candidate')
    if baseline.get('timestamp_utc') == candidate.get('timestamp_utc') and baseline['environment'] == candidate['environment']:
        warnings.append('Both sources describe the same run; this is a format/roundtrip comparison, not a hardware comparison')
    if baseline['environment'] != candidate['environment']:
        warnings.append('Environments differ. Ratios describe complete hardware/software setups, not isolated hardware effects')
    if any(r.get('unstable') for r in baseline['results'] + candidate['results']):
        warnings.append('Source runs contain unstable measurements; raw samples remain in the original reports')
    if any(r.get('accuracy_within_tolerance') is False for r in baseline['results'] + candidate['results']):
        warnings.append('Source precision cases exceeded their accuracy target; performance ratios do not establish acceptable accuracy')
    return dict(schema_version=6, experiment='cross_device', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        settings=dict(labels=list(labels), matched_methodology=_method(baseline)),
        environment=dict(baseline=baseline['environment'], candidate=candidate['environment']),
        warnings=warnings, results=results, trial_results=[], cuda_skip_reason=None,
        source_settings=dict(baseline=baseline['settings'], candidate=candidate['settings']))


def compare_files(baseline_path, candidate_path, labels=('baseline', 'candidate')):
    report = compare_reports(load_report(baseline_path), load_report(candidate_path), labels)
    report['sources'] = [dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                         for path in (baseline_path, candidate_path)]
    return report
