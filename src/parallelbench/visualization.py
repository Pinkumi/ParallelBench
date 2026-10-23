"""Headless comparisons of pooled trial measurements with variability and provenance."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(report, directory):
    if report.get('experiment') == 'cross_device':
        return plot_device_comparison(report, directory)
    if report.get('experiment') == 'precision':
        return plot_precision_results(report, directory)
    if report.get('experiment') == 'cuda_kernels':
        return plot_kernel_results(report, directory)
    if report.get("experiment") == "cpu_scalability":
        return plot_scalability_results(report, directory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "comparison.png"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output directory")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    try:
        for column, operation in enumerate(("compute", "end_to_end")):
            latency, speedup = axes[0, column], axes[1, column]
            for backend in ("cpu", "cuda"):
                # CPU already receives and returns host arrays, so compute is its E2E baseline.
                selected = "compute" if backend == "cpu" else operation
                rows = sorted((r for r in report["results"] if r["backend"] == backend
                               and r["operation"] == selected), key=lambda row: row["size"])
                if rows:
                    latency.errorbar([r["size"] for r in rows], [r["median_ms"] for r in rows],
                        yerr=[r["mad_ms"] for r in rows], marker="o", label=backend.upper())
                    if backend == "cuda":
                        speedup.plot([r["size"] for r in rows], [r["speedup"] for r in rows], marker="o")
            title = "Compute only" if operation == "compute" else "End to end (host to host)"
            latency.set(title=title, xlabel="Square matrix dimension", ylabel="Median latency (ms; ± MAD)")
            latency.legend()
            speedup.axhline(1, color="gray", linestyle="--")
            speedup.set(title=f"{title} speedup", xlabel="Square matrix dimension",
                        ylabel="CPU / CUDA median latency")
            if report["cuda_skip_reason"]:
                speedup.text(0.5, 0.5, "CUDA not measured", ha="center", transform=speedup.transAxes)
        warning_count = len(report.get("warnings", []))
        fig.suptitle(f"ParallelBench · float32 · {report['timestamp_utc']}\n"
                     f"Environment and {warning_count} warning(s): accompanying results.json / results.csv",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        with path.open("xb") as target:
            fig.savefig(target, format="png", dpi=160)
        return path
    finally:
        plt.close(fig)


def plot_scalability_results(report, directory):
    """Five measured comparisons and experiment context; PDF preserves vector lines."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'comparison.png'
    pdf_path = path.with_suffix('.pdf')
    for target in (path, pdf_path):
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}; choose a new output directory')
    rows = [row for row in report['results'] if row['operation'] == 'compute']
    parallel = [row for row in rows if row['backend'] == 'numba_parallel']
    counts = sorted({row['actual_numba_threads'] for row in parallel})
    sizes = sorted({row['size'] for row in rows})
    with plt.rc_context({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 13))
        try:
            latency, speedup, efficiency = axes[0, 0], axes[0, 1], axes[1, 0]
            for index, size in enumerate(sizes):
                group = sorted((row for row in parallel if row['size'] == size),
                               key=lambda row: row['actual_numba_threads'])
                threads = [row['actual_numba_threads'] for row in group]
                color = f'C{index % 10}'
                latency.errorbar(threads, [row['median_ms'] for row in group],
                    yerr=[row['mad_ms'] for row in group], marker='o', color=color, label=f'n={size} parallel')
                sequential = next(row['median_ms'] for row in rows
                                  if row['backend'] == 'numba_sequential' and row['size'] == size)
                latency.axhline(sequential, linestyle='--', color=color, alpha=0.6,
                                label=f'n={size} sequential')
                for field, label, style in (
                        ('speedup_vs_sequential', 'vs sequential', '-'),
                        ('speedup_vs_parallel_one_thread', 'vs parallel(1)', '--')):
                    speedup.plot(threads, [row[field] for row in group], marker='o', linestyle=style,
                                 color=color, label=f'n={size} {label}')
                efficiency.plot(threads, [row['parallel_efficiency'] for row in group],
                                marker='o', color=color, label=f'n={size} vs sequential')
                efficiency.plot(threads, [row['efficiency_vs_parallel_one_thread'] for row in group],
                                linestyle='--', color=color, label=f'n={size} vs parallel(1)')
            speedup.plot(counts, counts, color='black', linestyle=':', label='Ideal linear speedup = p')
            efficiency.axhline(1, color='black', linestyle=':', label='Ideal efficiency = 1')
            latency.set(title='Execution latency vs Numba threads', xlabel='Configured Numba threads',
                        ylabel='Median latency (ms; ± MAD)', yscale='log')
            speedup.set(title='Parallel speedup: two measured baselines', xlabel='Configured Numba threads',
                        ylabel='Baseline median / parallel median')
            efficiency.set(title='Parallel efficiency', xlabel='Configured Numba threads',
                           ylabel='Speedup / threads')
            for axis in (latency, speedup, efficiency):
                axis.set_xticks(counts)
                axis.legend(fontsize=7)

            comparison, gpu = axes[1, 1], axes[2, 0]
            labels = {'numba_sequential': 'Numba sequential', 'cpu': 'NumPy (BLAS)',
                      'torch_cpu': 'PyTorch CPU', 'cuda': 'CUDA compute'}
            for backend in ('numba_sequential', 'numba_parallel', 'cpu', 'torch_cpu', 'cuda'):
                variants = counts if backend == 'numba_parallel' else [None]
                for threads in variants:
                    group = sorted((row for row in rows if row['backend'] == backend and
                                    (threads is None or row['actual_numba_threads'] == threads)),
                                   key=lambda row: row['size'])
                    if not group:
                        continue
                    unit = 'thread' if threads == 1 else 'threads'
                    label = f'Numba parallel ({threads} {unit})' if threads is not None else labels[backend]
                    if backend != 'cuda':
                        comparison.errorbar([r['size'] for r in group], [r['median_ms'] for r in group],
                            yerr=[r['mad_ms'] for r in group], marker='o', label=label)
                    if backend in ('cpu', 'torch_cpu', 'cuda'):
                        gpu.errorbar([r['size'] for r in group], [r['median_ms'] for r in group],
                            yerr=[r['mad_ms'] for r in group], marker='o', label=label)
            comparison.set(title='Custom kernels vs optimized CPU libraries',
                           xlabel='Square matrix dimension', ylabel='Median latency (ms; ± MAD)', yscale='log')
            gpu.set(title='CPU vs CUDA compute (transfers excluded)', xlabel='Square matrix dimension',
                    ylabel='Median latency (ms; ± MAD)', yscale='log')
            comparison.legend(fontsize=7)
            gpu.legend(fontsize=7)
            if not any(row['backend'] == 'cuda' for row in rows):
                gpu.text(0.5, 0.9, 'CUDA not measured', ha='center', transform=gpu.transAxes)
            settings, environment = report['settings'], report['environment']
            info = axes[2, 1]
            info.axis('off')
            lines = [environment['cpu'],
                     f"Physical cores: {environment['physical_cpu_count']}; logical CPUs: {environment['logical_cpu_count']}",
                     f"Numba layer: {environment['numba']['threading_layer']}",
                     f"Numba thread capacity: {environment['numba']['max_threads']}",
                     f"NumPy / PyTorch CPU requested threads: {settings['cpu_threads']}",
                     f"Trials: {settings['trials']}; repetitions: {settings['repetitions']}; warm-up: {settings['warmup']}",
                     f"CPU batch calibration target: {settings['min_cpu_sample_ms']} ms",
                     f"Seed: {settings['seed']}; randomized order: {settings['randomize_order']}",
                     f"Warnings: {len(report['warnings'])}; all raw samples retained",
                     'MAD bars show dispersion, not confidence intervals.',
                     'Full environment and samples: accompanying CSV / JSON.']
            info.text(0, 0.95, '\n\n'.join(lines), va='top', fontsize=9)
            fig.suptitle(f"ParallelBench CPU scalability · float32 · {report['timestamp_utc']}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            for target, format_name in ((path, 'png'), (pdf_path, 'pdf')):
                with target.open('xb') as output:
                    fig.savefig(output, format=format_name, dpi=200)
            return path
        finally:
            plt.close(fig)


def _save_research_plot(fig, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    targets = [directory / 'comparison.png', directory / 'comparison.pdf']
    if any(path.exists() for path in targets):
        raise FileExistsError('Refusing to overwrite existing plots; choose a new output directory')
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for path in targets:
            with path.open('xb') as output:
                fig.savefig(output, format=path.suffix[1:], dpi=200)
        return targets[0]
    finally:
        plt.close(fig)


def plot_kernel_results(report, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    selections = [('compute', 'host_wall', 'Host compute latency (ms)'),
                  ('compute', 'cuda_event', 'CUDA event elapsed (ms)'),
                  ('end_to_end', 'host_wall', 'Host end-to-end latency (ms)'),
                  ('compute', 'cuda_event', 'Event speedup vs naive baseline')]
    for axis, (operation, timer, title) in zip(axes.flat, selections):
        rows = [r for r in report['results'] if r['operation'] == operation and r['timer'] == timer]
        keys = dict.fromkeys((r['backend'], r['variant']) for r in rows)
        for backend, variant in keys:
            group = sorted((r for r in rows if (r['backend'], r['variant']) == (backend, variant)),
                           key=lambda r: r['size'])
            field = 'speedup_vs_naive' if axis is axes[1, 1] else 'median_ms'
            axis.errorbar([r['size'] for r in group], [r[field] for r in group], marker='o',
                          yerr=[r['mad_ms'] for r in group] if field == 'median_ms' else None,
                          label=f'{backend}: {variant}')
        axis.set(title=title, xlabel='Square matrix dimension', ylabel=title)
        axis.grid(alpha=0.25)
        if keys:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, 'CUDA not measured', ha='center', transform=axis.transAxes)
        if axis is axes[1, 1]:
            axis.axhline(1, color='gray', linestyle='--')
    fig.suptitle(f"ParallelBench custom CUDA · {report['timestamp_utc']}\n"
                 f"Baseline: {report['naive_baseline']} · {len(report['warnings'])} warning(s)\n"
                 'Host and CUDA-event speedups use separate timing boundaries. Environment: results.json', fontsize=10)
    return _save_research_plot(fig, directory)


def plot_precision_results(report, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    modes = report['settings']['precisions']
    rows = [r for r in report['results'] if r['operation'] == 'compute']
    groups = dict.fromkeys((r['backend'], r['size']) for r in rows)
    for backend, size in groups:
        group = [r for r in rows if r['backend'] == backend and r['size'] == size]
        for timer, axis in [('host_wall', axes[0, 0]), ('cuda_event', axes[0, 1])]:
            selected = {r['precision']: r for r in group if r['timer'] == timer}
            if selected:
                axis.errorbar(range(len(modes)), [selected[p]['median_ms'] if p in selected else float('nan') for p in modes],
                              yerr=[selected[p]['mad_ms'] if p in selected else 0 for p in modes],
                              marker='o', label=f'{backend}, n={size}')
        selected = {r['precision']: r for r in group if r['timer'] == 'host_wall'}
        axes[1, 0].plot(range(len(modes)), [selected[p]['accuracy']['relative_frobenius_error'] if p in selected else None
                                          for p in modes], marker='o', label=f'{backend}, n={size}')
        axes[1, 1].plot(range(len(modes)), [sum(selected[p]['metadata'][key] for key in
            ('host_buffer_bytes', 'device_buffer_bytes')) if p in selected else None for p in modes],
            marker='o', label=f'{backend}, n={size}')
    titles = ['Host compute latency (ms)', 'CUDA event elapsed (ms)',
              'Relative Frobenius error vs float64 reference', 'Live host + device buffer bytes (excludes workspaces)']
    for axis, title in zip(axes.flat, titles):
        axis.set(title=title, xlabel='Precision contract', ylabel=title)
        axis.set_xticks(range(len(modes)), modes)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
    axes[1, 0].set_yscale('symlog', linthresh=1e-12)
    fig.suptitle(f"ParallelBench precision analysis · {report['timestamp_utc']}\n"
                 f"{len(report['warnings'])} warning(s); unsupported cases are recorded in results.json\n"
                 'CPU FP16 storage modes use explicitly emulated FP32 GEMM; no native CPU FP16 speed claim.', fontsize=10)
    return _save_research_plot(fig, directory)


def plot_device_comparison(report, directory):
    rows = [r for r in report['results'] if r['operation'] == 'compute'] or report['results']
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    labels = report['settings']['labels']
    keys = dict.fromkeys((r['backend'], r['variant'], r['precision'], r['timer']) for r in rows)
    for backend, variant, precision, timer in keys:
        selected = sorted((r for r in rows if (r['backend'], r['variant'], r['precision'], r['timer']) ==
                            (backend, variant, precision, timer)), key=lambda r: r['size'])
        name = f'{backend}/{variant}/{precision}/{timer}'
        axes[0].plot([r['size'] for r in selected], [r['relative_speedup'] for r in selected],
                     marker='o', label=name)
        axes[1].plot([r['size'] for r in selected], [r['latency_change_percent'] for r in selected],
                     marker='o', label=name)
    axes[0].axhline(1, color='gray', linestyle='--')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[0].set(title=f'{labels[0]} / {labels[1]} latency ratio (>1 favors candidate)', ylabel='Relative speedup')
    axes[1].set(title='Candidate latency change (negative means lower latency)', ylabel='Latency change (%)')
    for axis in axes:
        axis.set_xlabel('Square matrix dimension')
        axis.grid(alpha=0.25)
        if keys:
            axis.legend(fontsize=6, ncol=2)
    fig.suptitle(f"ParallelBench cross-run comparison · {len(report['warnings'])} warning(s)\n"
                 'Equivalent algorithms, precision contracts and timing boundaries only. Full environments: results.json', fontsize=10)
    return _save_research_plot(fig, directory)
