"""Command-line entry point."""
import argparse
import sys
from . import __version__
from .engine import BenchmarkConfig, run
from .export import ensure_output_available, export_results
from .visualization import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible CPU/GPU matrix multiplication experiments")
    parser.add_argument('--version', action='version', version=f'ParallelBench {__version__}')
    parser.add_argument("--sizes", nargs="+", type=int,
                        help="Matrix sizes (default: 256 512 1024; scalability: 128 256 512 1024)")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32"], default="float32")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--cpu-threads", type=int, help="BLAS/OpenMP and PyTorch intra-op thread limit")
    parser.add_argument("--trials", type=int, help="Independent in-process trials (default: 1)")
    parser.add_argument("--randomize-order", action=argparse.BooleanOptionalAction, default=None,
                        help="Seeded size and backend order")
    parser.add_argument("--profile", choices=['quick', 'standard', 'research'])
    parser.add_argument("--compare", nargs=2, metavar=('BASELINE', 'CANDIDATE'), help="Compare exported JSON/CSV runs")
    parser.add_argument("--labels", nargs=2, default=['baseline', 'candidate'])
    parser.add_argument("--min-cpu-sample-ms", type=float,
                        help="CPU batch calibration target in ms; 0 disables batching (default: 5)")
    parser.add_argument("--instability-threshold", type=float, default=0.10,
                        help="Coefficient of variation warning threshold (default: 0.10)")
    parser.add_argument("--scalability", action="store_true", help="Run the Numba CPU thread sweep")
    parser.add_argument("--custom-cuda", action="store_true", help="Compare naive, tiled, and library CUDA kernels")
    parser.add_argument("--precision-study", action="store_true", help="Measure explicit precision contracts and accuracy")
    parser.add_argument("--precisions", nargs="+", choices=['float16', 'float32', 'float64', 'mixed'])
    parser.add_argument("--tiles", nargs="+", type=int, help="Custom CUDA tiles, e.g. 8 16 32")
    parser.add_argument("--blocks", nargs="+", help="Naive CUDA blocks, e.g. 8x8 16x16 32x8")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--numba-threads", nargs="+", type=int,
                        help="Numba thread counts, including 1 (scalability default: 1 2 4 8 16)")
    parser.add_argument("--include-torch-cpu", action="store_true",
                        help="Also measure PyTorch CPU in scalability mode")
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    if args.compare is not None:
        if any((args.custom_cuda, args.scalability, args.precision_study, args.profile)):
            parser.error('--compare cannot be combined with an experiment mode or profile')
        if any(value is not None for value in (args.sizes, args.warmup, args.repetitions, args.trials,
                args.cpu_threads, args.min_cpu_sample_ms, args.tiles, args.blocks, args.precisions,
                args.numba_threads, args.randomize_order)) or args.device != 'auto' or args.include_torch_cpu:
            parser.error('--compare reads the saved settings; do not pass benchmark overrides')
        try:
            from .comparison import compare_files
            ensure_output_available(args.output)
            report = compare_files(*args.compare, labels=args.labels)
            for warning in report['warnings']:
                print(f'Warning: {warning}', file=sys.stderr)
            if not report['results']:
                raise ValueError('No equivalent valid measurements; see compatibility warnings')
            paths = export_results(report, args.output)
            graph = plot_results(report, args.output)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f'Comparison failed: {exc}', file=sys.stderr)
            return 1
        print(f"Compared {len(report['results'])} equivalent measurements. Saved:", *paths, graph)
        return 0
    if sum((args.custom_cuda, args.scalability, args.precision_study)) > 1:
        parser.error("Choose one experiment mode")
    if args.precisions is not None and not args.precision_study:
        parser.error("--precisions requires --precision-study")
    if not args.custom_cuda and (args.tiles is not None or args.blocks is not None):
        parser.error("--tiles and --blocks require --custom-cuda")
    if not args.scalability and (args.numba_threads is not None or args.include_torch_cpu):
        parser.error("--numba-threads and --include-torch-cpu require --scalability")
    try:
        from .profiles import profile_settings
        preset = profile_settings(args.profile) if args.profile is not None else {}
        config_class, execute = BenchmarkConfig, run
        extra = {}
        if args.scalability:
            from .scalability import ScalabilityConfig, run_scalability
            config_class, execute = ScalabilityConfig, run_scalability
            extra = dict(numba_threads=tuple(args.numba_threads) if args.numba_threads is not None
                         else preset.get('numba_threads', ScalabilityConfig.numba_threads),
                         include_torch_cpu=args.include_torch_cpu)
        elif args.custom_cuda:
            from .experiments import CUDAConfig, run_cuda_kernels
            config_class, execute = CUDAConfig, run_cuda_kernels
            extra = dict(tiles=tuple(args.tiles) if args.tiles is not None else CUDAConfig.tiles,
                         blocks=tuple(tuple(int(x) for x in value.lower().split('x')) for value in args.blocks)
                         if args.blocks is not None else CUDAConfig.blocks,
                         max_memory_fraction=args.gpu_memory_fraction)
        elif args.precision_study:
            from .precision import PrecisionConfig, run_precision
            config_class, execute = PrecisionConfig, run_precision
            extra = dict(precisions=tuple(args.precisions) if args.precisions is not None else PrecisionConfig.precisions,
                         max_memory_fraction=args.gpu_memory_fraction)
        def selected(name):
            value = getattr(args, name)
            return value if value is not None else preset.get(name, getattr(config_class, name))
        config = config_class(sizes=tuple(selected('sizes')),
                                 warmup=selected('warmup'),
                                 repetitions=selected('repetitions'), seed=args.seed,
                                 rtol=args.rtol, atol=args.atol, device=args.device,
                                 precision=args.precision, cpu_threads=selected('cpu_threads'),
                                 trials=selected('trials'), randomize_order=selected('randomize_order'),
                                 min_cpu_sample_ms=selected('min_cpu_sample_ms'),
                                 instability_threshold=args.instability_threshold, profile=args.profile, **extra)
        ensure_output_available(args.output)
        report = execute(config)
        paths = export_results(report, args.output)
        graph = plot_results(report, args.output)
    except (ValueError, RuntimeError, OSError, AssertionError, MemoryError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    if report["cuda_skip_reason"]:
        print(report["cuda_skip_reason"])
    for warning in report["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    for row in report["results"]:
        threads = f" threads={row['requested_threads']}" if row.get('requested_threads') is not None else ""
        detail = f" {row['variant']} {row['precision']} [{row['timer']}]" if 'timer' in row else ''
        print(f"n={row['size']} {row['backend']}{threads}{detail} {row['operation']}: "
              f"median={row['median_ms']:.4f} ms, mean={row['mean_ms']:.4f} ms, "
              f"std={row['std_ms']:.4f} ms, MAD={row['mad_ms']:.4f} ms, "
              f"min={row['min_ms']:.4f} ms, max={row['max_ms']:.4f} ms")
        if row["backend"] == "cuda" and row["operation"] == "compute" and 'compute_speedup' in row:
            print(f"  Compute speedup: {row['compute_speedup']}; "
                  f"end-to-end speedup: {row['end_to_end_speedup']}")
        if row["backend"] == "numba_parallel":
            print(f"  Speedup vs sequential: {row['speedup_vs_sequential']}; "
                  f"efficiency: {row['parallel_efficiency']}; "
                  f"speedup vs parallel(1): {row['speedup_vs_parallel_one_thread']}")
    print("Saved:", *paths, graph)
    print("Trial samples:", graph.parent / "trial_results.csv")
    if args.scalability or args.custom_cuda or args.precision_study:
        print("Vector plot:", graph.with_suffix('.pdf'))
    return 0
