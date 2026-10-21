"""Command-line entry point."""
import argparse
import sys
from .engine import BenchmarkConfig, run
from .export import ensure_output_available, export_results
from .visualization import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark float32 square matrix multiplication")
    parser.add_argument("--sizes", nargs="+", type=int,
                        help="Matrix sizes (default: 256 512 1024; scalability: 128 256 512 1024)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32"], default="float32")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--cpu-threads", type=int, help="BLAS/OpenMP and PyTorch intra-op thread limit")
    parser.add_argument("--trials", type=int, default=1, help="Independent in-process trials (default: 1)")
    parser.add_argument("--randomize-order", action="store_true", help="Seeded size and backend order")
    parser.add_argument("--min-cpu-sample-ms", type=float, default=5.0,
                        help="CPU batch calibration target in ms; 0 disables batching (default: 5)")
    parser.add_argument("--instability-threshold", type=float, default=0.10,
                        help="Coefficient of variation warning threshold (default: 0.10)")
    parser.add_argument("--scalability", action="store_true", help="Run the Numba CPU thread sweep")
    parser.add_argument("--numba-threads", nargs="+", type=int,
                        help="Numba thread counts, including 1 (scalability default: 1 2 4 8 16)")
    parser.add_argument("--include-torch-cpu", action="store_true",
                        help="Also measure PyTorch CPU in scalability mode")
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    if not args.scalability and (args.numba_threads is not None or args.include_torch_cpu):
        parser.error("--numba-threads and --include-torch-cpu require --scalability")
    try:
        config_class, execute = BenchmarkConfig, run
        extra = {}
        if args.scalability:
            from .scalability import ScalabilityConfig, run_scalability
            config_class, execute = ScalabilityConfig, run_scalability
            extra = dict(numba_threads=tuple(args.numba_threads) if args.numba_threads is not None
                         else ScalabilityConfig.numba_threads,
                         include_torch_cpu=args.include_torch_cpu)
        config = config_class(sizes=tuple(args.sizes) if args.sizes is not None else config_class.sizes,
                                 warmup=args.warmup,
                                 repetitions=args.repetitions, seed=args.seed,
                                 rtol=args.rtol, atol=args.atol, device=args.device,
                                 precision=args.precision, cpu_threads=args.cpu_threads,
                                 trials=args.trials, randomize_order=args.randomize_order,
                                 min_cpu_sample_ms=args.min_cpu_sample_ms,
                                 instability_threshold=args.instability_threshold, **extra)
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
        print(f"n={row['size']} {row['backend']}{threads} {row['operation']}: "
              f"median={row['median_ms']:.4f} ms, mean={row['mean_ms']:.4f} ms, "
              f"std={row['std_ms']:.4f} ms, MAD={row['mad_ms']:.4f} ms, "
              f"min={row['min_ms']:.4f} ms, max={row['max_ms']:.4f} ms")
        if row["backend"] == "cuda" and row["operation"] == "compute":
            print(f"  Compute speedup: {row['compute_speedup']}; "
                  f"end-to-end speedup: {row['end_to_end_speedup']}")
        if row["backend"] == "numba_parallel":
            print(f"  Speedup vs sequential: {row['speedup_vs_sequential']}; "
                  f"efficiency: {row['parallel_efficiency']}; "
                  f"speedup vs parallel(1): {row['speedup_vs_parallel_one_thread']}")
    print("Saved:", *paths, graph)
    print("Trial samples:", graph.parent / "trial_results.csv")
    if args.scalability:
        print("Vector plot:", graph.with_suffix('.pdf'))
    return 0
