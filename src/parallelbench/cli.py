"""Command-line entry point."""
import argparse
import sys
from .engine import BenchmarkConfig, run
from .export import ensure_output_available, export_results
from .visualization import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark float32 square matrix multiplication")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024])
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
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(sizes=tuple(args.sizes), warmup=args.warmup,
                                 repetitions=args.repetitions, seed=args.seed,
                                 rtol=args.rtol, atol=args.atol, device=args.device,
                                 precision=args.precision, cpu_threads=args.cpu_threads,
                                 trials=args.trials, randomize_order=args.randomize_order,
                                 min_cpu_sample_ms=args.min_cpu_sample_ms,
                                 instability_threshold=args.instability_threshold)
        ensure_output_available(args.output)
        report = run(config)
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
        print(f"n={row['size']} {row['backend']} {row['operation']}: "
              f"median={row['median_ms']:.4f} ms, mean={row['mean_ms']:.4f} ms, "
              f"std={row['std_ms']:.4f} ms, MAD={row['mad_ms']:.4f} ms, "
              f"min={row['min_ms']:.4f} ms, max={row['max_ms']:.4f} ms")
        if row["backend"] == "cuda" and row["operation"] == "compute":
            print(f"  Compute speedup: {row['compute_speedup']}; "
                  f"end-to-end speedup: {row['end_to_end_speedup']}")
    print("Saved:", *paths, graph)
    print("Trial samples:", graph.parent / "trial_results.csv")
    return 0
