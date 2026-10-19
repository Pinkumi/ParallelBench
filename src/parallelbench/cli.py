"""Command-line entry point."""
import argparse
import sys
from .engine import BenchmarkConfig, run
from .export import export_results
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
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(tuple(args.sizes), args.warmup, args.repetitions,
                                 args.seed, args.rtol, args.atol, args.device, args.precision)
        report = run(config)
        paths = export_results(report, args.output)
        graph = plot_results(report, args.output)
    except (ValueError, RuntimeError, OSError, AssertionError, MemoryError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    if report["cuda_skip_reason"]:
        print(report["cuda_skip_reason"])
    for row in report["results"]:
        print(f"n={row['size']} {row['backend']} {row['operation']}: "
              f"median={row['median_ms']:.4f} ms, MAD={row['mad_ms']:.4f} ms")
    print("Saved:", *paths, graph)
    return 0
