"""Headless comparison plots with variability and run provenance."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, (latency, speedup) = plt.subplots(1, 2, figsize=(11, 4))
    try:
        for backend in ("cpu", "cuda"):
            rows = [r for r in report["results"] if r["backend"] == backend and r["operation"] == "compute"]
            if rows:
                latency.errorbar([r["size"] for r in rows], [r["median_ms"] for r in rows],
                    yerr=[r["mad_ms"] for r in rows], marker="o", label=backend.upper())
                if backend == "cuda":
                    speedup.plot([r["size"] for r in rows], [r["speedup"] for r in rows], marker="o")
        latency.set(xlabel="Square matrix dimension", ylabel="Median compute latency (ms; ± MAD)")
        latency.legend()
        speedup.axhline(1, color="gray", linestyle="--")
        speedup.set(xlabel="Square matrix dimension", ylabel="CPU / CUDA compute latency")
        if report["cuda_skip_reason"]:
            speedup.text(0.5, 0.5, "CUDA not measured", ha="center", transform=speedup.transAxes)
        fig.suptitle(f"ParallelBench · float32 · {report['timestamp_utc']}\nEnvironment: accompanying results.json / results.csv", fontsize=9)
        fig.tight_layout()
        path = directory / "comparison.png"
        fig.savefig(path, dpi=160)
        return path
    finally:
        plt.close(fig)
