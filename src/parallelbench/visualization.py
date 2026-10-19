"""Headless comparisons of pooled trial measurements with variability and provenance."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(report, directory):
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
