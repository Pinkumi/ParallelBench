# ParallelBench

ParallelBench is an open-source research framework for reproducible comparisons of CPU and GPU numerical algorithms. Phase 1 benchmarks dense square matrix multiplication using NumPy on the CPU and optional PyTorch CUDA on the GPU.

The goal is to understand how problem size, hardware, numerical precision, parallelism, and transfer overhead affect performance. Results are measurements of the local environment, not universal hardware rankings. This repository contains no invented experimental results.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate with `.venv\Scripts\activate`. CPU use requires no NVIDIA software. For CUDA, install a PyTorch build appropriate for your NVIDIA driver and operating system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the optional dependency:

```bash
python -m pip install -e '.[cuda]'
```

A CUDA-enabled PyTorch build and a functioning NVIDIA driver are required; the optional dependency alone does not guarantee CUDA availability. Linux and WSL2 use the same CLI; CUDA under WSL2 also requires a compatible Windows NVIDIA driver. No hardware paths or GPU models are hardcoded. Intended research systems include a Ryzen 7 5800H / GTX 1650 laptop on native Ubuntu and a Ryzen 5 5500 / RTX 3060 Ti system under Ubuntu WSL2.

## Usage

```bash
parallelbench --sizes 256 512 1024 --warmup 3 --repetitions 20 --seed 42 --output results/run-001
python -m parallelbench --device cpu --sizes 64 128 --output results/cpu-check
parallelbench --device cuda --precision float32 --sizes 512 1024
pytest -q
```

`--device auto` (default) falls back to CPU with an explicit reason if PyTorch or CUDA is unavailable. Explicit `--device cuda` fails if CUDA is unavailable. Allocation failures, invalid settings, numerical validation failures, and output errors produce a nonzero exit status. At least one warm-up and one measured repetition are required. Increase repetitions for research runs; tiny smoke runs are only functional checks.

Each successful run produces `results.json`, `results.csv`, and `comparison.png`. Use a new output directory per run: filenames within a directory are overwritten. JSON and CSV include settings, UTC timestamp, environment, raw samples, validation status, and summary statistics. Transfer rows have no speedup value. Failed runs do not export a new successful report; use a fresh directory to avoid confusing old artifacts with a failed run.

## Architecture

```text
src/parallelbench/
  backends.py       NumPy / optional PyTorch CUDA operations and timing
  engine.py         Configuration, deterministic inputs, validation, orchestration
  hardware.py       CPU, GPU, OS, Python, libraries, BLAS and thread metadata
  export.py         CSV and JSON reports
  visualization.py  Headless Matplotlib comparison graph
  cli.py            Command-line interface
```

The CLI coordinates small independent modules. PyTorch is imported only when CUDA discovery is requested, allowing CPU-only installations. Tests cover configuration, correctness, export, plots, CLI behavior, and optional real CUDA execution.

## Benchmark methodology

- For each size, a seed derived from the configured seed and size generates two contiguous float32 matrices with uniform values in `[-1, 1)`. Both backends receive identical values. Changing size order does not change inputs.
- A float64 NumPy product serves as the validation reference. CPU and GPU outputs must pass `allclose` with default `rtol=1e-3`, `atol=1e-3`; tolerances can be changed explicitly. CUDA is also checked against the CPU output. These tolerances accommodate float32 accumulation and cancellation; they are not proof of correctness for every possible input or matrix size.
- Input generation, float64 reference calculation, CUDA initialization, buffer allocation, warm-up, validation, statistics, plotting, and export are outside compute timing. CPU and CUDA both reuse output buffers. Warm-up primes library initialization before measurement. Internal library workspace behavior is implementation dependent.
- CUDA timing uses host `perf_counter` with synchronization immediately before and after every sample. This measures synchronous wall-clock call latency including dispatch and synchronization overhead, not kernel-only CUDA event time. CPU uses the same wall-clock timer.
- CUDA input upload, matrix multiplication, and result download are separate measured operations using preallocated buffers. Host buffers are ordinary pageable memory. Transfer metrics include the copy API and synchronization overhead. Compute inputs remain on the GPU between repetitions. Reported speedup is CPU compute median divided by CUDA compute median; it excludes transfers and is not end-to-end application speedup.
- TF32 is disabled during a CUDA run and the original setting is restored afterward. Precision is float32 for both implementations. CPU and CUDA use different libraries and may use different reduction orders.
- NumPy is **not assumed to be single-threaded**. The report records detected BLAS thread pools, NumPy build configuration, and common thread environment variables. Set thread variables before launching Python if controlling parallelism, for example `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 parallelbench ...`. Such a setting is a distinct experimental condition.
- The report includes CPU model, logical CPU count, GPU name/memory/compute capability when used, OS, architecture, Python/library versions, PyTorch CUDA runtime, settings, raw samples, median, population standard deviation, IQR, and median absolute deviation (MAD). Graph error bars show MAD, not confidence intervals. CPU-only graphs explicitly indicate that CUDA was not measured.
- CPU measurements precede GPU measurements per size. Thermal state, power mode, background load, driver differences, BLAS implementation, and this fixed order can bias comparisons. Run on stable AC power, document driver/power settings externally, repeat independent runs, and inspect raw samples. A fixed seed reproduces inputs, not timings. Do not compare machines without documenting their environments. Current runs do not control CPU affinity, clock rates, or thermal state.

The float64 reference and all live buffers require extra memory. Start with modest sizes, especially on the GTX 1650. There is no automatic memory-size selection. No experimental performance claims are included in this README.

## Research objectives and roadmap

Phase 1 establishes validated float32 matrix multiplication, portable environment records, separate transfer measurements, and reproducible exports. Future development will investigate:

1. Controlled CPU parallelism, affinity, and BLAS thread scaling.
2. Custom CUDA kernels and comparison with library implementations.
3. Float64, float16, mixed precision, numerical error, and accuracy/performance tradeoffs.
4. Additional numerical algorithms and transfer-inclusive application benchmarks.
5. Independent-run statistics, configurable measurement ordering, and stronger hardware telemetry.
6. AI-assisted code optimization with correctness checks, recorded changes, and reproducible evaluation.

All optimizations should be evaluated with real measurements and explicit scientific limitations.
