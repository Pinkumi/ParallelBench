# ParallelBench

ParallelBench is an open-source research framework for reproducible comparisons of CPU and GPU numerical algorithms. Phases 1 and 2 benchmark dense square matrix multiplication using NumPy on the CPU and optional PyTorch CUDA on the GPU.

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

For a comparison with repeated trials and controlled CPU parallelism:

```bash
parallelbench --device cuda --sizes 256 512 1024 2048 \
  --warmup 5 --repetitions 20 --trials 5 --cpu-threads 8 \
  --randomize-order --seed 42 --output results/reliability-001
```

`--device cuda` measures both CPU and CUDA, requiring a working GPU. Choose the CPU thread count for the experiment; eight is an example, not a hardware-independent recommendation.

| Reliability option | Default | Meaning |
| --- | --- | --- |
| `--cpu-threads N` | Unchanged | Temporarily limit loaded BLAS/OpenMP pools and PyTorch intra-op threads |
| `--trials N` | `1` | Fresh inputs, buffers, warm-up, and measurements per trial |
| `--randomize-order` | Off | Reproducibly shuffle sizes and CPU/CUDA backend order per size per trial |
| `--min-cpu-sample-ms X` | `5.0` | Calibration target for CPU batches; `0` uses one call per sample |
| `--instability-threshold X` | `0.10` | Warn when a coefficient of variation exceeds this fraction |

The original options remain available (`--sizes`, `--warmup`, `--repetitions`, `--seed`, `--device`, `--precision`, `--rtol`, `--atol`, `--output`). Run `parallelbench --help` for all options. Single-trial operation remains the default for compatibility; use multiple trials for reliability studies.

Each successful run produces:

- `results.json`: schema version 2, settings, environment, execution order, warnings, aggregate statistics, and complete per-trial records.
- `results.csv`: aggregate statistics and pooled normalized timing samples, with environment metadata.
- `trial_results.csv`: per-trial statistics, trial IDs, batch sizes, original batch durations, normalized timing samples, and environment metadata.
- `comparison.png`: compute latency, end-to-end latency, compute speedup, and end-to-end speedup.

Use a new output directory for every run. The CLI checks for existing artifact names before measurement and refuses to overwrite them. Export and plotting also reject existing output files. Files from earlier experiments are never migrated or rewritten. A failed export can leave partial new artifacts; use a fresh directory for a retry.

Schema 2 retains the Phase 1 aggregate fields (`samples_ms`, `median_ms`, `std_ms`, `iqr_ms`, `mad_ms`, `speedup`, and validation status) and adds the trial and reliability fields. Transfer rows retain an empty `speedup`; `compute_speedup` and `end_to_end_speedup` explicitly distinguish the two comparisons. Numeric schema 2 timings should be compared with Phase 1 only with attention to batching and thread configuration.

## Architecture

```text
src/parallelbench/
  backends.py       NumPy / optional PyTorch CUDA operations, timing, CPU calibration
  engine.py         Trials, ordering, validation, statistics, instability warnings
  hardware.py       CPU, GPU, OS, Python, libraries, BLAS and thread metadata
  export.py         CSV and JSON reports
  visualization.py  Headless Matplotlib comparison graph
  cli.py            Command-line interface
```

The CLI coordinates small independent modules. PyTorch is optional; when installed, its CPU parallelism is recorded even in CPU-only runs. Tests cover configuration, correctness, export, plots, CLI behavior, and optional real CUDA execution.

## Benchmark methodology

### Inputs and correctness

For each matrix size and trial, deterministic seeds generate two contiguous float32 matrices with uniform values in `[-1, 1)`. Both backends receive identical values. Trial zero preserves Phase 1 seeds `[seed, size]`; later trials use `[seed, size, zero_based_trial]`. Changing execution order does not change inputs. The ordering RNG is independent of the input RNG, uses the configured seed, and the actual backend sequence is exported.

A float64 NumPy product serves as the validation reference. CPU and GPU outputs must pass `allclose` with default `rtol=1e-3`, `atol=1e-3`; tolerances can be changed explicitly. CUDA is also checked against the CPU output after both backends finish, including when CUDA executes first. These tolerances accommodate float32 accumulation and cancellation; they are not proof of correctness for every possible input or matrix size. Validation failures prevent exporting a successful report.

### CPU timing and threads

Input generation, reference computation, buffer allocation, library initialization, warm-up, calibration, validation, statistics, plotting, and export occur outside measured samples. CPU and GPU operations reuse buffers. Python cyclic garbage collection is temporarily disabled during measurements and restored, including on failure. This does not eliminate scheduler or operating system interference.

Timing uses the monotonic `perf_counter_ns` clock. Before CPU sampling, calibration doubles the batch size until a batch reaches the requested minimum duration or a cap of 1,048,576 calls is reached. The chosen batch size stays fixed for that size and trial. Each of the configured repetitions measures one batch. Its normalized latency is `elapsed_samples_ms / iterations_per_sample`, so CPU statistics summarize per-call batch averages. Original batch durations and batch sizes are exported alongside every normalized sample. No individual call timings within a batch are collected or inferred. Batching amortizes timer and loop overhead but can smooth short outliers and favor cache reuse; it represents repeated steady-state multiplication. It is not a cold-cache latency measurement. Use `--min-cpu-sample-ms 0` for individual calls. Calibration is a target, not a guarantee that every measured batch will reach it.

NumPy is **not assumed to be single-threaded**. `--cpu-threads N` temporarily limits loaded BLAS/OpenMP pools via `threadpoolctl` and PyTorch intra-op threads via `torch.set_num_threads`, then restores prior settings even when a run fails. Without this flag, library defaults remain in effect. PyTorch inter-op parallelism is recorded but never changed: its process-wide configuration cannot safely be repeatedly reset. The CPU backend uses NumPy rather than PyTorch; PyTorch intra-op threads do not select the NumPy BLAS thread count. Both are controlled and recorded separately.

The report records effective thread pools inside the control context, common thread environment variables, PyTorch intra-op/inter-op counts and parallel configuration, timer details, CPU model/logical CPU count, OS, architecture, Python/library versions, GPU name/memory/compute capability when used, CUDA runtime, and TF32 setting. Environment variables can also be set before launch, for example `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 parallelbench ...`. Thread pool sizes are configured limits, not a count of active workers observed during every multiplication. Drivers, power settings, and thermal conditions should be documented externally.

### CUDA computation, transfers, and end-to-end timing

CUDA samples synchronize the selected device immediately before starting and after completing the timed operation. Host wall-clock timings include dispatch and completion synchronization overhead; they are not kernel-only CUDA event measurements. CUDA is initialized and warmed outside timing. TF32 is disabled for the run and the original setting is restored afterward.

Four CUDA operations are measured separately:

1. Upload of both input matrices from host to device.
2. Matrix multiplication with resident GPU inputs.
3. Download of the output from device to host.
4. A complete upload → multiplication → download sequence, timed directly with one outer synchronization boundary before and after the sequence.

Host buffers use ordinary pageable memory and copies may block internally. All input/output buffers are preallocated. End-to-end timing includes transfer APIs, compute, and completion, but excludes allocation, initialization, generation, and validation. It must not be reconstructed by adding component medians: they are separate experiments. Operations within a GPU backend block retain the fixed order listed above; optional randomization applies to sizes and CPU/CUDA blocks, not to individual operations or samples.

The CPU already starts with host inputs and returns a host output, so its compute latency is also the CPU end-to-end baseline. There is no synthetic second CPU timing series.

- Compute-only speedup = CPU compute median / CUDA compute median.
- End-to-end speedup = CPU compute median / directly measured CUDA end-to-end median.

Both ratios use medians of all pooled normalized samples across trials. Per-trial ratios are also exported. An aggregate ratio is not an average of the trial ratios. A speedup above one favors CUDA for that defined measurement boundary, not necessarily for a complete application.

### Trials, statistics, and instability

Each trial recreates matrices and buffers, repeats warm-up and CPU calibration, and records `repetitions` samples for every operation. Trials share a process, CUDA context, library caches, and machine state; they are repeated measurement trials, not guaranteed statistically independent observations or isolated process runs. Without randomization, sizes follow CLI order and CPU precedes CUDA. `--randomize-order` shuffles both levels with a reproducible seed. It reduces a source of fixed ordering bias without controlling thermal drift or guaranteeing balanced order.

Per-trial and pooled statistics include median, mean, population standard deviation (`ddof=0`), MAD (unscaled median absolute deviation), IQR, minimum, maximum, and CV (standard deviation / mean). Every trial has equal configured sample count; pooled statistics weight its normalized samples equally, even if calibrated batch sizes differ. Aggregate minimum and maximum are extrema of normalized samples, not individual multiplication times hidden within batches. Plots show pooled medians with MAD error bars, not confidence intervals.

Instability warnings are heuristic: a group is marked unstable if its pooled sample CV, any within-trial sample CV, or the CV of trial medians exceeds `--instability-threshold`. Warnings appear on stderr and in exports; every raw sample, including outliers, remains untouched. The plot notes the number of report warnings. A single sample or trial cannot establish stability; a limited-assessment warning recommends at least two repetitions and two trials. An absent warning is not proof of stable hardware behavior. Increasing trials and repetitions and inspecting raw data remain necessary for research claims.

Background load, CPU affinity, clocks, power mode, thermal state, drivers, operating system, BLAS implementation, batch size, and thread limits affect comparisons. Run on stable AC power, repeat independent process executions, and document the environment. A fixed seed reproduces inputs and requested ordering, not timings. Do not compare machines without documenting their environments. This framework does not currently control CPU affinity, clock rates, or thermal state.

The float64 reference and live CPU/GPU buffers require extra memory. Start with modest sizes on a 4 GB GTX 1650. Trials execute sequentially and do not retain device arrays from prior sizes. There is no automatic memory-size selection. No experimental performance claims are included in this README.

## Research objectives and roadmap

Phase 1 establishes validated float32 matrix multiplication and reproducible exports. Phase 2 adds CPU batch timing, controlled thread counts, repeated trials, randomized ordering, directly measured CUDA end-to-end latency, and instability warnings. Future development will investigate:

1. CPU affinity, systematic BLAS thread scaling, and additional parallel CPU algorithms.
2. Custom CUDA kernels and comparison with library implementations.
3. Float64, float16, mixed precision, numerical error, and accuracy/performance tradeoffs.
4. Additional numerical algorithms, pinned-memory experiments, and complete application benchmarks.
5. Isolated process trials, confidence intervals, balanced ordering, and stronger hardware telemetry.
6. AI-assisted code optimization with correctness checks, recorded changes, and reproducible evaluation.

All optimizations should be evaluated with real measurements and explicit scientific limitations.
