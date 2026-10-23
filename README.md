# ParallelBench

Version **1.0.0** source candidate for reproducible numerical computing research. No GitHub release or tag is created automatically.

ParallelBench is an open-source research framework for reproducible comparisons of CPU and GPU numerical algorithms. Phases 1 and 2 benchmark dense square matrix multiplication using NumPy on the CPU and optional PyTorch CUDA on the GPU. Phase 3 adds custom sequential and parallel Numba kernels for CPU scalability experiments.

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
  engine.py         Shared trials, inputs, timing, validation, statistics, warnings
  numba_backend.py  Optional sequential / row-parallel float32 loop kernels
  scalability.py    Numba thread sweeps, two speedup baselines, efficiency
  hardware.py       CPU, GPU, OS, Python, libraries, BLAS and thread metadata
  export.py         CSV and JSON reports
  cupy_backend.py   Optional NVRTC kernels and explicit cuBLAS precision contracts
  kernels/matmul.cu Naive and shared-memory tiled CUDA C kernels
  experiments.py    CPU/custom CUDA/PyTorch/cuBLAS experiments
  precision.py      Accuracy metrics, precision policies, buffer accounting
  comparison.py     Strict JSON/CSV cross-run compatibility and ratios
  profiles.py       Reproducible quick, standard, research presets
  visualization.py  Headless comparisons and vector PDF export
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

## Phase 3: sequential and parallel CPU computing

### Installation and environment compatibility

Numba is optional, so the original CLI works without it. Install the scalability extra to use custom kernels and run their tests:

```bash
python -m pip install -e '.[dev,scalability]'
python -c "import sys, numba, numpy; print(sys.version); print(numba.__version__, numpy.__version__)"
```

This phase was developed with Python 3.14.4, NumPy 2.5.4, Numba 0.68.0, and llvmlite 0.50.0 on native Ubuntu. The optional extra selects Numba `>=0.67,<0.69`, whose supported dependency ranges include this combination. Consult the [official Numba version support table](https://github.com/numba/numba/blob/main/docs/source/user/installing.rst#version-support-information) when changing Python or NumPy. Pip resolves the compatible NumPy/llvmlite dependencies. Do not force incompatible packages with `--no-deps`. `psutil` supplies portable physical core counts; an unavailable physical count is recorded as null rather than inferred from logical CPUs.

### Sequential, parallel, and optimized implementations

- `numba_sequential` uses `@njit(parallel=False, fastmath=False)` and explicit nested `i-k-j` loops. It initializes each output row and accumulates float32 products into a preallocated float32 output. There are no matrix multiplication APIs or BLAS calls inside this kernel.
- `numba_parallel` uses `@njit(parallel=True, fastmath=False)` with `prange` over independent output rows. Arithmetic and C-contiguous input/output layouts match the sequential version. Every worker writes only its own rows, so no shared reduction or atomic operation is needed. Both kernels overwrite the output on every call.
- `cpu` retains the existing optimized NumPy/BLAS backend. NumPy may use multiple threads, vectorization, and cache blocking. It is not a sequential baseline.
- `torch_cpu` is an optional optimized CPU comparison, enabled with `--include-torch-cpu`. It requires installed PyTorch; requesting it without PyTorch fails explicitly.
- `cuda` retains the existing PyTorch CUDA backend, with synchronized computation, separate transfers, and directly measured end-to-end timing. No custom CUDA kernel is introduced.

JIT compilation converts Python loop code to native machine code. Before any benchmark or calibration samples, both Numba kernels compile using small C-contiguous float32 buffers and initialize the threading runtime. Shapes do not change the signatures used in this experiment. Each measured configuration then receives its own output allocation and actual-size warm-up, all outside timing. Compilation and process startup costs are not reported as execution latency. Timed Numba calls include Python dispatcher overhead and the output zeroing needed by the algorithm; they do not allocate result arrays. `fastmath=False` avoids enabling unsafe floating-point reassociation. NumPy/GPU libraries may still use different reduction orders.

### Thread sweep and reproducibility

`--scalability` activates Phase 3; omitting it preserves the original benchmark behavior and default sizes. The scalability defaults are matrix sizes `128 256 512 1024` and Numba thread counts `1 2 4 8 16`. Larger sizes can be passed with `--sizes`.

`--numba-threads` controls Numba separately from `--cpu-threads`, which sets the fixed NumPy/BLAS and optional PyTorch CPU comparison limit. The NumPy reference and optimized CPU measurements run within that control context; Numba configurations use their own temporary thread masks. Counts are checked against `numba.config.NUMBA_NUM_THREADS` before compilation. An explicit count above capacity is rejected, not silently reduced. If your environment supports fewer than 16 threads, pass a valid sweep, for example `--numba-threads 1 2 4 8`. Include one thread to measure the parallel-one-thread baseline. Prior Numba and library thread settings are restored after each configuration, including failures.

The Ryzen 7 5800H has eight physical cores and sixteen logical CPUs; SMT threads share core resources. The framework records detected hardware rather than relying on that target description. A sixteen-thread run is not assumed to be faster than an eight-thread run. Numba's `actual_numba_threads` field records `get_num_threads()` inside the measurement context: the configured active runtime mask, not an operating system observation of simultaneously busy workers. The runtime threading layer (such as TBB, OpenMP, or workqueue), capacity, requested mask, actual mask, `NUMBA_NUM_THREADS`, and `NUMBA_THREADING_LAYER` are recorded. Libraries and virtual machines may expose different capacities.

All implementations and thread counts for a size/trial receive equivalent input values. The existing seeded input generator, float64 NumPy oracle, tolerances, and timing utilities are reused. A separately computed float32 NumPy oracle also validates every configuration. The timed NumPy backend remains a real measurement rather than the untimed validation product. Optional randomization shuffles sizes and the full list of backend/thread configurations, and exports the actual sequence. It does not change seeds or input values. Each trial recreates inputs and buffers but shares the process/runtime caches as described in Phase 2.

CPU batching, raw batch durations, per-call normalization, statistics, and CV instability detection are reused. Thread counts remain distinct aggregation groups. Every trial and configuration preserves its samples, including outliers. The physical/logical CPU counts, thread configuration, matrix shape, float32 dtype, versions, OS, seed, warm-up, repetitions, and trial count are included in the JSON/CSV report. The original sample timing and variability limitations also apply to these experiments.

### Speedup and efficiency

For each matrix size, let `Tseq` be the custom sequential median, `Tpar(p)` the parallel median with `p` configured Numba threads, and `Tpar(1)` its one-thread median:

```text
speedup_vs_sequential(p)          = Tseq / Tpar(p)
parallel_efficiency(p)            = speedup_vs_sequential(p) / p
speedup_vs_parallel_one_thread(p) = Tpar(1) / Tpar(p)
efficiency_vs_parallel_one_thread = speedup_vs_parallel_one_thread(p) / p
```

The baselines are measured independently. A parallel kernel running with one thread can have different dispatch, scheduling, or compiler overhead from a sequential kernel. Speedup relative to `parallel(1)` must therefore not be interpreted as sequential speedup. Efficiency is a dimensionless ratio; multiply by 100 for a percentage. Superlinear values can arise from compiler, cache, and baseline effects and do not establish ideal hardware scaling. Only parallel Numba configurations receive a parallel efficiency value; optimized library operations do not have an equivalent known worker count merely because they expose thread limits.

Per-trial metrics use that trial's medians; aggregate metrics use pooled normalized sample medians, as in Phase 2. Explicit `speedup_vs_sequential` is provided for all compute backends. The legacy `speedup`, `compute_speedup`, and `end_to_end_speedup` fields keep their NumPy/CUDA meanings for `cpu` and `cuda`; `speedup` on Numba or PyTorch CPU rows refers to the custom sequential baseline. Use the explicit baseline fields for comparisons across implementations.

### Reproducing the experiments on Ubuntu / Ryzen 7 5800H

From the repository root, with its virtual environment already created:

```bash
.venv/bin/python -m pip install -e '.[dev,scalability]'
.venv/bin/python -m pytest -q -rs

# Small real CPU experiment; choose a new output directory on every invocation.
.venv/bin/python -m parallelbench --scalability --device cpu \
  --sizes 128 256 --numba-threads 1 2 4 8 16 --cpu-threads 1 \
  --warmup 2 --repetitions 5 --trials 3 --randomize-order --seed 42 \
  --output results/phase3-cpu-small-001

# Full CPU sweep, with a fixed eight-thread optimized-library comparison.
.venv/bin/python -m parallelbench --scalability --device cpu \
  --sizes 128 256 512 1024 --numba-threads 1 2 4 8 16 --cpu-threads 8 \
  --warmup 5 --repetitions 20 --trials 5 --randomize-order --seed 42 \
  --output results/phase3-5800h-cpu-001

# With existing CUDA-enabled PyTorch: include optimized PyTorch CPU and CUDA.
.venv/bin/python -m parallelbench --scalability --device cuda --include-torch-cpu \
  --sizes 128 256 512 1024 --numba-threads 1 2 4 8 16 --cpu-threads 8 \
  --warmup 5 --repetitions 20 --trials 5 --randomize-order --seed 42 \
  --output results/phase3-5800h-cuda-001
```

Use `--device auto` to include CUDA when available and record a skip reason otherwise. `--device cpu` never probes CUDA. The new mode produces schema version 3 with `experiment="cpu_scalability"`, retaining `results.json`, `results.csv`, `trial_results.csv`, and `comparison.png`. It also produces vector `comparison.pdf`. The five graph panels show latency vs threads, both speedup baselines, efficiency, custom kernels vs optimized CPU libraries, and CPU vs CUDA compute when available. Ideal linear speedup/efficiency references are theoretical lines, not measured results. MAD bars describe dispersion, not confidence intervals. A context panel records hardware, settings, and warning count.

The full sweep commands above are reproduction instructions, not a claim that every listed research run has already been performed. Check your local output reports for executed configurations. Tests without the Numba extra skip the Phase 3 module; CUDA tests skip when unavailable. Neither condition counts as a successful execution of those tests.

### Limitations

These simple loop kernels are not tiled or tuned to match optimized BLAS. An experiment compares algorithm/compiler/library combinations as well as parallelism. Report both Numba baselines, the actual environment, and NumPy's effective thread settings. Batches repeatedly reuse resident inputs and can smooth short outliers; samples are steady-state batch averages rather than cold-cache observations. Small matrices can be dominated by dispatch/scheduling costs. More logical threads may reduce performance. Random ordering and CV warnings do not control thermal state, clocks, affinity, other workloads, or SMT sharing. Independent process trials, confidence intervals, hardware counters, and cache-blocked kernels remain future work.

## Phases 4–7: CUDA, precision, multiple computers, and version 1.0

The supported algorithm is dense matrix multiplication. CLI runs use square sizes; the custom CUDA and Numba kernels also support compatible rectangular matrices through their Python backend APIs. There is no GUI, Docker requirement, cloud service, or automatic code optimizer. [The research report](docs/research_report.md) contains measured laptop results, archived inputs/metadata, actual graphs, and explicitly pending desktop experiments.

### Optional CUDA tooling

CPU-only installation and the original CLI remain available with `pip install -e '.[dev,scalability]'`. Custom kernels need one CuPy package matching the CUDA major version, a compatible NVIDIA driver, and NVRTC/runtime headers. For the existing CUDA 13 environment:

```bash
python -m pip install -e '.[dev,scalability,cuda,custom-cuda13]'
python -c "import cupy; cupy.show_config()"
```

Use `custom-cuda12` instead of `custom-cuda13` with CUDA 12. **Install only one CuPy CUDA variant.** Follow the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html). If a system toolkit or PyTorch's NVIDIA component wheels do not supply the runtime, NVRTC, and headers, CuPy's optional `cupy-cuda13x[ctk]` (or `cupy-cuda12x[ctk]`) installation supplies toolkit components. Do not replace working driver/toolkit versions merely to run CPU experiments. A system `nvcc` executable is not necessary: readable CUDA C source is shipped in the Python package and compiled with NVRTC outside measurements.

### Custom CUDA methodology

```bash
parallelbench --custom-cuda --profile quick --device cuda \
  --blocks 8x8 16x16 32x8 --tiles 8 16 32 --output results/laptop-kernels-quick-001
```

The naive kernel assigns one output element per thread. The tiled kernel cooperatively loads two shared-memory tiles, synchronizes all threads before and after using them, and zero-pads boundary tiles. No thread exits before a barrier. Both use FP32 fused multiply-add and validate against the same float64 reference. Tile sizes 8, 16, and 32 consume respectively 512, 2048, and 8192 shared-memory bytes per block. Thread/block dimensions, compiled kernel thread limits, registers, grids, and device shared-memory limits are checked or recorded. Unsupported configurations fail explicitly. A larger tile is not assumed to be faster.

The experiment compares custom naive/tiled kernels, NumPy, optional Numba sequential, PyTorch CUDA when available, and cuBLAS through CuPy. Compilation, initialization, allocation, reference generation, warm-up, and allocator cleanup are untimed. Live GPU buffers are checked against `--gpu-memory-fraction` (default 0.5 of currently free device memory). The check does not predict all library workspaces or prevent interference from other processes. Defaults are conservative; large sizes remain the user's choice.

Every GPU case records synchronized host-wall samples for upload, resident-input compute, download, and directly measured end-to-end execution. Separate compute samples use events on the same GPU stream. `timer=host_wall` and `timer=cuda_event` are different boundaries. Event elapsed time measures the GPU stream interval; on tiny operations it can include scheduling/host enqueue gaps and is not an instruction-level profiler measurement. Event objects are initialized before sampling. Events exclude transfer operations in this experiment. **CPU speedup is calculated only against host-wall GPU compute or end-to-end time.** GPU event speedups use other GPU event samples. The naive baseline is the first block configuration in the saved settings, not whichever configuration happens to execute first or finish fastest.

### Numerical precision contracts

```bash
parallelbench --precision-study --profile quick --device auto \
  --precisions float32 float16 float64 mixed --output results/laptop-precision-quick-001
```

The legacy `--precision float32` retains its original meaning. New precision sweeps use `--precisions`. All modes derive from the same float32 master inputs and float64 reference. FP16 modes quantize those input values, so errors include input quantization. FP64 represents the same master values exactly; the experiment does not create extra information by converting FP32 inputs to FP64.

| Backend / mode | Input storage | Multiplication / accumulation contract | Output |
| --- | --- | --- | --- |
| cuBLAS FP16 | FP16 | Pedantic FP16 compute / FP16 accumulation | FP16 |
| cuBLAS FP32 | FP32 | Pedantic FP32 compute / FP32 accumulation | FP32 |
| cuBLAS FP64 | FP64 | Pedantic FP64 compute / FP64 accumulation | FP64 |
| cuBLAS mixed | FP16 | Pedantic FP32 compute / FP32 accumulation | FP32 |
| NumPy FP32 / FP64 | Same as mode | Same precision GEMM | Same as mode |
| NumPy FP16 **emulation** | FP16 | Inputs preconverted to FP32; FP32 GEMM; timed FP16 output rounding | FP16 |
| NumPy mixed storage | FP16 | Inputs preconverted to FP32; FP32 GEMM | FP32 |

The CPU FP16 path is explicitly an emulation and does not claim native FP16 arithmetic or FP16 CPU speedup. Conversion buffers are included in memory accounting. Native GPU modes are capability-probed by a real untimed GEMM; cuBLAS `NOT_SUPPORTED` / `ARCH_MISMATCH` cases are recorded in `skipped`, without substituting another mode. Unexpected errors fail the run. Pedantic policies prioritize explicit arithmetic contracts rather than claiming the fastest possible tensor-core path. FP64 or FP16 support does not imply efficient execution on either GPU.

Accuracy records maximum absolute error, mean absolute error, relative Frobenius error, and RMSE against float64. For a zero reference, zero error gives relative error zero; nonzero error gives JSON null and `relative_error_defined=false`, avoiding division by zero or a fabricated finite error. Each trial preserves its accuracy record. Aggregate maximum absolute error is the maximum over trials; mean absolute and relative errors are means over trials; aggregate RMSE is the root mean of squared per-trial RMSE values.

Implementation sanity is checked against an entrywise rounding bound on quantized inputs, using `gamma_k = k*eps/(1-k*eps)` plus output rounding and float64 reference rounding. Machine epsilon rather than half epsilon makes this bound conservative. If `k*eps >= 1`, the standard bound is not useful: finiteness is checked and a warning/flag records the limitation. A separate `accuracy_within_tolerance` compares against the requested `--rtol`/`--atol` on original inputs. Exceeding that accuracy target produces a warning and retains the precision measurements; it does not imply that a correctly rounded low-precision algorithm was replaced or silently promoted.

Memory metrics report bytes of live host/device buffers, including CPU conversion scratch. They exclude runtime contexts, allocator reserves, and library workspaces; they are not peak process/VRAM telemetry. Precision graphs show host latency, separate CUDA-event elapsed time, numerical error, and buffer bytes. Performance ratios relative to FP32 match backend, operation, and timer.

### Profiles and equivalent multi-device experiments

| Profile | Sizes | Warm-up | Repetitions | Trials | CPU threads | CPU batch target |
| --- | --- | --- | --- | --- | --- | --- |
| `quick` | 32, 64 | 2 | 5 | 2 | 1 | 1 ms |
| `standard` | 128, 256, 512, 1024 | 3 | 20 | 3 | 1 | 5 ms |
| `research` | 128, 256, 512, 1024 | 5 | 50 | 7 | 1 | 10 ms |

Profiles enable seeded randomized order. Explicit flags override profile values, and all resolved settings are saved. Without `--profile`, previous defaults remain unchanged. Standard/research scalability profiles use Numba threads 1, 2, 4, 8, which fit both target systems' logical capacity; hardware capacity is still checked. Use the same explicit sweep on each machine. `--no-randomize-order` is available when fixed ordering is part of the experiment.

Environment records include physical/logical CPUs, affinity where available, system RAM, OS/WSL detection, Python and libraries, CUDA runtime, GPU memory, and NVIDIA driver/device inventory. Missing/inaccessible `nvidia-smi` is recorded as an inventory status, not a benchmark success or proof that no GPU exists. GPU compute information is also recorded through the active library. Thread counts represent configured limits rather than observations of simultaneously busy workers.

After creating a virtual environment and installing compatible optional GPU packages **on each computer**, run identical commands, changing only output directory names:

```bash
# Native Ubuntu laptop
parallelbench --custom-cuda --profile standard --device cuda --seed 42 \
  --blocks 8x8 16x16 32x8 --tiles 8 16 32 --output results/laptop-kernels-standard-001
parallelbench --precision-study --profile standard --device cuda --seed 42 \
  --precisions float32 float16 float64 mixed --output results/laptop-precision-standard-001
parallelbench --scalability --profile standard --device cpu --seed 42 \
  --numba-threads 1 2 4 8 --output results/laptop-cpu-standard-001

# Ubuntu WSL2 desktop, executed manually when it is accessible
parallelbench --custom-cuda --profile standard --device cuda --seed 42 \
  --blocks 8x8 16x16 32x8 --tiles 8 16 32 --output results/desktop-kernels-standard-001
parallelbench --precision-study --profile standard --device cuda --seed 42 \
  --precisions float32 float16 float64 mixed --output results/desktop-precision-standard-001
parallelbench --scalability --profile standard --device cpu --seed 42 \
  --numba-threads 1 2 4 8 --output results/desktop-cpu-standard-001

# Copy the real source reports to the same computer before comparing them.
parallelbench --compare results/laptop-kernels-standard-001/results.json \
  results/desktop-kernels-standard-001/results.csv --labels GTX1650 RTX3060Ti \
  --output results/cross-device-kernels-standard-001
```

WSL2 requires a working compatible Windows NVIDIA driver; no assumption is made that the desktop is currently accessible. For CPU-only systems use `--device cpu`; `--device auto` records omitted GPU cases when CUDA/CuPy is unavailable. `research` is a longer campaign; finish quick validation before running it. Old output directories are never overwritten.

The comparison loader accepts original aggregate JSON or `results.csv`, including earlier schemas. It checks algorithms, sizes, precision contracts, requested thread masks, operation, timer boundary, seed, warm-up, repetitions, trials, CPU limits, order policy, and batch methodology before reporting a ratio. Different profiles can match only if their concrete settings match. Calibrated batch sizes may differ by hardware; comparisons remain averages under the same batch calibration policy. Incompatible settings generate warnings and no ratios. Missing cases, invalid numerical results, and unstable sources are identified. A comparison of the same run's JSON and CSV is labeled a format roundtrip, not a hardware experiment. Changed library/OS/hardware environments are retained and warned about: cross-device ratios describe entire setups and do not isolate hardware alone.

`relative_speedup = baseline_latency / candidate_latency`; values above one favor the candidate. `latency_change_percent = (candidate_latency / baseline_latency - 1)*100`; negative values mean lower latency. Input files are never edited, and comparison reports retain source paths and SHA-256 hashes. **RTX 3060 Ti results and its comparison graph remain pending until that machine is actually measured.**

### Version 1.0 validation and release checklist

```bash
python -m pytest -q -rs
python -m build
parallelbench --version
```

The wheel and source distribution must include `parallelbench/kernels/matmul.cu`. CPU CI is configured for Ubuntu and Windows with Python 3.12/3.14; CUDA tests explicitly skip without a usable GPU. A CI definition is not a claim that those remote jobs have already run. Local CUDA tests require the optional CuPy variant and GPU access. No skipped GPU test counts as a successful CUDA execution.

Before publishing a release, run the standard/research profiles on both real computers, inspect variability/accuracy warnings, archive the reports, compare equivalent configurations, review CI/build results, choose an appropriate project license, and review/commit the changes. Creation of a GitHub tag or release is a separate manual action. No desktop measurement, universal speedup, or AI optimization result is claimed by this source candidate.

## Research objectives and roadmap

Phase 1 establishes validated float32 matrix multiplication and reproducible exports. Phase 2 adds CPU batch timing, controlled thread counts, repeated trials, randomized ordering, directly measured CUDA end-to-end latency, and instability warnings. Phase 3 adds validated Numba kernels, multicore thread sweeps, dual speedup baselines, and parallel efficiency. Future development will investigate:

1. CPU affinity, systematic BLAS thread scaling, and additional parallel CPU algorithms.
2. Custom CUDA kernels and comparison with library implementations.
3. Float64, float16, mixed precision, numerical error, and accuracy/performance tradeoffs.
4. Additional numerical algorithms, pinned-memory experiments, and complete application benchmarks.
5. Isolated process trials, confidence intervals, balanced ordering, and stronger hardware telemetry.
6. AI-assisted code optimization with correctness checks, recorded changes, and reproducible evaluation.

All optimizations should be evaluated with real measurements and explicit scientific limitations.
