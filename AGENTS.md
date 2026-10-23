# Development instructions

- Keep code and documentation in English. Follow the user's preferred explanation language.
- Use src/parallelbench and small focused modules; avoid unnecessary abstractions.
- Run pytest and a small real CLI benchmark after changes; report failures and skips honestly.
- Never fabricate measurements or claim CUDA validation without a real CUDA execution.
- Keep allocation, initialization, validation, and warm-up outside timed compute loops.
- Synchronize CUDA before and after timing. Measure transfers separately.
- Use identical float32 inputs and validate against a higher precision reference.
- Preserve seeds, settings, environment metadata, and raw timing samples in exports.
- NumPy may use multithreaded BLAS; record thread pools and environment settings.
- Do not commit generated results or push without explicit user instructions.
- Version 1.0 adds optional CuPy CUDA C sources; include kernels/*.cu in built packages.
- Keep CUDA-event and host-wall samples separate; never mix their speedup baselines.
- Precision contracts must state input, multiplication, accumulation, and output dtypes.
- CPU FP16 storage emulation must remain explicitly labeled, including conversion buffers.
- Cross-device ratios require matching settings, algorithms, precision contracts, and timing boundaries.
- Never create desktop/RTX 3060 Ti results before a real desktop execution.
