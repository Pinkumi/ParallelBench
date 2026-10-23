"""Hardware-independent experimental presets; explicit CLI settings take precedence."""
PROFILES = {
    'quick': dict(sizes=(32, 64), warmup=2, repetitions=5, trials=2, cpu_threads=1,
                  numba_threads=(1, 2), min_cpu_sample_ms=1.0, randomize_order=True),
    'standard': dict(sizes=(128, 256, 512, 1024), warmup=3, repetitions=20, trials=3,
                     cpu_threads=1, numba_threads=(1, 2, 4, 8), min_cpu_sample_ms=5.0,
                     randomize_order=True),
    'research': dict(sizes=(128, 256, 512, 1024), warmup=5, repetitions=50, trials=7,
                     cpu_threads=1, numba_threads=(1, 2, 4, 8), min_cpu_sample_ms=10.0,
                     randomize_order=True),
}


def profile_settings(name):
    if name not in PROFILES:
        raise ValueError(f'Unknown profile: {name}')
    return dict(PROFILES[name])
