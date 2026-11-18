"""One set of checks per model family, the same on every hardware.

Kitten TTS 0.8 and Kitten TTS 2 mini, nano and pico each run on several backends (the CPU, Apple silicon GPUs through
Metal, and NVIDIA GPUs through CUDA for mini/nano/pico). The options a model accepts, the errors it raises for a
setting that cannot work, the speed rule, and the languages it lists must not depend on the backend, so they are
checked here, before any file is downloaded, and in one wording for both families.
"""
import math
import numbers

NAMES = {'tts08': 'Kitten TTS 0.8', 'tts2_tiny': 'Kitten TTS 2 mini, nano and pico'}
# Backends each family runs on, and where (for the error text).
BACKENDS = {'tts08': ('cpu', 'metal'), 'tts2_tiny': ('cpu', 'metal', 'cuda')}
WHERE = {'tts08': 'the CPU and Apple silicon GPUs',
         'tts2_tiny': 'the CPU, NVIDIA GPUs and Apple silicon GPUs'}
GPU_BACKENDS = ('metal', 'cuda')
PRECISIONS = ('fp32', 'int8')
# Backend names that exist for another family; the error says which model runs there.
ELSEWHERE = {
    'cuda': " (NVIDIA GPUs run Kitten TTS 2 mini, nano and pico with backend='cuda', and Kitten TTS 2)",
    'rocm': " (AMD GPUs run Kitten TTS 2: KittenTTS('KittenML/kitten-tts-2', backend='rocm'))",
    'amd_gpu': " (AMD GPUs run Kitten TTS 2: KittenTTS('KittenML/kitten-tts-2', backend='rocm'))",
}
# Constructor options (besides backend and cache_dir) and synthesis options, per family.
MODEL_OPTIONS = {'tts08': ('precision', 'low_memory'), 'tts2_tiny': ('precision', 'revision', 'low_memory')}
SYNTH_OPTIONS = {'tts08': ('speed', 'normalize', 'language'), 'tts2_tiny': ('speed', 'normalize', 'language')}
# Speed: 1.0 is normal, 2.0 twice as fast. Kitten TTS 2 mini/nano/pico bound it (the C library checks the same range);
# Kitten TTS 0.8 takes any positive value, as kittenml does.
SPEED_RANGE = {'tts08': (0.0, math.inf), 'tts2_tiny': (0.25, 4.0)}


def _or_list(values):
    quoted = [repr(v) for v in values]
    return quoted[0] if len(quoted) == 1 else ', '.join(quoted[:-1]) + ' or ' + quoted[-1]


def check_backend(family, backend):
    """None (automatic) or one of the family's backends, in lower case; ValueError otherwise, the same text on
    every computer (it does not depend on what this computer has)."""
    if backend is None:
        return None
    choices = BACKENDS[family]
    if not isinstance(backend, str):
        raise ValueError(f"backend must be {_or_list(choices)} for {NAMES[family]}, not {backend!r}")
    key = backend.strip().lower()
    if key in choices:
        return key
    if key in ELSEWHERE:
        verb, them = ('runs', 'it') if family == 'tts08' else ('run', 'them')
        raise ValueError(f"{NAMES[family]} {verb} on {WHERE[family]}: backend={backend!r} is not available for "
                         f"{them}{ELSEWHERE[key]}. Leave out backend (or pass backend='cpu').")
    raise ValueError(f"backend must be {_or_list(choices)} for {NAMES[family]}, not {backend!r}")


def check_precision(family, precision):
    """None (the model's own precision), 'fp32' or 'int8'."""
    if precision is None or precision in PRECISIONS:
        return precision
    raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")


def check_gpu_rules(family, backend, precision, low_memory):
    """The rules for GPU backends, the same for both families: they run the float32 model, and low_memory is for the
    CPU."""
    if backend in GPU_BACKENDS and precision == 'int8':
        raise ValueError(f"backend={backend!r} runs the float32 model: leave out precision (or pass "
                         "precision='fp32'), or use the CPU for int8")
    if low_memory and backend in GPU_BACKENDS:
        raise ValueError(f"low_memory=True is for the CPU: leave out backend (or pass backend='cpu'), not "
                         f"backend={backend!r}")


def check_model_options(family, options):
    unknown = set(options) - set(MODEL_OPTIONS[family])
    if unknown:
        raise TypeError(f"{NAMES[family]} does not support: {', '.join(sorted(unknown))} (model options: "
                        f"{', '.join(MODEL_OPTIONS[family])}, backend, cache_dir)")


def check_synth_options(family, options):
    unknown = set(options) - set(SYNTH_OPTIONS[family])
    if unknown:
        raise TypeError(f"{NAMES[family]} does not support: {', '.join(sorted(unknown))} (synthesis options: "
                        f"{', '.join(SYNTH_OPTIONS[family])})")


def check_speed(family, speed):
    """speed as a float; ValueError (one wording) when it is not a number in the family's range."""
    lo, hi = SPEED_RANGE[family]
    rule = f"from {lo} to {hi}" if hi != math.inf else "above 0"
    try:
        value = float(speed)
    except (TypeError, ValueError):
        raise ValueError(f"speed must be a number {rule} (1.0 is normal), not {speed!r}") from None
    ok = lo <= value <= hi if hi != math.inf else (value > 0 and math.isfinite(value))
    if not ok:   # also refuses nan
        raise ValueError(f"speed must be {rule} (1.0 is normal), not {speed!r}")
    return value


def check_language(name, language, languages):
    """None or one of `languages`; ValueError in the wording Kitten TTS 2 mini/nano/pico use."""
    if language is not None and language not in languages:
        raise ValueError(f"{name} does not speak {language!r}; languages: {list(languages)}")
    return language


def check_chunk_chars(chunk_chars):
    """stream()'s chunk_chars, the same rule and wording for both families (and the C library, where 0 is None):
    None (the model's default: one sentence per chunk for Kitten TTS 0.8, 380 characters for mini, nano and pico) or a
    positive whole number of characters (whole sentences packed into chunks of at most that many characters). Returns
    None or the number; a value past the native chunker's int range means the same as its largest value."""
    if chunk_chars is None:
        return None
    if isinstance(chunk_chars, bool) or not isinstance(chunk_chars, numbers.Integral) or chunk_chars <= 0:
        raise ValueError("chunk_chars must be None (the model's default) or a positive whole number of characters, "
                         f"not {chunk_chars!r}")
    return min(int(chunk_chars), 2**31 - 1)
