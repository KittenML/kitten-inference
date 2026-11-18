import json
import os
from pathlib import Path
from ._deps import require as _require
_require()  # before the imports below, so one message names every missing package
from huggingface_hub import hf_hub_download
from .cpp_model import KittenTTS_1_Cpp, check_metal
from .preprocess import normalize_text


# kittenml's default model (KittenTTS() there loads nano 0.8, with its speed priors).
DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.8"
NATIVE_CONFIG_TYPES = {"CPP1", "NATIVE1", "MODEL_INFERENCE"}
NATIVE_WEIGHTS_REPO = "KittenML/meownn-models"
# Built-in models always load this exact revision of the native weights.
NATIVE_WEIGHTS_REVISION = "5e9bca0cc3a4e2a617f604b1333d5f34536c1eed"
VOICE_NAMES = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]
# speed_priors of the reference ONNX nano 0.8 models (config.json of KittenML/kitten-tts-nano-0.8-fp32 and
# -int8). The voice's prior multiplies the requested speed, as in kittenml 0.9.6. micro and mini have none.
NANO_08_SPEED_PRIORS = {
    "expr-voice-2-f": 0.8,
    "expr-voice-2-m": 0.8,
    "expr-voice-3-m": 0.8,
    "expr-voice-3-f": 0.8,
    "expr-voice-4-m": 0.9,
    "expr-voice-4-f": 0.8,
    "expr-voice-5-m": 0.8,
    "expr-voice-5-f": 0.8,
}
DEFAULT_VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}
BUILTIN_NATIVE_MODELS = {
    # An earlier name for the nano model: the same 0.8 weights and voices, so the same speed priors.
    "kittenml/kitten-tts-nano-0.1": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
        "speed_priors": NANO_08_SPEED_PRIORS,
    },
    "kittenml/kitten-tts-nano-0.8": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
        "speed_priors": NANO_08_SPEED_PRIORS,
    },
    "kittenml/kitten-tts-nano-0.8-fp32": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
        "speed_priors": NANO_08_SPEED_PRIORS,
    },
    "kittenml/kitten-tts-nano-0.8-int8": {
        "arch_file": "kitten_int8_15m_arch.json",
        "weights_file": "kitten_int8_15m.bin",
        "voice_dir": "voices_kitten_15m",
        "speed_priors": NANO_08_SPEED_PRIORS,
    },
    "kittenml/kitten-tts-micro-0.8": {
        "arch_file": "kitten_fp32_40m_arch.json",
        "weights_file": "kitten_fp32_40m.bin",
        "voice_dir": "voices_kitten_40m",
    },
    "kittenml/kitten-tts-micro-0.8-int8": {
        "arch_file": "kitten_int8_40m_arch.json",
        "weights_file": "kitten_int8_40m.bin",
        "voice_dir": "voices_kitten_40m",
    },
    "kittenml/kitten-tts-mini-0.8": {
        "arch_file": "kitten_fp32_80m_arch.json",
        "weights_file": "kitten_fp32_80m.bin",
        "voice_dir": "voices_kitten_80m",
    },
    "kittenml/kitten-tts-mini-0.8-int8": {
        "arch_file": "kitten_int8_80m_arch.json",
        "weights_file": "kitten_int8_80m.bin",
        "voice_dir": "voices_kitten_80m",
    },
}


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None, backend=None, precision=None, low_memory=False):
        """Initialize KittenTTS with a native model.
        
        Args:
            model_name: Hugging Face repository ID, local model directory, or a
                KittenML model name such as ``kitten-tts-mini-0.8``.
            cache_dir: Directory to cache downloaded files
            backend: ``"cpu"`` or ``"metal"``. Defaults to Metal on Apple
                Silicon when a Metal device is available, except for int8
                models and low_memory, and to the CPU otherwise.
            precision: ``"fp32"`` or ``"int8"``; default: the model name's
                (``kitten-tts-mini-0.8`` is fp32, ``kitten-tts-mini-0.8-int8``
                int8). ``precision="int8"`` with ``kitten-tts-mini-0.8`` loads
                ``kitten-tts-mini-0.8-int8``.
            low_memory: ``True`` uses the least memory (the CPU only; working
                memory is freed after each call), at some cost in speed.

        Backend, precision and low_memory are checked before any download, with the same rules and messages as Kitten
        TTS 2 mini, nano and pico (kitten_inference/_family.py).
        """
        from .. import _family
        backend = _family.check_backend("tts08", backend)
        precision = _family.check_precision("tts08", precision)
        repo_id = _with_precision(_resolve_model_id(model_name), precision)
        spec = None if os.path.isdir(repo_id) else _builtin_native_spec(repo_id)
        # A built-in model's precision is known from its name; others are checked once their graph is read.
        known = precision or (spec and ("int8" if "int8" in spec["arch_file"] else "fp32"))
        _family.check_gpu_rules("tts08", backend, known, low_memory)
        if backend == "metal":
            check_metal()   # before the download, as for Kitten TTS 2 mini, nano and pico
        self.model = download_from_huggingface(repo_id=repo_id, cache_dir=cache_dir, backend=backend,
                                               precision=precision, low_memory=low_memory)

    @property
    def default_voice_name(self):
        """The default voice as list_voices() names it (Leo for the built-in models)."""
        target = DEFAULT_VOICE_ALIASES["Leo"]
        if self.model.voice_aliases.get("Leo") == target and "Leo" in self.model.all_voice_names:
            return "Leo"
        return target
    
    def normalize_text(self, text, locale="en-US", return_spans=False):
        """Normalize text for TTS without generating audio."""
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        return self.model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False, trim_lead=True,
                        chunk_chars=None):
        """Generate audio as a stream of chunks (chunk_chars None, the default: one sentence per chunk; N packs whole
        sentences into chunks of at most N characters, at most 400).

        trim_lead (default True): the first chunk starts at its first audible sample (less 20 ms), not after the
        model's ~0.4 s of silence; every kept sample is unchanged.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        yield from self.model.generate_stream(text, voice=voice, speed=speed, clean_text=clean_text,
                                              trim_lead=trim_lead, chunk_chars=chunk_chars)

    def generate_to_file(
        self,
        text,
        output_path,
        voice="expr-voice-5-m",
        speed=1.0,
        sample_rate=24000,
        clean_text=True,
    ):
        """Generate audio from text and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        return self.model.generate_to_file(
            text,
            output_path,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
            clean_text=clean_text,
        )
    
    @property
    def available_voices(self):
        """Get list of available voices."""
        return self.model.all_voice_names


def _resolve_local_file(root, filename):
    return filename if os.path.isabs(filename) else os.path.join(root, filename)


def _download_file(repo_id, filename, cache_dir, revision=None):
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, revision=revision)


def _builtin_arch_path(filename):
    return str(Path(__file__).resolve().parent.joinpath("model_defs", filename))


def _builtin_native_spec(repo_id):
    return BUILTIN_NATIVE_MODELS.get(repo_id.lower())


def _resolve_model_id(model_name):
    if os.path.isdir(model_name):
        return model_name
    if "/" not in model_name:
        return f"KittenML/{model_name}"
    return model_name


def _with_precision(repo_id, precision):
    """The built-in model of the requested precision: kitten-tts-mini-0.8 + "int8" -> kitten-tts-mini-0.8-int8.
    A name that already says its precision must agree with precision=."""
    if precision is None or os.path.isdir(repo_id) or _builtin_native_spec(repo_id) is None:
        return repo_id
    spec = _builtin_native_spec(repo_id)
    current = "int8" if "int8" in spec["arch_file"] else "fp32"
    if current == precision:
        return repo_id
    name = repo_id.lower()
    if name.endswith(("-int8", "-fp32")):
        raise ValueError(f"{repo_id!r} is the {current} model: leave out precision, or use the model name without "
                         f"'-{current}' and precision={precision!r}")
    other = repo_id + "-int8" if precision == "int8" else None
    if other is None or _builtin_native_spec(other) is None:
        raise ValueError(f"{repo_id!r} has no {precision} version")
    return other


def _config_file(config, primary, *fallbacks):
    for key in (primary,) + fallbacks:
        value = config.get(key)
        if value:
            return value
    raise ValueError(f"Native KittenTTS config is missing required field {primary!r}.")


def _load_local_model(repo_dir, backend=None, **model_options):
    config_path = os.path.join(repo_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return _instantiate_native_model(
        config=config,
        get_file=lambda filename: _resolve_local_file(repo_dir, filename),
        backend=backend,
        **model_options,
    )


def _load_builtin_native_model(repo_id, spec, cache_dir=None, backend=None, **model_options):
    voice_paths = {
        voice: _download_file(
            NATIVE_WEIGHTS_REPO,
            f"{spec['voice_dir']}/{voice}.bin",
            cache_dir,
            NATIVE_WEIGHTS_REVISION,
        )
        for voice in VOICE_NAMES
    }

    return KittenTTS_1_Cpp(
        arch_path=_builtin_arch_path(spec["arch_file"]),
        weights_path=_download_file(
            NATIVE_WEIGHTS_REPO, spec["weights_file"], cache_dir, NATIVE_WEIGHTS_REVISION
        ),
        voice_paths=voice_paths,
        speed_priors=spec.get("speed_priors", {}),
        voice_aliases=DEFAULT_VOICE_ALIASES,
        backend=backend,
        style_index="text_length",  # the reference ONNX path picks the style row by text length
        sample_rate=24000,
        **model_options,
    )


def _instantiate_native_model(config, get_file, backend=None, **model_options):
    config_type = config.get("type")
    if config_type not in NATIVE_CONFIG_TYPES:
        if config_type in {"ONNX1", "ONNX2"}:
            raise ValueError(
                "kitten_inference.tts08 runs native models only. "
                "Use a native model repo/config with type CPP1, arch_file, and weights_file."
            )
        raise ValueError(f"Unsupported model type: {config_type!r}.")

    arch_path = get_file(_config_file(config, "arch_file", "arch"))
    weights_path = get_file(_config_file(config, "weights_file", "weights"))

    voices_path = None
    if config.get("voices"):
        voices_path = get_file(config["voices"])

    voice_paths = {
        voice: get_file(filename)
        for voice, filename in (config.get("voice_files") or {}).items()
    }

    return KittenTTS_1_Cpp(
        arch_path=arch_path,
        weights_path=weights_path,
        voices_path=voices_path,
        voice_paths=voice_paths,
        speed_priors=config.get("speed_priors", {}),
        voice_aliases=config.get("voice_aliases", {}),
        backend=backend,
        style_index=config.get("style_index", "token_count"),
        sample_rate=config.get("sample_rate", 24000),
        **model_options,
    )


def download_from_huggingface(repo_id=DEFAULT_MODEL, cache_dir=None, backend=None, **model_options):
    """Download model files from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Directory to cache downloaded files
        
    Returns:
        KittenTTS_1_Cpp: Instantiated model ready for use
    """
    if os.path.isdir(repo_id):
        return _load_local_model(repo_id, backend=backend, **model_options)

    builtin_spec = _builtin_native_spec(repo_id)
    if builtin_spec is not None:
        return _load_builtin_native_model(
            repo_id=repo_id,
            spec=builtin_spec,
            cache_dir=cache_dir,
            backend=backend,
            **model_options,
        )

    # Download config file first
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return _instantiate_native_model(
        config=config,
        get_file=lambda filename: _download_file(repo_id, filename, cache_dir),
        backend=backend,
        **model_options,
    )


def get_model(repo_id=DEFAULT_MODEL, cache_dir=None, backend=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend)
