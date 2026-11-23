"""Kitten TTS 2 mini / nano / pico on the native engine (the TTS 0.8 graph engine): the CPU, Metal on Apple
silicon, or CUDA on NVIDIA GPUs.

The official files are downloaded from Hugging Face (KittenML/kitten-tts-2-{mini,nano,pico}) and converted for the
engine on first use (kitten_inference.tts2_tiny.convert, numpy only); the converted files are cached next to the
Hugging Face cache. Synthesis follows kittenml 0.9.8 (kittentts2_tiny/model.py): the same text front end, token ids,
voice style row, chunk trimming, fades and pauses between chunks.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np


class _LazyConvert:
    """The converter module, imported on first use, so `python -m kitten_inference.tts2_tiny.convert` does not find it
    already imported (runpy's RuntimeWarning)."""

    def __getattr__(self, name):
        from . import convert
        return getattr(convert, name)


_convert = _LazyConvert()

SAMPLE_RATE = 24000
GAP_S = 0.15
END_TOKEN = 10
REPOS = {
    "kittenml/kitten-tts-2-mini": "KittenML/kitten-tts-2-mini",
    "kittenml/kitten-tts-2-nano": "KittenML/kitten-tts-2-nano",
    "kittenml/kitten-tts-2-pico": "KittenML/kitten-tts-2-pico",
}
# Built-in models load this exact revision of the official files (the converter was checked against it).
REVISIONS = {
    "KittenML/kitten-tts-2-mini": "c01797caec3967b400da457656e011eb81f530a4",
    "KittenML/kitten-tts-2-nano": "54526b8d8e45333cd720d3a082c106aacf416064",
    "KittenML/kitten-tts-2-pico": "da8c9d029e16d9302f1827adb303d92316e25f73",
}
PRECISIONS = ("fp32", "int8")
# low_memory=True: the least memory, at the cost of speed. The C library's KT08_MEMORY_EXTREME_LOW (the phones'
# default): working memory and per-layer scratch freed after each call, and direct convolutions instead of Winograd's
# larger transformed weights, so the audio differs from the default mode by rounding.
LOW_MEMORY_MODE = "extreme-low-memory"
# speed: 1.0 is normal, 2.0 twice as fast. The range keeps one chunk's audio (and memory) bounded: at speed 0.01 one
# short sentence needs 145 s of audio and 1.2 GB. The C library (kt08_synthesize_tiny) checks the same range.
SPEED_MIN, SPEED_MAX = 0.25, 4.0
# stream()'s default chunk size in characters (chunk_chars=None): KittenTTS 2's own chunk_chars default (kittenml
# kittentts2/text.py CHUNK_CHARS). The C library's kt08_stream_open uses the same for chunk_chars 0
# (KT08_STREAM_TINY_CHUNK_CHARS).
STREAM_CHUNK_CHARS = 380


def check_speed(speed):
    from ..tts08._family import check_speed as _check   # one rule and wording for both small families
    return _check("tts2_tiny", speed)
# English text goes through spaCy, regex and addict, Japanese text (mini) through fugashi, jaconv and addict (the code of
# misaki 0.9.4 and num2words 0.5.14 they use is vendored in frontend/). kitten-inference installs them on CPython 3.10 to
# 3.14 (their markers in pyproject.toml): on Python 3.9 pip picks spaCy 3.8.17, which ships only as source and fails to
# build, and newer Pythons have no packages yet. fugashi has no package for Windows on Arm. The other languages need
# none of them.
FRONT_END_PYTHONS = ((3, 10), (3, 14))
FRONT_END_MODULES = ("spacy", "regex", "addict", "fugashi", "jaconv")


def front_end_unavailable(missing, lang="en", version=None, system=None, machine=None):
    """The error text when a package of the English or Japanese text front end cannot be imported."""
    import platform
    import sys
    major, minor = (version or sys.version_info)[:2]
    system, machine = system or platform.system(), machine or platform.machine()
    language = {"en": "English", "ja": "Japanese"}.get(lang, f"{lang!r}")
    (lo_major, lo_minor), (hi_major, hi_minor) = FRONT_END_PYTHONS
    if not FRONT_END_PYTHONS[0] <= (major, minor) <= FRONT_END_PYTHONS[1]:
        return (f"{language} text with Kitten TTS 2 mini, nano and pico is not supported on Python {major}.{minor}: "
                f"its text front end (spaCy) installs only on Python {lo_major}.{lo_minor} to "
                f"{hi_major}.{hi_minor}. The other languages work on this Python.")
    if missing.split(".")[0] == "fugashi" and system == "Windows" and machine.upper() == "ARM64":
        return ("Japanese text with Kitten TTS 2 mini is not supported on Windows on Arm: its Japanese tokenizer "
                "(fugashi) has no package for it. The other languages work here.")
    return (f"{language} text with Kitten TTS 2 mini, nano and pico needs {missing}, which kitten-inference installs on "
            f"Python {major}.{minor} ({system} {machine}), but it is missing: "
            "pip install --force-reinstall kitten-inference")


def repo_id(model):
    """'KittenML/kitten-tts-2-nano' or 'kitten-tts-2-nano' (any case) -> the Hugging Face repository; None if the
    name is not one of the three models."""
    name = str(model).strip().rstrip("/").lower()
    if "/" not in name:
        name = "kittenml/" + name
    return REPOS.get(name)


def is_tiny_model(model):
    if repo_id(model) is not None:
        return True
    path = str(model)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        try:
            with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
                return json.load(f).get("type") == "KITTEN2_TINY"
        except (OSError, ValueError):
            return False
    return False


def _default_cache_root(cache_dir):
    if cache_dir:
        return os.path.join(cache_dir, "kitten_inference_tts2_tiny")
    from huggingface_hub import constants
    return os.path.join(constants.HF_HOME, "kitten_inference", "tts2_tiny")


# backend="metal" runs the graph in two parts. The CPU part ends after the F0 and energy predictor (and their stride-2
# convolutions, dec_n_conv). It holds every step that sets the frame counts (the duration sum, t_dur), so the audio has
# exactly the CPU's length; the frame-by-frame LSTM over the length-regulated frames, which on Metal costs two GPU
# dispatches per frame; and the F0 track, which must be bit-identical to the CPU's: the harmonic source integrates F0
# into a phase over every sample, so a one-ulp F0 difference became a 30-40 dB signal-to-noise loss in the audio, and
# the excitation noise is seeded from F0's bits. The rest (decoder and vocoder) runs on Metal.
METAL_CUT = "dec_n_conv"


def split_for_metal(arch_path, end=METAL_CUT):
    """Writes (once) and returns the paths of two engine graphs cut from one converted architecture file after the
    node `end`: the first part (run on the CPU) and the rest (run on Metal). The second graph takes the first one's
    outputs as inputs, by name; the first also outputs t_dur. Both read the same weights file."""
    base = arch_path[:-len(".json")] if arch_path.endswith(".json") else arch_path
    text_path, dec_path = f"{base}_{end}_cpu.json", f"{base}_{end}_metal.json"
    if os.path.isfile(text_path) and os.path.isfile(dec_path):
        return text_path, dec_path
    with open(arch_path, encoding="utf-8") as f:
        arch = json.load(f)
    nodes = arch["nodes"]
    cut = next((i for i, n in enumerate(nodes) if n["name"] == end), None)
    if cut is None:
        raise ValueError(f"{arch_path}: no node {end!r} to split the graph at")
    head, tail = nodes[:cut + 1], nodes[cut + 1:]
    made_by_tail = {o for n in tail for o in n["outputs"]}
    made_by_head = {o for n in head for o in n["outputs"]}
    inputs = {i["name"]: i for i in arch["inputs"]}
    needed = []
    for n in tail:
        for t in n["inputs"]:
            if t not in made_by_tail and t not in needed:
                needed.append(t)
    passed = [t for t in needed if t in made_by_head]
    if any(t not in made_by_head and t not in inputs for t in needed):
        raise ValueError(f"{arch_path}: cannot split the graph at {end!r}")
    if "t_dur" not in passed:
        passed.append("t_dur")
    by_output = {o: n for n in head for o in n["outputs"]}
    widths = {t: _width(t, by_output, inputs) for t in passed}
    text = dict(arch, outputs=[{"name": t} for t in passed], nodes=head)
    dec = dict(arch, inputs=[inputs[t] for t in needed if t in inputs]
               + [{"name": t, "shape": [1, 0, 1, widths[t]], "dtype": "float32"} for t in passed if t in needed],
               outputs=[o for o in arch["outputs"] if o["name"] != "t_dur"], nodes=tail)
    for path, graph in ((text_path, text), (dec_path, dec)):
        tmp = _convert._tmp_name(path)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(graph, f)
        os.replace(tmp, path)
    return text_path, dec_path


def _width(tensor, by_output, inputs):
    """Channel count of a tensor of the text stage, from the node that makes it (the ops the converter uses there)."""
    if tensor in inputs:
        return int(inputs[tensor]["shape"][-1])
    n = by_output[tensor]
    op = n["op"]
    if op == "lstm_fp32":
        return int(n["hidden_size"]) * (2 if n.get("bidirectional") else 1)
    if op == "concat1d_fp32":
        return _width(n["inputs"][0], by_output, inputs) + int(n["C2"])
    if op == "ada_layer_norm_fp32":
        return int(n["C"])
    if op in ("seqgemm_fp32", "conv1d_fp32"):
        return int(n["C_out"])
    if op == "sum_channels_fp32":
        return 1
    if op in ("layernorm_fp32", "leaky_relu_fp32", "sigmoid_fp32", "scale_fp32", "add_fp32", "length_regulate_fp32"):
        return _width(n["inputs"][0], by_output, inputs)
    raise ValueError(f"cannot tell the channel count of {tensor!r} ({op})")


class KittenTTS2Tiny:
    """A Kitten TTS 2 mini, nano or pico model on the native engine: by default Metal on Apple silicon (float32), the
    CPU elsewhere and for int8; backend="cpu", "metal" or "cuda" (NVIDIA GPUs, float32) chooses.

        m = KittenTTS2Tiny("KittenML/kitten-tts-2-nano")
        audio = m.generate("Guten Morgen.", language="de")     # the language's first voice
        audio = m.generate("Hello there.", voice="Eleanor")    # a voice name (any case) or description
    """

    def __init__(self, model="KittenML/kitten-tts-2-nano", backend=None, cache_dir=None, precision="fp32",
                 revision=None, low_memory=False):
        # The same rules and messages as Kitten TTS 0.8 (kitten_inference/tts08/_family.py), before any download.
        from ..tts08 import _family
        backend = _family.check_backend("tts2_tiny", backend)
        precision = _family.check_precision("tts2_tiny", precision) or "fp32"
        _family.check_gpu_rules("tts2_tiny", backend, precision, low_memory)
        self.precision, self.cache_dir, self._backend_req = precision, cache_dir, backend
        self.low_memory = bool(low_memory)
        self._cuda = None
        if backend == "cuda":   # before any download: ValueError with the reason when there is no NVIDIA GPU or driver
            from . import _cuda
            self._cuda = _cuda.engine()
        if backend == "metal":  # also before any download
            from ..tts08 import cpp_model as _cm
            _cm.check_metal()
        self.repo = repo_id(model)
        if self.repo is None:
            if not is_tiny_model(model):
                raise ValueError(f"{model!r} is not a Kitten TTS 2 mini, nano or pico model")
            self.local_dir, self.revision = os.path.abspath(str(model)), "local"
        else:
            self.local_dir, self.revision = None, revision or REVISIONS[self.repo]
        config_path = self._fetch("config.json")
        if self.local_dir is None:
            # A branch or tag name (e.g. "main") can move: key the converted files by the commit it resolved to.
            parts = os.path.normpath(config_path).split(os.sep)
            if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
                self.revision = parts[parts.index("snapshots") + 1]
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        if self.config.get("type") != "KITTEN2_TINY":
            raise ValueError(f"{model!r}: config type {self.config.get('type')!r} is not KITTEN2_TINY")
        with open(self._fetch(self.config.get("vocab", "vocab.json")), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.name = self.config.get("name", self.repo or os.path.basename(self.local_dir))
        self.sample_rate = int(self.config.get("sample_rate", SAMPLE_RATE))
        self.languages = list(self.config["languages"])
        self.default_language = self.config.get("default_language", self.languages[0])
        self._voices = {lang: list(vs) for lang, vs in self.config["voices"].items()}
        self._frontend = None
        self._sessions, self._packs = {}, {}
        self._lock = threading.Lock()          # one synthesis at a time per model (engine buffers, espeak)
        self._load_lock = threading.Lock()
        from ..tts08 import cpp_model as _cm   # sets up espeak-ng's data path and loads the engine
        self._cm = _cm
        _cm._check_glibc()                     # espeak-ng needs glibc 2.29+; say so clearly
        self._mi = _cm._import_engine()
        self.backend = None

    # ---- files --------------------------------------------------------------------------------------------------
    def _fetch(self, filename):
        if self.local_dir is not None:
            return os.path.join(self.local_dir, filename)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.repo, filename=filename, revision=self.revision, cache_dir=self.cache_dir)

    def _g2p_dir(self):
        d = self.config.get("g2p_dir", "g2p")
        if self.local_dir is not None:
            return os.path.join(self.local_dir, d)
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=self.repo, revision=self.revision, cache_dir=self.cache_dir,
                                 allow_patterns=[f"{d}/*", f"{d}/**"])
        return os.path.join(root, d)

    def _converted(self, model_key):
        """Paths of the converted engine files of one model, converting them on first use."""
        if self.local_dir is not None:   # never written into the user's folder: cached by path and file times
            import hashlib
            src = self.config["models"][model_key]["safetensors"]
            stamp = f"{self.local_dir}|{os.path.getmtime(os.path.join(self.local_dir, 'config.json'))}|" \
                    f"{os.path.getmtime(os.path.join(self.local_dir, src))}"
            out = os.path.join(_default_cache_root(self.cache_dir), "local",
                               hashlib.sha256(stamp.encode()).hexdigest()[:16], f"v{_convert.FORMAT_VERSION}")
        else:
            out = os.path.join(_default_cache_root(self.cache_dir), self.repo.replace("/", "--"), self.revision,
                               f"v{_convert.FORMAT_VERSION}")
        p = self.precision
        arch, weights = (os.path.join(out, f"{model_key}_{p}_arch.json"), os.path.join(out, f"{model_key}_{p}.bin"))
        # Only the precision asked for is converted (less time, memory and disk on first use). "<key>.done" marks a
        # cache an earlier version wrote with both precisions.
        done = os.path.join(out, f"{model_key}_{p}.done")
        if not os.path.isfile(done) and not os.path.isfile(os.path.join(out, f"{model_key}.done")):
            entry = self.config["models"][model_key]
            src = self._fetch(entry["safetensors"])
            _convert.convert_model(self.config, model_key, src, out, int8=p == "int8", fp32=p == "fp32")
            tmp = _convert._tmp_name(done)
            with open(tmp, "w") as f:
                f.write("ok\n")
            os.replace(tmp, done)
        return arch, weights

    # ---- engine -------------------------------------------------------------------------------------------------
    def _session(self, lang):
        key = self.config["languages"][lang]["model"]
        with self._load_lock:
            if key not in self._sessions:
                arch, weights = self._converted(key)
                if self.backend is None:
                    # Default: Metal on Apple silicon for float32 (faster than the macOS CPU engine for all three
                    # models), as for Kitten TTS 0.8; the CPU everywhere else and for int8.
                    self.backend = self._backend_req or ("cpu" if self.low_memory else
                                                         self._cm._default_backend(self._mi, arch))
                if self.backend == "metal":
                    # Text, durations, the frame LSTM and F0/energy on the CPU, the rest on Metal (see METAL_CUT): the
                    # frame counts and F0 are then exactly the CPU's (on Metal alone, nano and mini rounded some
                    # durations differently).
                    text_arch, dec_arch = split_for_metal(arch)
                    with open(dec_arch, encoding="utf-8") as f:
                        dec_inputs = frozenset(i["name"] for i in json.load(f)["inputs"])
                    with self._cm._ENGINE_GATE.use(self._mi, "cpu"):
                        text = self._mi.InferenceModel(text_arch, weights)
                    with self._cm._ENGINE_GATE.use(self._mi, "metal"):
                        dec = self._mi.InferenceModel(dec_arch, weights)
                    self._sessions[key] = (text, dec, dec_inputs, "metal")
                elif self.backend == "cuda":
                    # The whole graph on the GPU, in the separate NVIDIA GPU library (_cuda.py), which serialises
                    # its own calls.
                    self._sessions[key] = self._cuda.InferenceModel(arch, weights)
                else:
                    with self._cm._ENGINE_GATE.use(self._mi, self.backend):
                        # low_memory: free working memory after each call
                        self._sessions[key] = (self._mi.InferenceModel(arch, weights, LOW_MEMORY_MODE)
                                               if self.low_memory else self._mi.InferenceModel(arch, weights))
            return self._sessions[key]

    def _pack(self, lang, key):
        path = self.config["languages"][lang]["voices"]
        if path not in self._packs:
            self._packs[path] = _convert.voice_packs(self._fetch(path))
        return self._packs[path][key]

    def _run(self, lang, tokens, style, speed):
        """tokens -> (waveform float32, frames per token int64), the reference's (wav, durations)."""
        session = self._session(lang)
        speed = check_speed(speed)
        feed = {"phoneme_ids": np.asarray(tokens, dtype=np.float32),
                "style_dec": np.ascontiguousarray(style[0, :128], dtype=np.float32),
                "style_pred": np.ascontiguousarray(style[0, 128:], dtype=np.float32)}
        if isinstance(session, tuple):  # Metal: (CPU session, Metal session, the Metal graph's inputs, its backend)
            text, dec, dec_inputs, dec_backend = session
            with self._cm._ENGINE_GATE.use(self._mi, "cpu"):
                text.set_speed(speed)
                mid = text.forward(feed)
            feed = {k: v for k, v in {**feed, **mid}.items() if k in dec_inputs}
            with self._cm._ENGINE_GATE.use(self._mi, dec_backend):
                dec.set_speed(speed)
                out = dec.forward(feed)
            out["t_dur"] = mid["t_dur"]
        elif self.backend == "cuda":
            session.set_speed(speed)
            out = session.forward(feed)
        else:
            with self._cm._ENGINE_GATE.use(self._mi, self.backend):
                session.set_speed(speed)
                out = session.forward(feed)
        wav = np.asarray(out["t_audio"], dtype=np.float32).reshape(-1)
        d = np.asarray(out["t_dur"], dtype=np.float32).reshape(-1) * (np.float32(1.0) / np.float32(speed))  # as the engine
        return wav, np.maximum(1, np.floor(d + np.float32(0.5))).astype(np.int64)

    # ---- voices -------------------------------------------------------------------------------------------------
    @property
    def available_voices(self):
        """Voice names, per language: {language: [name, ...]}."""
        return {lang: [v["name"] for v in vs] for lang, vs in self._voices.items()}

    def voices(self, language=None):
        """The voices of one language (default: all), as dicts with "key", "name", "description" and "language"."""
        langs = [self._language(language)] if language else self.languages
        return [dict(v, language=lang) for lang in langs for v in self._voices[lang]]

    @property
    def default_voice(self):
        return self._voices[self.default_language][0]["name"]

    def _language(self, language):
        if language is None:
            return self.default_language
        if language not in self.config["languages"]:
            raise ValueError(f"{self.name} does not speak {language!r}; languages: {self.languages}")
        return language

    def resolve_voice(self, voice=None, language=None):
        """-> (language, voice key), as kittenml: a name in any case, a description, or None (the language's first
        voice); without `language`, a name only one language has decides it; a name several share means the
        default language's voice."""
        if voice is None:
            lang = self._language(language)
            return lang, self._voices[lang][0]["key"]
        langs = [self._language(language)] if language else self.languages
        want = str(voice).casefold()
        hits = [(lang, v["key"]) for lang in langs for v in self._voices[lang]
                if want in {str(v.get(f, "")).casefold() for f in ("key", "name", "description", "id") if f in v}]
        if not hits:
            raise ValueError(f"no voice {voice!r}" + (f" for {language!r}" if language else "") + "; voices: "
                             + str(self.available_voices if not language else self.available_voices[langs[0]]))
        if len(hits) > 1:
            if language is None and any(lang == self.default_language for lang, _ in hits):
                return next(h for h in hits if h[0] == self.default_language)
            raise ValueError(f"voice {voice!r} exists in {[lang for lang, _ in hits]}: pass language=")
        return hits[0]

    # ---- synthesis ----------------------------------------------------------------------------------------------
    @property
    def frontend(self):
        if self._frontend is None:
            from .frontend import Frontend
            self._frontend = Frontend(self.config.get("phonemizer", "v1"), self.vocab, _LazyDir(self._g2p_dir))
        return self._frontend

    def _front(self, text, lang):
        try:
            return self.frontend(text, lang)[0]
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in FRONT_END_MODULES:
                raise ImportError(front_end_unavailable(exc.name, lang)) from exc
            raise

    def phonemize(self, text, language=None):
        """text -> the phoneme chunks the model will speak (one string per chunk, at most 350 phonemes each)."""
        return self._front(text, self._language(language))

    def _tokens(self, phonemes):
        ids = []
        for ch in phonemes:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            elif ch in "()":
                ids.append(self.vocab[","])
        return [0] + ids + [END_TOKEN, 0]

    def _chunk(self, lang, key, phonemes, speed, keep_start):
        tok = self._tokens(phonemes)
        pack = self._pack(lang, key)
        style = pack[min(len(tok) - 1, pack.shape[0] - 1)][None]
        wav, dur = self._run(lang, tok, style, speed)
        w = np.clip(wav, -1, 1)
        total = int(dur.sum())
        spf = max(1, round(len(w) / max(total, 1)))             # samples per duration frame
        lead = 0 if keep_start else max(0, int(dur[0]) - 1) * spf
        trail = max(0, int(dur[-1]) - 1) * spf
        if lead + trail + spf < len(w):
            w = w[lead: len(w) - trail] if trail else w[lead:]
        w = np.ascontiguousarray(w)
        f = min(120, len(w) // 8)                                # ~5 ms fades
        if f:
            w[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            w[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        return w

    def generate_stream(self, text, voice=None, language=None, speed=1.0, trim_lead=True, chunk_chars=None):
        """Yield the audio of each chunk as it is produced. Play the chunks with 0.15 s of silence between them, as
        generate() does. speed: 0.25 to 4.
        chunk_chars (default None: the model's default, 380, as KittenTTS 2): whole sentences packed into chunks of at
        most that many characters (a longer sentence is cut at spaces). Smaller chunks give an earlier first
        sound, larger ones fewer engine calls. The same text gives the same chunks on every machine (the C library's
        kt08_stream runs the same chunker). Each chunk is phonemised only when it is next; a chunk of more than 350
        phonemes is spoken in parts cut at sentence ends (then spaces), as generate() cuts long text.
        trim_lead (default True): the first chunk starts at its first audible sample (less 20 ms), not after the
        model's ~0.2-0.3 s of silence. Silence (never sound) is removed at chunk edges only: every chunk drops its
        silent start the same way, and a chunk that another follows keeps 30 ms of silence after its last audible
        sample."""
        speed = check_speed(speed)                 # before any work, also for empty text
        from ..tts08 import _family, _stream
        chunk_chars = _family.check_chunk_chars(chunk_chars)
        trim = _stream.check_trim_lead(trim_lead)
        lang, key = self.resolve_voice(voice, language)
        yield from self._stream_chunks(text, lang, key, speed, trim, chunk_chars)

    def _package_chunks(self, text, lang, key, speed):
        """generate()'s chunks: the whole text phonemised, then chunks of up to 350 phonemes."""
        with self._lock:
            chunks = self._front(text, lang)
        for i, ps in enumerate(chunks):
            with self._lock:
                audio = self._chunk(lang, key, ps, float(speed), keep_start=(i == 0))
            yield audio

    def stream_chunks(self, text, language=None, chunk_chars=None):
        """The text chunks stream() speaks for this text (before phonemes), the same on every machine."""
        from ..tts08 import _family
        lang = self.resolve_voice(None, language)[0]
        return self._text_chunks(text, lang, _family.check_chunk_chars(chunk_chars))[0]

    def _text_chunks(self, text, lang, chunk_chars):
        """(chunks, g2p, fold): the text normalised once (English), cut at sentence ends as the front end cuts long
        text (SENT_MULTI), then packed by the native chunker."""
        from .frontend import _LOCK, SENT_MULTI
        fold = None
        try:
            with _LOCK:
                g2p = self.frontend.g2p(lang)
                norm = g2p._normalize(text or "") if hasattr(g2p, "_normalize") else (text or "")
            if lang == "en" and self.frontend.phon == "v2":
                from .frontend import phonemize_v2
                fold = phonemize_v2.fold_extra
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in FRONT_END_MODULES:
                raise ImportError(front_end_unavailable(exc.name, lang)) from exc
            raise
        chunks = self._mi.stream.pack_sentences(SENT_MULTI.split(norm), chunk_chars or STREAM_CHUNK_CHARS)
        return chunks, g2p, fold

    def _stream_chunks(self, text, lang, key, speed, trim, chunk_chars):
        from ..tts08 import _stream
        from .frontend import _LOCK, front_end
        st = self._mi.stream
        chunks, g2p, fold = self._text_chunks(text, lang, chunk_chars)

        def phonemize(chunk):
            # The front end's own cut (front_end): a chunk of up to 350 phonemes is one piece; a longer one is cut at
            # sentence ends, then at spaces, and packed up to 350, as generate() does. The text is already normalised.
            try:
                with self._lock, _LOCK:
                    parts = front_end(g2p, chunk, normalise=False)[0]
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] in FRONT_END_MODULES:
                    raise ImportError(front_end_unavailable(exc.name, lang)) from exc
                raise
            return [fold(p) for p in parts] if fold else parts

        def synth(ph, _text, first, last):
            # Every chunk keeps the model's start (dropping the start pad's frames, as generate()'s later chunks do,
            # cut the first sound of some words at a stream's chunk starts), then drops its silent start to 20 ms before
            # the first sample >= -50 dBFS, except a first chunk kept whole (trim_lead=False). A chunk that another
            # follows keeps 30 ms of silence after its last audible sample. Same as kt08_stream in the C library.
            with self._lock:
                w = self._chunk(lang, key, ph, float(speed), keep_start=True)
            if trim or not first:
                w = w[st.lead_trim_offset(w, self.sample_rate):]
            return w if last else w[:st.tail_keep(w, self.sample_rate)]

        yield from _stream.run(chunks, phonemize, synth)

    def generate(self, text, voice=None, language=None, speed=1.0):
        """Synthesize `text`; float32 audio at `sample_rate` (24 kHz), with 0.15 s between chunks."""
        gap = np.zeros(int(GAP_S * self.sample_rate), dtype=np.float32)
        out = []
        speed = check_speed(speed)
        lang, key = self.resolve_voice(voice, language)
        for w in self._package_chunks(text, lang, key, speed):
            out += [w, gap]
        return np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)

    def generate_to_file(self, text, output_path, voice=None, language=None, speed=1.0):
        """Synthesize `text` into an audio file (WAV, FLAC, ... by the file name) at `sample_rate` (24 kHz)."""
        import soundfile as sf
        audio = self.generate(text, voice=voice, language=language, speed=speed)
        sf.write(str(output_path), audio, self.sample_rate)
        return str(output_path)


class _LazyDir(os.PathLike):
    """A folder path resolved (downloaded) the first time it is used: only English needs the g2p files."""

    def __init__(self, resolve):
        self._resolve, self._path = resolve, None

    def __fspath__(self):
        if self._path is None:
            self._path = self._resolve()
        return self._path

    def __str__(self):
        return self.__fspath__()
