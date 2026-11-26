"""Speech recognition: ``KittenASR(model)`` turns audio files, arrays and live audio into text."""
from __future__ import annotations

import math
import os
import platform
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import _audio

SAMPLE_RATE = _audio.SAMPLE_RATE
_MAX_WINDOW_S = 30.0       # the longest audio the model is given in one piece
_CUT_SEARCH_S = 10.0       # a cut goes to the quietest place in the last 10 s of a window
_MIN_AUDIO_S = 0.1         # shorter audio holds no word
_SILENCE_PEAK = 1e-4       # audio that never rises above this (-80 dB of full scale) is not sent to the model
_BACKGROUND = 10           # engine priority of file work: live streams are served first
# Keywords only one engine knows: with the backend picked automatically, those of the other engine are left out.
_CPU_ONLY_OPTIONS = frozenset(("threads", "decode_threads", "audio_window", "flash_attn", "gpu", "kv_type", "n_ctx",
                               "n_batch", "repack", "files", "batch", "encoder_precision"))
_GPU_ONLY_OPTIONS = frozenset(("device", "dtype", "kv_tokens", "max_batch", "max_seq", "prefill_cap", "reserve_rows",
                               "max_background_rows", "page_ahead", "stream_ttl_s", "progress_every", "profile",
                               "resync", "resync_min", "resync_window", "resync_max", "pass_gap_ms", "fuse", "compile",
                               "cuda_graph", "batch_buckets", "conv_batch", "head_rows", "prefill_buckets",
                               "enc_buckets", "prefill_graph", "enc_graph", "pad_short", "attention", "kv_buckets",
                               "page", "encoder_window", "start_timeout_s"))
_FILE_TIMEOUT_S = 600.0    # a window of a file may wait behind live streams and behind other windows
_TEXT_TAG = "<asr_text>"
# A window that ran away: greedy decoding repeats one piece until the window's token budget is used up. Only a window
# that stopped at its budget (finish_reason "length") is trimmed, and only when its output ends in one piece repeated
# at least 4 times (40 characters or more, pieces of up to 80 characters): that tail is cut to one copy. A window
# that ended by itself keeps every word, so real repeated speech ("Thank you. Thank you.") is never cut.
_LOOP_MIN_REPEATS, _LOOP_MIN_CHARS, _LOOP_MAX_PERIOD = 4, 40, 80
# Live audio: on the first second or two of speech the model sometimes writes a stock phrase ("Thank you.") that the
# next update replaces. Such a phrase is not shown as the partial text while nothing is committed and the open window
# is shorter than _FILLER_WINDOW_S; the final text is never changed by this.
_FILLER_PARTIALS = frozenset(("thank you", "thank you very much", "thanks for watching", "thank you for watching",
                              "you", "bye"))
_FILLER_WINDOW_S = 2.0
# Live audio, speech gate: no update is decoded until the session has heard something louder than its background.
# Energy of 20 ms frames in dB; the background is the 10th percentile of the frames of the last 60 s. A frame 15 dB
# above it, or louder than -30 dB of full scale, opens the gate for the rest of the session. Steady noise (a fan,
# hiss, a quiet room) does not open it; speech does within its first syllable.
_GATE_FRAME = 320
_GATE_RISE_DB = 15.0
_GATE_LOUD_DB = -30.0
_GATE_HISTORY = 3000       # frames (60 s) the background is estimated from: a long quiet start costs no more
_LANGUAGE_PREFIX = "language "


@dataclass
class Segment:
    """The text of one window of the audio; ``start`` and ``end`` are seconds from the start of the audio."""
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """``segments`` and ``raw`` (the model's output, unparsed) have one entry per window, in order."""
    text: str
    language: Optional[str]
    segments: List[Segment] = field(default_factory=list)
    raw: List[str] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """``text`` is everything so far: ``committed`` (final) followed by ``partial`` (may still change)."""
    text: str
    committed: str
    partial: str


def parse_output(raw, language: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Split the model's output into (language, text).

    The model writes ``language <Name><asr_text><transcript>``, or only the transcript when the prompt already
    names the language (pass that name as ``language``). Output without the prefix is taken as the transcript.
    Only leading and trailing whitespace is removed: tags inside the transcript stay.
    """
    text = str(raw).strip()
    head, tag, body = text.partition(_TEXT_TAG)
    if not tag:
        return language, text
    head = head.strip()
    name = head[len(_LANGUAGE_PREFIX):].strip() if head.lower().startswith(_LANGUAGE_PREFIX) else ""
    if not name or name.lower() == "none":       # "None" is the model's word for "no speech heard"
        name = language
    return name, body.strip()


def _language_name(language) -> Optional[str]:
    if language is None:
        return None
    name = language.strip() if isinstance(language, str) else ""
    # The name becomes part of the prompt, so it holds only words of letters with single spaces or hyphens
    # between them. A first word of one or two letters is a code ("en", "en-US"), which the model does not know.
    words = name.replace("-", " ").split(" ")
    if len(words[0]) < 3 or not all(word.isalpha() for word in words):
        raise ValueError("language must be a language name such as 'English' (not a code such as 'en'), "
                         "or None to let the model detect it")
    return name[:1].upper() + name[1:].lower()   # the model knows languages by capitalised name


def _seconds(value, name: str, low: float, high: float) -> float:
    try:
        ok = not isinstance(value, bool) and low <= value <= high
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{name} must be between {low:g} and {high:g} seconds, not {value!r}")
    return float(value)


def _worth_decoding(x: np.ndarray) -> bool:
    """False for audio that is too short or too quiet to hold a word: its text is "" without asking the model."""
    return len(x) >= int(_MIN_AUDIO_S * SAMPLE_RATE) and float(np.abs(x).max()) >= _SILENCE_PEAK


def _token_budget(n_samples: int) -> int:
    """The most tokens one window may produce: room for fast speech, an end for a runaway repetition."""
    return 64 + 16 * int(math.ceil(n_samples / SAMPLE_RATE))


def _trim_loop(text: str) -> str:
    """``text`` without a run-away tail: when it ends in the same piece repeated at least 4 times (40 characters or
    more in all), one copy of the piece is kept. Any other text is returned unchanged."""
    n, best = len(text), None
    for p in range(1, min(_LOOP_MAX_PERIOD, n // _LOOP_MIN_REPEATS) + 1):
        k = 1
        while n - (k + 1) * p >= 0 and text[n - (k + 1) * p:n - k * p] == text[n - p:]:
            k += 1
        if k >= _LOOP_MIN_REPEATS and k * p >= _LOOP_MIN_CHARS and (best is None or k * p > best[0] * best[1]):
            best = (k, p)
    if best is None:
        return text
    k, p = best
    return text[:n - (k - 1) * p].rstrip()


def _ran_out(result) -> bool:
    """True when a decode stopped because it used up its token budget."""
    return (getattr(result, "info", None) or {}).get("finish_reason") == "length"


def _window_text(result) -> str:
    """The raw text of one decode, with a run-away tail cut to one copy when the decode used up its budget."""
    return _trim_loop(result.text) if _ran_out(result) else result.text


def _windows(x: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end) sample ranges of at most 30 s that cover ``x``. Longer audio is cut at the quietest place in
    the last 10 s of each window, so every window but the last is 20 to 30 s long."""
    limit, search = int(_MAX_WINDOW_S * SAMPLE_RATE), int(_CUT_SEARCH_S * SAMPLE_RATE)
    shortest = int(_MIN_AUDIO_S * SAMPLE_RATE)
    spans, start = [], 0
    while len(x) - start > limit:
        # A cut leaves at least 0.1 s behind it: a shorter rest would be a window too short to decode.
        cut = _audio.quiet_cut(x, start + limit - search, min(start + limit, len(x) - shortest))
        spans.append((start, cut))
        start = cut
    if len(x) - start >= shortest:        # false only for audio shorter than 0.1 s in all
        spans.append((start, len(x)))
    return spans


def _to_16k(item, sample_rate: int) -> np.ndarray:
    if isinstance(item, (str, os.PathLike)):
        path = os.fspath(item)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("audio path is empty")
        return _audio.load(path, SAMPLE_RATE)[0]
    if isinstance(item, np.ndarray):
        return _audio.resample(_audio.to_mono_f32(item), sample_rate)
    raise ValueError(f"audio must be a file path, a numpy array of samples, or a list of these, "
                     f"not {type(item).__name__}")


class KittenASR:
    """Load a Kitten speech recognition model by Hugging Face name or local folder.

        asr = KittenASR("KittenML/kitten-asr-tiny")
        text = asr.transcribe("meeting.wav")

    ``backend``: leave it out to pick automatically: an NVIDIA GPU when PyTorch is installed and finds one that can
    run the model, otherwise the CPU (on a Mac with Apple silicon, the Mac GPU). Or name it: "cuda" (NVIDIA GPU,
    needs PyTorch), "cpu" (no PyTorch needed) or "metal" (the Mac GPU).

    ``precision`` is the number format of the decoder's weights; leave it out for the backend's default. On an
    NVIDIA GPU: "bf16" (the published weights, default), "fp16" or "int8" (8-bit weights). On the CPU and the Mac
    GPU: "q8_0" (8-bit weights, default), "f16" or "bf16". Small-enhanced also accepts its published "int4" and
    "int8" weights. Linux and Windows CPU and CUDA with Triton use packed weight matrices. See the engine README.

    ``system_prompt``: "auto" (default) uses the system message the model folder sets in
    ``asr_config.json``, if it has one (otherwise none); "none" never sends one. ``capabilities["system_prompt"]`` says which one is in use.

    ``process=True`` runs the engine in a process of its own: use it in a server whose own threads keep Python
    busy. ``workers`` is the number of windows of file audio waiting on the engine at once. Other keywords go to
    the engine (:class:`kitten_inference.asr.Engine`). With ``backend`` left out, keywords that only the other
    engine knows (for example ``threads`` and ``batch`` of the CPU engine on a computer with an NVIDIA GPU) are
    left out, so the same code runs on both kinds of computer.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, model, *, backend=None, precision=None, cache_dir=None, process=False, workers=16,
                 system_prompt="auto", **engine_options):
        self._closed = True               # until the engine is up, so close() is safe after a failed start
        self._engine = None
        self._pool = None
        self._lock = threading.Lock()
        self._prompts = {}
        self._attention = None
        self._precision = None
        self._system_prompt = None
        name = os.fspath(model) if isinstance(model, (str, os.PathLike)) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model must be a Hugging Face model name or a local model folder")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"workers must be a whole number of 1 or more, not {workers!r}")
        self._workers = workers
        self._process = bool(process)
        from . import _deps, _precision, _system
        # Checked before anything is imported or downloaded, in the same words on every computer.
        backend = _precision.check_backend(backend)
        if _system.check_mode(system_prompt) != "auto":   # "auto" is both engines' own default
            engine_options["system_prompt"] = _system.check_mode(system_prompt)
        if self._process and backend in ("cpu", "metal"):
            raise ValueError("process=True is for the NVIDIA GPU engine; the CPU engine runs in this process")
        if backend is not None:
            precision = _precision.check(backend, precision)
        checked = False                   # the GPU checks already ran (and passed) while picking the backend
        gpu_problem = None
        if backend is None:
            if self._process or "device" in engine_options:
                backend = "cuda"
            else:
                try:
                    _deps.require("asr")
                    backend, checked = "cuda", True
                except (ImportError, RuntimeError) as exc:
                    backend, gpu_problem = "cpu", exc   # no PyTorch or no usable NVIDIA GPU: the CPU engine needs neither
            precision = _precision.check(backend, precision)
            other = _CPU_ONLY_OPTIONS if backend == "cuda" else _GPU_ONLY_OPTIONS
            engine_options = {k: v for k, v in engine_options.items() if k not in other}
        elif backend == "metal":
            if not (sys.platform == "darwin" and platform.machine() == "arm64"):
                raise RuntimeError("backend='metal' is the GPU of a Mac with Apple silicon, which this computer is "
                                   "not. Leave out backend (or pass backend='cpu').")
            if engine_options.get("gpu", True) is not True:
                raise ValueError("backend='metal' runs on the Mac GPU: leave out gpu, or pass backend='cpu'")
            engine_options["gpu"] = True
        elif backend == "cpu":
            engine_options.setdefault("gpu", False)
        self._backend = backend
        cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        if backend != "cuda":
            try:
                _deps.require("asr-cpu")
            except ImportError as exc:
                if gpu_problem is None:
                    raise
                raise ImportError(f"{exc} An NVIDIA GPU cannot be used here either: {gpu_problem}") from exc
            from ._cpu import CpuEngine
            self._device = "cpu"
            self._engine = CpuEngine(name.strip(), cache_dir=cache_dir, precision=precision, **engine_options)
            if getattr(self._engine, "params", {}).get("use_gpu"):
                self._backend = self._device = "metal"   # the automatic choice on a Mac with Apple silicon
            elif backend == "metal" or engine_options.get("gpu") is True:
                self._engine.close()                     # the Mac GPU was asked for, and the engine found none
                raise RuntimeError("backend='metal' (or gpu=True) was asked for, but no Mac GPU could be used here. "
                                   "Pass backend='cpu' to run on the CPU.")
            self._closed = False
            return
        self._device = str(engine_options.get("device", "cuda"))
        # Checks Python, the platform, PyTorch and the NVIDIA GPU. With process=True, PyTorch and the GPU are
        # only looked at in the engine's own process, so this one stays light.
        if self._process:
            _deps.require("asr-process")
            from ._proc import EngineProcess as engine_type
        else:
            if not checked:
                _deps.require("asr", device=engine_options.get("device"))
            from ._engine import Engine as engine_type
        if "dtype" not in engine_options:
            engine_options["precision"] = precision
        self._engine = engine_type(name.strip(), cache_dir=cache_dir, **engine_options)
        self._closed = False

    def _open(self):
        if self._closed:
            raise RuntimeError("this KittenASR is closed")
        return self._engine

    @property
    def capabilities(self) -> dict:
        engine = self._open()
        if self._attention is None:
            snap = engine.snapshot()
            self._attention = snap.get("attention")
            self._precision = snap.get("precision")
            self._system_prompt = snap.get("system_prompt")
        batching = self._backend == "cuda" or getattr(engine, "batch", 1) > 1
        return {"streaming": True, "batching": batching, "backend": self._backend, "device": self._device,
                "attention": self._attention, "precision": self._precision,
                "process": self._process, "sample_rate": SAMPLE_RATE, "max_window_seconds": _MAX_WINDOW_S,
                "system_prompt": self._system_prompt}

    def _prompt(self, language) -> Tuple[str, Optional[str]]:
        name = _language_name(language)
        prompt = self._prompts.get(name)
        if prompt is None:
            prompt = self._prompts[name] = self._open().default_prompt(name)
        return prompt, name

    # ------------------------------------------------------------ files and arrays
    def transcribe(self, audio, *, sample_rate=None, language=None, details=False):
        """Audio -> text.

        ``audio`` is a file path, or a numpy array of float or int16 samples (1-D, or 2-D [samples, channels])
        at ``sample_rate`` (default 16000), or a list of these; a list returns a list, and all of it is on the
        engine at the same time. ``language`` is a name such as "English"; None lets the model detect it.
        ``details=True`` returns a :class:`Transcript` instead of the text.

        Audio longer than 30 s is cut into windows at quiet points and the texts are joined with a space. Audio
        shorter than 0.1 s, and silence, give "".
        """
        self._open()
        rate = SAMPLE_RATE if sample_rate is None else _audio.check_rate(sample_rate)
        prompt, name = self._prompt(language)
        many = isinstance(audio, (list, tuple))
        if getattr(self._engine, "batch", 1) > 1:     # the CPU engine: windows decoded together
            results = self._transcribe_together(audio if many else [audio], rate, prompt, name, details)
            return results if many else results[0]
        jobs = []
        try:
            for item in (audio if many else [audio]):
                wave = _to_16k(item, rate)
                job = []
                jobs.append(job)
                for start, end in _windows(wave):
                    job.append((start, end, self._submit(wave[start:end], prompt)))
            results = [self._collect(job, name, details) for job in jobs]
        except BaseException:
            for job in jobs:              # one window failed, or the caller gave up: drop what has not started
                for _, _, future in job:
                    if future is not None:
                        future.cancel()
            raise
        return results if many else results[0]

    def _transcribe_together(self, items, rate: int, prompt: str, name: Optional[str], details: bool):
        """Every window of every item in one engine call, which decodes several windows in the same steps (the CPU
        engine's decode_batch; a live session on the same model still gets its turn within seconds, see
        CpuEngine._turn). Same windows, prompt and token budgets as one window at a time."""
        waves = [_to_16k(item, rate) for item in items]
        jobs, todo = [], []
        for wave in waves:
            job = []
            for start, end in _windows(wave):
                piece = wave[start:end]
                job.append((start, end, len(todo) if _worth_decoding(piece) else None))
                if job[-1][2] is not None:
                    todo.append(piece)
            jobs.append(job)
        texts = []
        if todo:
            with self._lock:
                engine = self._open()
            # Longest windows first: they produce the most tokens, so fewer steps run with empty slots at the end.
            # (KITTEN_ASR_BATCH_ORDER=input keeps the given order; for measurements.)
            order = list(range(len(todo)))
            if os.environ.get("KITTEN_ASR_BATCH_ORDER") != "input":
                order.sort(key=lambda k: -len(todo[k]))
            res = engine.decode_batch([todo[k] for k in order], prompt, [_token_budget(len(todo[k])) for k in order])
            texts = [None] * len(todo)
            for k, r in zip(order, res):
                texts[k] = r

        def done(k):
            if k is None:
                return None
            f = Future()
            f.set_result(texts[k])
            return f
        return [self._collect([(s, e, done(k)) for s, e, k in job], name, details) for job in jobs]

    def _submit(self, window: np.ndarray, prompt: str):
        if not _worth_decoding(window):
            return None
        with self._lock:
            self._open()
            if self._pool is None:
                # The pool only waits on the engine; the engine batches whatever is waiting on the GPU.
                self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="kitten-asr")
            return self._pool.submit(self._decode_window, window, prompt)

    def _decode_window(self, window: np.ndarray, prompt: str):
        return self._open().decode(window, prompt, max_new_tokens=_token_budget(len(window)),
                                   priority=_BACKGROUND, timeout_s=_FILE_TIMEOUT_S)

    @staticmethod
    def _collect(job, name: Optional[str], details: bool):
        texts, segments, raws, language = [], [], [], name
        for start, end, future in job:
            result = None if future is None else future.result()      # a DecodeResult, or None for silence
            raw = "" if result is None else result.text
            found, text = parse_output("" if result is None else _window_text(result), name)
            if language is None:
                language = found
            raws.append(raw)
            segments.append(Segment(start / SAMPLE_RATE, end / SAMPLE_RATE, text))
            if text:
                texts.append(text)
        text = " ".join(texts)
        return Transcript(text, language, segments, raws) if details else text

    # ------------------------------------------------------------ live audio
    def stream(self, *, language=None, step=0.5, max_window=20.0, speech_gate=True) -> "StreamSession":
        """Start a session for one live speaker: ``push(samples)`` as audio arrives, ``finish()`` at the end.

        Every ``step`` seconds of new audio the text is brought up to date; ``max_window`` seconds is the most
        audio that stays open to change before its text is committed. Sessions on several threads share the engine.
        ``speech_gate`` (default True): no update is decoded before the first sound clearly louder than the
        background, so quiet noise before speech costs nothing and produces no words; ``finish()`` and full windows
        are always decoded.
        """
        self._open()
        return StreamSession(self, language, step, max_window, speech_gate)

    # ------------------------------------------------------------ lifetime
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True           # windows still queued now fail instead of starting
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        if self._engine is not None:
            self._engine.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()


class StreamSession:
    """Live transcription of one speaker. Made by :meth:`KittenASR.stream`.

    Audio is 16 kHz mono. The session keeps one open window of audio. Each update decodes that window from its
    start, but the engine keeps what it already worked out for it, so only the new audio costs work. A window
    that reaches ``max_window`` seconds is cut at a quiet point: the text before the cut is committed and the
    rest starts the next window.
    """

    def __init__(self, asr: KittenASR, language, step, max_window, speech_gate=True):
        max_window = _seconds(max_window, "max_window", 1.0, _MAX_WINDOW_S)
        step = _seconds(step, "step", _MIN_AUDIO_S, max_window)
        self._asr = asr
        self._prompt, self._forced = asr._prompt(language)
        self._step = int(round(step * SAMPLE_RATE))
        self._max = int(round(max_window * SAMPLE_RATE))
        # The cut of a full window is looked for in its last 10 s, or in its second half when that is shorter.
        self._search = min(int(_CUT_SEARCH_S * SAMPLE_RATE), self._max // 2)
        self._lock = threading.Lock()
        self._id = uuid.uuid4().hex[:12]
        self._serial = 0                              # number of the open window: part of its engine key
        self._audio = np.zeros(0, dtype=np.float32)   # the open window
        self._fresh = 0                               # samples of it that no decode has seen yet
        self._ids = None                              # token ids of its last decode: the next decode's draft
        self._partial = ""
        self._committed = []
        self._language = self._forced
        self._done = False
        self._gate = _SpeechGate() if speech_gate else None

    @property
    def text(self) -> str:
        """Everything recognised so far."""
        partial = self._shown_partial()
        return " ".join(self._committed + ([partial] if partial else []))

    def _shown_partial(self) -> str:
        p = self._partial
        if p and not self._committed and len(self._audio) < _FILLER_WINDOW_S * SAMPLE_RATE:
            words = "".join(ch for ch in p.lower() if ch.isalnum() or ch.isspace()).split()
            if " ".join(words) in _FILLER_PARTIALS:
                return ""
        return p

    @property
    def language(self) -> Optional[str]:
        """The language asked for, or the one the model detected (None until it has)."""
        return self._language

    def push(self, samples) -> Optional[StreamUpdate]:
        """Add 16 kHz mono samples (float or int16). Returns a :class:`StreamUpdate` when the text was brought
        up to date, or None when this audio only went into the buffer."""
        with self._lock:
            if self._done:
                raise RuntimeError("this stream is finished")
            self._asr._open()
            new = _audio.to_mono_f32(samples)
            if not len(new):
                return None
            self._audio = np.concatenate([self._audio, new])
            self._fresh += len(new)
            heard = self._gate is None or self._gate.feed(new)
            decoded = False
            while len(self._audio) >= self._max:
                self._commit(_audio.quiet_cut(self._audio, self._max - self._search, self._max))
                decoded = True
            # After a cut the rest is decoded at once, however short: its words were in the last partial text.
            if decoded or (self._fresh >= self._step and heard):
                self._partial = self._decode(self._audio)
                self._fresh = 0
                decoded = True
            if not decoded:
                return None
            committed = " ".join(self._committed)
            return StreamUpdate(self.text, committed, self._shown_partial())

    def finish(self) -> str:
        """Decode what is left, commit it and return the full text. The session takes no audio afterwards."""
        with self._lock:
            if not self._done:
                self._asr._open()
                if len(self._audio):
                    self._commit(len(self._audio))
                self._done = True
            return " ".join(self._committed)

    def close(self) -> None:
        """Stop without a last decode and free what the engine kept for this stream."""
        with self._lock:
            if self._done:
                return
            self._done = True
            self._audio = np.zeros(0, dtype=np.float32)
            self._partial = ""
            try:
                self._asr._open().release(self._key())
            except RuntimeError:          # the model was closed first: nothing is left to free
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _key(self) -> str:
        return f"{self._id}:{self._serial}"

    def _decode(self, audio: np.ndarray) -> str:
        """Text of the open window's first ``len(audio)`` samples. The previous result goes along as a draft:
        the part of it the model still agrees with costs no decode steps."""
        engine = self._asr._open()
        if not _worth_decoding(audio):
            self._ids = None
            return ""
        if self._serial > 0 and self._gate is not None and self._gate.quiet(audio):
            # a later window (after a cut) that holds nothing louder than the session's background: the model
            # would only invent words on the noise, which a file transcription of the same audio does not have
            self._ids = None
            return ""
        result = engine.decode(audio, self._prompt, draft_ids=self._ids, stream_key=self._key(),
                               max_new_tokens=_token_budget(len(audio)))
        self._ids = list(result.token_ids)           # the model's own tokens stay the next draft, loop and all
        found, text = parse_output(_window_text(result), self._forced)
        if self._language is None:
            self._language = found
        return text

    def _commit(self, cut: int) -> None:
        """Close the open window at sample ``cut``: its text becomes final, the rest starts a new window."""
        if cut == len(self._audio) and self._fresh == 0 and self._ids is not None:
            text = self._partial          # the last update already decoded exactly this audio
        else:
            text = self._decode(self._audio[:cut])
        self._asr._open().release(self._key())
        if text:
            self._committed.append(text)
        self._audio = self._audio[cut:].copy()
        self._serial += 1
        self._fresh = len(self._audio)
        self._ids = None
        self._partial = ""


class _SpeechGate:
    """Opens at the first 20 ms frame 15 dB above the background heard so far, or louder than -30 dBFS."""

    def __init__(self):
        self.open = False
        self.floor = None
        self._rest = np.zeros(0, dtype=np.float32)
        self._db = []

    def feed(self, x: np.ndarray) -> bool:
        if self.open:
            return True
        x = np.concatenate([self._rest, x])
        n = len(x) // _GATE_FRAME
        self._rest = x[n * _GATE_FRAME:].copy()
        if n == 0:
            return False
        power = np.square(x[:n * _GATE_FRAME].astype(np.float64)).reshape(n, _GATE_FRAME).mean(axis=1)
        db = 10.0 * np.log10(power + 1e-12)
        self._db.append(db)
        while len(self._db) > 1 and sum(len(d) for d in self._db) - len(self._db[0]) >= _GATE_HISTORY:
            self._db.pop(0)
        floor = float(np.percentile(np.concatenate(self._db), 10))
        top = float(db.max())
        self.open = top >= floor + _GATE_RISE_DB or top >= _GATE_LOUD_DB
        if self.open:
            self._db = []
            self.floor = floor
        return self.open

    def quiet(self, x: np.ndarray) -> bool:
        """True when the gate is open and no 20 ms frame of ``x`` rises above what opened it."""
        n = len(x) // _GATE_FRAME
        if not self.open or n == 0:
            return False
        power = np.square(x[:n * _GATE_FRAME].astype(np.float64)).reshape(n, _GATE_FRAME).mean(axis=1)
        top = float(10.0 * np.log10(power.max() + 1e-12))
        return top < self.floor + _GATE_RISE_DB and top < _GATE_LOUD_DB
