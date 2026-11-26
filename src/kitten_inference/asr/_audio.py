"""Audio input for speech recognition: read a file, mix to mono, resample to 16 kHz, find a quiet place to cut.

Only numpy is used. The model hears what the resampler produces, so its filter is described in full below.
"""
from __future__ import annotations

import functools
import math
import os
from fractions import Fraction

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000

# The resampler's low-pass filter is a sinc with a Kaiser window:
#   _ZEROS    zero crossings of the sinc on each side (more = sharper filter, more work per sample),
#   _ROLLOFF  the -6 dB point, as a fraction of the lower of the two Nyquist frequencies,
#   _BETA     shape of the Kaiser window (larger = less leakage above Nyquist, wider roll-off).
# Going down to 16 kHz, these values keep everything below 7 kHz within 0.2 dB and hold anything above 8 kHz,
# which would otherwise fold back into the speech band, about 70 dB down.
_ZEROS = 32
_ROLLOFF = 0.93
_BETA = 7.0
# A rate pair that needs more filter phases than this is matched to within 0.05 % (500 parts in a million)
# instead, so the filter table stays small; that holds for every rate from 100 Hz up. The usual rates (8, 11.025,
# 22.05, 24, 32, 44.1, 48, 96 kHz) are exact.
_MAX_PHASES = 1024
_BLOCK_ELEMENTS = 1 << 20      # samples x filter taps handled at once (about 4 MB per temporary array)
_READ_FRAMES = 1 << 20         # frames read from a file, or fed to the resampler, at once
_CUT_FRAME = 320               # 20 ms at 16 kHz: the energy frame of quiet_cut
_CUT_SMOOTH = 15               # frames averaged (300 ms), so a pause between phrases wins over a dip inside a word


def check_rate(value) -> int:
    """A sample rate as a whole number of Hz; ValueError unless it is a positive number."""
    try:
        ok = not isinstance(value, bool) and math.isfinite(value) and round(value) >= 1
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"sample_rate must be a positive number of Hz, not {value!r}")
    return int(round(value))


def to_mono_f32(audio) -> np.ndarray:
    """Float or int16 samples as a 1-D float32 array. A 2-D array is [samples, channels] and is averaged."""
    x = np.asarray(audio)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype.kind == "f":
        x = x.astype(np.float32, copy=False)
    else:
        raise ValueError(f"audio samples must be float or int16, not {x.dtype}")
    if x.ndim == 2:
        if x.shape[1] > 8 and x.shape[1] > x.shape[0]:
            # [channels, samples] would otherwise be averaged down to one or two samples without a word
            raise ValueError(f"a 2-D audio array must be [samples, channels]; got shape {x.shape}, transpose it")
        x = x[:, 0] if x.shape[1] == 1 else x.mean(axis=1, dtype=np.float32)
    elif x.ndim != 1:
        raise ValueError(f"audio must be a 1-D array of samples or a 2-D array [samples, channels], "
                         f"not {x.ndim}-D")
    if x.size and not np.isfinite(x).all():
        raise ValueError("audio contains NaN or infinite samples")
    return np.ascontiguousarray(x)


@functools.lru_cache(maxsize=8)
def _filter_bank(up: int, down: int):
    """Filter taps for every output phase -> (float32 [up, taps], half), with taps = 2 * half + 1.

    Output sample n sits at input time t = n * down / up = idx + phase / up. Its value is the sum over the input
    samples k = idx - half .. idx + half of x[k] * g(t - k), where g is the windowed sinc. Row ``phase`` holds
    g(phase / up - j) for j = -half .. half.
    """
    cutoff = _ROLLOFF * min(1.0, up / down)            # as a fraction of the input Nyquist frequency
    width = _ZEROS / cutoff                            # filter half-width in input samples
    half = int(math.ceil(width))
    j = np.arange(-half, half + 1, dtype=np.float64)
    tau = np.arange(up, dtype=np.float64)[:, None] / up - j[None, :]
    inside = np.clip(1.0 - (tau / width) ** 2, 0.0, None)
    window = np.where(np.abs(tau) < width, np.i0(_BETA * np.sqrt(inside)) / np.i0(_BETA), 0.0)
    bank = cutoff * np.sinc(cutoff * tau) * window
    bank /= bank.sum(axis=1, keepdims=True)            # every phase passes a constant level unchanged
    return bank.astype(np.float32), half


class Resampler:
    """Band-limited sample rate conversion, a block at a time.

    ``feed(block)`` returns the output samples that are final so far and ``flush()`` the rest; together they are
    exactly what one pass over the whole signal gives, so a long recording never has to sit in memory at its
    original rate. Equal rates pass the samples through untouched.
    """

    def __init__(self, sr_in, sr_out=SAMPLE_RATE):
        sr_in, sr_out = check_rate(sr_in), check_rate(sr_out)
        self.same = sr_in == sr_out
        ratio = Fraction(sr_in, sr_out).limit_denominator(_MAX_PHASES)
        if ratio.numerator == 0:
            raise ValueError(f"cannot resample from {sr_in} Hz to {sr_out} Hz")
        self.down, self.up = ratio.numerator, ratio.denominator      # output n sits at input time n * down / up
        if self.same:
            return
        self._bank, self._half = _filter_bank(self.up, self.down)
        self._taps = 2 * self._half + 1
        self._block = max(256, _BLOCK_ELEMENTS // self._taps)
        # Input that later output still needs. It starts with the silence before the first sample.
        self._buf = np.zeros(self._half, dtype=np.float32)
        self._buf_start = -self._half      # position of _buf[0] in the input signal
        self._seen = 0                     # input samples fed so far
        self._n = 0                        # next output sample

    def output_length(self, n_in: int) -> int:
        return n_in if self.same else -(-n_in * self.up // self.down)

    def feed(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.same:
            return x
        self._seen += len(x)
        self._buf = np.concatenate([self._buf, x])
        # Output n reads the input up to n * down // up + half: stop where that runs past what has arrived.
        return self._run(self.output_length(self._seen - self._half))

    def flush(self) -> np.ndarray:
        """The last samples, computed with silence after the end. Do not feed again afterwards."""
        if self.same:
            return np.zeros(0, dtype=np.float32)
        self._buf = np.concatenate([self._buf, np.zeros(self._half, dtype=np.float32)])
        return self._run(self.output_length(self._seen))

    def _run(self, n_stop: int) -> np.ndarray:
        out = []
        if self._n < n_stop:
            # Row i of `windows` is the input from _buf_start + i on, one filter length long (a view, no copy).
            windows = sliding_window_view(self._buf, self._taps)
            while self._n < n_stop:
                at = np.arange(self._n, min(n_stop, self._n + self._block), dtype=np.int64) * self.down
                rows = at // self.up - self._half - self._buf_start
                out.append(np.einsum("ij,ij->i", windows[rows], self._bank[at % self.up]))
                self._n += len(at)
            # Drop the input no later output reads; the copy lets the fed block be freed.
            keep = max(0, self._n * self.down // self.up - self._half - self._buf_start)
            self._buf = self._buf[keep:].copy()
            self._buf_start += keep
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def resample(x, sr_in, sr_out=SAMPLE_RATE) -> np.ndarray:
    """1-D float32 samples at ``sr_in`` -> the same signal at ``sr_out`` (unchanged when the rates are equal)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("resample takes a 1-D array of samples")
    resampler = Resampler(sr_in, sr_out)
    if resampler.same:
        return x
    out = np.empty(resampler.output_length(len(x)), dtype=np.float32)
    at = 0
    for start in range(0, len(x), _READ_FRAMES):       # in blocks: the temporaries stay a few MB
        y = resampler.feed(x[start:start + _READ_FRAMES])
        out[at:at + len(y)] = y
        at += len(y)
    y = resampler.flush()
    out[at:at + len(y)] = y
    return out


def load(path, sample_rate=None):
    """Read an audio file -> (float32 mono samples, sample rate).

    The file's own rate is kept unless ``sample_rate`` is given; then the audio is resampled while it is read,
    block by block, and that rate is returned. Reads what the soundfile package reads (WAV, FLAC, OGG, MP3).
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio file not found: {path}")
    import soundfile as sf            # on first use: importing this package needs no third-party module
    parts = []
    with sf.SoundFile(path) as f:
        rate = int(f.samplerate)
        resampler = None if sample_rate is None else Resampler(rate, sample_rate)
        while True:
            block = f.read(_READ_FRAMES, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1, dtype=np.float32)
            parts.append(mono if resampler is None else resampler.feed(mono))
    if resampler is not None:
        parts.append(resampler.flush())
        rate = check_rate(sample_rate)
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), rate


def quiet_cut(x, lo: int, hi: int) -> int:
    """The sample in (lo, hi] where cutting ``x`` in two does the least harm.

    Energy is summed over 20 ms frames from ``lo`` on; the cut goes to the middle of the quietest 300 ms (fifteen
    frames in a row, the first such place when several are equally quiet), so a real pause between phrases wins
    over a short dip inside a word. A range shorter than 300 ms has no such place: the answer is then ``hi``,
    the hard cut.
    """
    lo, hi = max(0, int(lo)), min(len(x), int(hi))
    frames = (hi - lo) // _CUT_FRAME
    if frames < _CUT_SMOOTH:
        return hi
    part = np.asarray(x[lo:lo + frames * _CUT_FRAME], dtype=np.float64)
    energy = np.square(part).reshape(frames, _CUT_FRAME).sum(axis=1)
    energy = np.convolve(energy, np.ones(_CUT_SMOOTH), mode="valid")
    return lo + int(np.argmin(energy)) * _CUT_FRAME + _CUT_SMOOTH * _CUT_FRAME // 2
