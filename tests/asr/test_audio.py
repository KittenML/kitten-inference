"""Audio input: mixing to mono, int16 scaling, the resampler (a tone keeps its frequency, level and place in
time; blocks give what one pass gives), reading files, and the quiet point where long audio is cut."""
import math

import numpy as np
import pytest

from kitten_inference.asr import _audio

SR = 16000


def sine(hz, rate, seconds=1.0, level=0.5):
    n = int(round(rate * seconds))
    return (level * np.sin(2 * np.pi * hz * np.arange(n) / rate)).astype(np.float32)


def noise(n, seed=0, level=0.5):
    return np.random.RandomState(seed).uniform(-level, level, n).astype(np.float32)


def peak_hz(y, rate=SR):
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spectrum)) * rate / len(y)


def level_db(y, amplitude):
    """Level of `y` against a sine of that amplitude, in dB."""
    rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
    return 20 * math.log10(max(rms, 1e-12) * math.sqrt(2) / amplitude)


# ---------------------------------------------------------------- sample rates
def test_sample_rate_is_16_khz():
    assert _audio.SAMPLE_RATE == SR


@pytest.mark.parametrize("value, rate", [(16000, 16000), (44100.0, 44100), (8000.4, 8000), (np.int64(48000), 48000)])
def test_check_rate(value, rate):
    out = _audio.check_rate(value)
    assert out == rate and type(out) is int


@pytest.mark.parametrize("value", [0, -16000, 0.4, None, "16000", True, float("nan"), float("inf"), [16000]])
def test_check_rate_refuses(value):
    with pytest.raises(ValueError, match="sample_rate must be a positive number of Hz"):
        _audio.check_rate(value)


# ---------------------------------------------------------------- samples
def test_int16_is_scaled_to_full_scale_one():
    out = _audio.to_mono_f32(np.array([-32768, 0, 16384, 32767], np.int16))
    assert out.dtype == np.float32 and out.tolist() == [-1.0, 0.0, 0.5, 32767 / 32768]


def test_float_samples_keep_their_values():
    x = sine(440.0, SR, 0.1)
    for given in (x, x.astype(np.float64), x.astype(np.float16).astype(np.float64), x.tolist()):
        out = _audio.to_mono_f32(given)
        assert out.dtype == np.float32 and out.ndim == 1 and out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out, np.asarray(given, dtype=np.float32))
    every_other = _audio.to_mono_f32(x[::2])
    assert every_other.flags["C_CONTIGUOUS"] and np.array_equal(every_other, x[::2])


def test_stereo_is_averaged():
    left, right = sine(300.0, SR, 0.1), sine(500.0, SR, 0.1, level=0.25)
    out = _audio.to_mono_f32(np.stack([left, right], axis=1))
    assert out.dtype == np.float32 and out.shape == left.shape
    np.testing.assert_allclose(out, (left.astype(np.float64) + right) / 2, rtol=0, atol=1e-6)
    assert np.array_equal(_audio.to_mono_f32(left[:, None]), left)                  # one channel, as a column
    pcm = np.array([[32767, -32768], [16384, 16384]], np.int16)
    np.testing.assert_allclose(_audio.to_mono_f32(pcm), [-0.5 / 32768, 0.5], rtol=0, atol=1e-9)


def test_channels_first_is_refused():
    with pytest.raises(ValueError, match=r"must be \[samples, channels\]; got shape \(2, 1000\), transpose it"):
        _audio.to_mono_f32(np.zeros((2, 1000), np.float32))
    assert _audio.to_mono_f32(np.zeros((1000, 2), np.float32)).shape == (1000,)


@pytest.mark.parametrize("bad, message", [
    (np.zeros(4, np.int32), "audio samples must be float or int16, not int32"),
    (np.zeros(4, np.uint8), "audio samples must be float or int16, not uint8"),
    (np.zeros(4, bool), "audio samples must be float or int16"),
    (np.zeros((2, 2, 2), np.float32), "not 3-D"),
    (np.float32(0.5), "not 0-D"),
    (np.array([0.0, np.nan], np.float32), "NaN or infinite"),
    (np.array([0.0, np.inf]), "NaN or infinite"),
])
def test_samples_that_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        _audio.to_mono_f32(bad)


def test_empty_audio_is_allowed():
    assert _audio.to_mono_f32(np.zeros(0, np.float32)).shape == (0,)
    assert _audio.resample(np.zeros(0, np.float32), 48000).shape == (0,)


# ---------------------------------------------------------------- resampler
@pytest.mark.parametrize("rate", [48000, 44100, 8000])
def test_resampling_keeps_a_1_khz_tone(rate):
    y = _audio.resample(sine(1000.0, rate), rate)
    assert y.dtype == np.float32 and y.ndim == 1 and len(y) == SR
    middle = y[2000:14000]                  # away from the ends, where the filter sees the silence around the tone
    assert abs(peak_hz(middle) - 1000.0) <= 2.0
    assert abs(level_db(middle, 0.5)) < 1.0


@pytest.mark.parametrize("rate", [48000, 44100, 8000])
def test_resampled_tone_is_not_moved_in_time(rate):
    # One sample of delay at 16 kHz would be an error of 0.2 here.
    y = _audio.resample(sine(1000.0, rate), rate)
    assert float(np.abs(y - sine(1000.0, SR))[2000:14000].max()) < 0.02


def test_constant_level_passes_unchanged():
    y = _audio.resample(np.full(44100, 0.25, np.float32), 44100)
    np.testing.assert_allclose(y[2000:14000], 0.25, rtol=0, atol=1e-4)


def test_tone_above_the_new_nyquist_is_removed():
    # 10 kHz cannot exist at 16 kHz; left in, it would fold down to 6 kHz, into the speech band.
    y = _audio.resample(sine(10000.0, 48000), 48000)
    assert level_db(y[2000:14000], 0.5) < -40.0


def test_16_khz_passes_through_untouched():
    x = sine(1000.0, SR)
    y = _audio.resample(x, SR)
    assert np.array_equal(y, x) and np.shares_memory(y, x)
    wide = _audio.resample(x.astype(np.float64), SR)
    assert wide.dtype == np.float32 and np.array_equal(wide, x)
    resampler = _audio.Resampler(SR)
    assert resampler.same and (resampler.up, resampler.down) == (1, 1) and resampler.output_length(123) == 123
    assert np.array_equal(resampler.feed(x), x) and resampler.flush().shape == (0,)


@pytest.mark.parametrize("rate, up, down", [(48000, 1, 3), (44100, 160, 441), (8000, 2, 1), (22050, 320, 441)])
def test_usual_rates_are_exact(rate, up, down):
    resampler = _audio.Resampler(rate)
    assert (resampler.up, resampler.down, resampler.same) == (up, down, False)


@pytest.mark.parametrize("n", [0, 1, 2, 441, 1000, 44101])
def test_output_length(n):
    expected = -(-n * 160 // 441)                         # ceil(n * 16000 / 44100)
    assert _audio.Resampler(44100).output_length(n) == expected
    assert len(_audio.resample(np.zeros(n, np.float32), 44100)) == expected


def test_blocks_give_the_samples_of_one_pass(monkeypatch):
    x = noise(3 * 44100)
    whole = _audio.resample(x, 44100)
    resampler = _audio.Resampler(44100)
    parts, start = [], 0
    for size in (1, 7, 100, 5000, 20000, len(x)):          # the last one takes what is left
        parts.append(resampler.feed(x[start:start + size]))
        start += size
    parts.append(resampler.flush())
    pieces = np.concatenate(parts)
    assert pieces.dtype == np.float32 and len(pieces) == len(whole) == resampler.output_length(len(x)) == 48000
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-6)
    # resample() itself, with its own block sizes made small: many blocks instead of one
    monkeypatch.setattr(_audio, "_READ_FRAMES", 4096)
    monkeypatch.setattr(_audio, "_BLOCK_ELEMENTS", 1 << 12)
    np.testing.assert_allclose(_audio.resample(x, 44100), whole, rtol=0, atol=1e-6)


def test_resample_takes_one_channel_and_a_real_rate():
    with pytest.raises(ValueError, match="resample takes a 1-D array of samples"):
        _audio.resample(np.zeros((10, 2), np.float32), 48000)
    with pytest.raises(ValueError, match="sample_rate must be a positive number of Hz"):
        _audio.resample(np.zeros(10, np.float32), 0)


# ---------------------------------------------------------------- files
def test_load_reads_mixes_and_resamples(tmp_path, monkeypatch):
    import soundfile as sf
    left, right = sine(300.0, 8000), sine(500.0, 8000, level=0.25)
    mono = ((left.astype(np.float64) + right) / 2).astype(np.float32)
    path = tmp_path / "two.wav"
    sf.write(str(path), np.stack([left, right], axis=1), 8000, subtype="FLOAT")

    audio, rate = _audio.load(path)                         # the file's own rate
    assert rate == 8000 and audio.dtype == np.float32 and audio.shape == (8000,)
    np.testing.assert_allclose(audio, mono, rtol=0, atol=1e-6)

    audio16, rate16 = _audio.load(str(path), SR)
    assert rate16 == SR and audio16.dtype == np.float32 and audio16.shape == (SR,)
    np.testing.assert_allclose(audio16, _audio.resample(mono, 8000), rtol=0, atol=1e-5)

    monkeypatch.setattr(_audio, "_READ_FRAMES", 1000)       # the file is now read in eight blocks
    np.testing.assert_allclose(_audio.load(path, SR)[0], audio16, rtol=0, atol=1e-6)
    np.testing.assert_allclose(_audio.load(path)[0], audio, rtol=0, atol=0)


def test_load_at_16_khz_is_the_file(tmp_path):
    import soundfile as sf
    x = sine(440.0, SR, 0.5)
    path = tmp_path / "one.wav"
    sf.write(str(path), x, SR, subtype="FLOAT")
    for audio, rate in (_audio.load(path), _audio.load(path, SR)):
        assert rate == SR and np.array_equal(audio, x)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="audio file not found: .*nothing.wav"):
        _audio.load(tmp_path / "nothing.wav")
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        _audio.load(tmp_path)                               # a folder


# ---------------------------------------------------------------- quiet point
def loud_with_gaps(n, *gaps):
    x = np.full(n, 0.5, np.float32)
    for start, end in gaps:
        x[start:end] = 0.0
    return x


def test_cut_goes_to_the_middle_of_the_quietest_300_ms():
    x = loud_with_gaps(SR, (4800, 9600))
    assert _audio.quiet_cut(x, 0, SR) == 7200
    assert _audio.quiet_cut(x, 3200, SR) == 7200            # frames are counted from the start of the range
    quieter = loud_with_gaps(SR)
    quieter[4800:9600] = 0.01                               # not silent, only the quietest place
    assert _audio.quiet_cut(quieter, 0, SR) == 7200


def test_a_pause_of_300_ms_wins_over_a_shorter_silent_dip():
    x = loud_with_gaps(SR, (1600, 3200))                    # 100 ms of silence
    x[8000:12800] = 0.05                                    # 300 ms that are quiet, not silent
    assert _audio.quiet_cut(x, 0, SR) == 10400


def test_cut_lands_inside_a_gap_that_is_off_the_frame_grid():
    x = loud_with_gaps(SR, (6500, 11500))
    assert 6500 < _audio.quiet_cut(x, 0, SR) < 11500


def test_first_of_equally_quiet_places_wins():
    assert _audio.quiet_cut(loud_with_gaps(SR, (1600, 6400), (9600, 14400)), 0, SR) == 4000
    assert _audio.quiet_cut(loud_with_gaps(SR), 4000, 12000) == 6400      # no quiet place at all: the first one


def test_hard_cut_when_the_range_is_shorter_than_300_ms():
    x = loud_with_gaps(SR, (320, 640))
    assert _audio.quiet_cut(x, 100, 4899) == 4899           # fourteen whole frames: nowhere to look
    assert _audio.quiet_cut(x, 100, 100) == 100
    assert _audio.quiet_cut(x, 0, 4800) == 2400             # fifteen frames: one place, its middle


def test_range_is_kept_inside_the_audio():
    x = loud_with_gaps(SR, (4800, 9600))
    assert _audio.quiet_cut(x, -500, 10 ** 9) == 7200
    assert _audio.quiet_cut(x, SR - 100, SR + 100) == SR
    assert _audio.quiet_cut(np.zeros(0, np.float32), 0, 100) == 0


@pytest.mark.parametrize("lo, hi", [(0, SR), (1234, 9876), (8000, 16000), (15000, 16000)])
def test_cut_is_inside_the_range(lo, hi):
    cut = _audio.quiet_cut(noise(SR, seed=3), lo, hi)
    assert lo < cut <= hi
