"""stream()'s chunks (engines/lite/engine/src/stream/chunker.hpp), the same on every path and every machine.

The chunker is one C++ function behind both the Python package's stream() (through the engine module, on the CPU,
Metal and CUDA) and the C library's kt08_stream. Rule: with chunk_chars=N, whole sentences packed into chunks of at most
N characters (a longer sentence cut at spaces); with chunk_chars=None (0 in C), the model's default: Kitten TTS 0.8 one
sentence per chunk, Kitten TTS 2 mini/nano/pico 380 characters. Nothing is measured.
Checked here:
  - the rule itself (against a plain-Python statement of it), the empty and white-space cases, the model caps
    (Kitten TTS 0.8: 400 characters; Kitten TTS 2: 350 phonemes, cut by the front end as generate() cuts), the
    silence trims;
  - with a real model (KITTEN_TTS2_TINY_REPO, KITTEN_TTS08_STREAM_MODEL): stream() follows the chunks and differs from
    generate() only by silence at chunk edges;
  - with the C library (KITTEN_TTS08_CAPI_LIB + KITTEN_TINY_FE_DATA / KITTEN_TTS08_PHONEMIZER_DATA): the C library,
    the engine module and the Python path give the same chunks for the same text and chunk_chars.
"""
import ctypes
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kitten_inference import _family, _stream

LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
FE_DATA = os.environ.get("KITTEN_TINY_FE_DATA")
T08 = os.environ.get("KITTEN_TTS08_STREAM_MODEL")
PH08 = os.environ.get("KITTEN_TTS08_PHONEMIZER_DATA")
THR = 10 ** (-50 / 20)
CHUNK_ERROR = "chunk_chars must be None \\(the model's default\\) or a positive whole number of characters"

TEXTS = [
    "Sure, your table for four is booked for seven thirty tonight. Please arrive a few minutes early, since the kitchen "
    "gets busy after eight. We will hold the table for fifteen minutes. If your plans change, just call us.",
    "Hello there. Is it streaming? Yes!  It is.",
    "One very long sentence " + " ".join(f"word{i}" for i in range(120)) + " ends here. Short one.",
    "",
    "   \n\t ",
]


def engine():
    from kitten_inference.tts08 import cpp_model
    return cpp_model._import_engine()


def reference_pack(sentences, chunk_chars, max_chars=0, punctuate=False):
    """The rule in plain Python (what chunker.cpp must do)."""
    from kitten_inference.tts08.preprocess import ensure_punctuation
    cut = ensure_punctuation if punctuate else (lambda x: x)
    limit = chunk_chars if chunk_chars and chunk_chars > 0 else 0
    if max_chars and limit > max_chars:
        limit = max_chars
    out, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not limit:
            out.append(s)
            continue
        if len(s) > limit:
            if cur:
                out.append(cur)
            cur = ""
            piece = ""
            for w in s.split():
                if piece and len(piece) + 1 + len(w) > limit:
                    out.append(cut(piece))
                    piece = ""
                piece = f"{piece} {w}" if piece else w
            if piece:
                out.append(cut(piece))
            continue
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def sentences_of(text):
    from kitten_inference.tts2_tiny.frontend import SENT_MULTI
    return SENT_MULTI.split(text)


class ChunkerTest(unittest.TestCase):
    def setUp(self):
        self.st = engine().stream

    def test_one_sentence_per_chunk(self):
        # The chunker's own 0 (ONE_SENTENCE): one sentence per chunk, Kitten TTS 0.8's default. Users never pass it:
        # chunk_chars None (0 in C) asks for the model's default, which each family turns into the chunker's value.
        pack = self.st.pack_sentences
        s = ["Hello there.", " Is it streaming? ", "", "Yes!"]
        self.assertEqual(self.st.ONE_SENTENCE, 0)
        self.assertEqual(pack(s), ["Hello there.", "Is it streaming?", "Yes!"])
        self.assertEqual(pack(s, self.st.ONE_SENTENCE), pack(s))
        self.assertEqual(pack(s, -3), pack(s))

    def test_chunk_chars_packs_whole_sentences(self):
        pack = self.st.pack_sentences
        s = ["Aa bb.", "Cc dd.", "Ee ff gg.", "Hh."]                 # 6, 6, 9, 3 characters
        self.assertEqual(pack(s, 13), ["Aa bb. Cc dd.", "Ee ff gg. Hh."])   # 6 + 1 + 6 = 13 fits; 9 + 1 + 3 = 13
        self.assertEqual(pack(s, 12), ["Aa bb.", "Cc dd.", "Ee ff gg.", "Hh."])   # 9 + 1 + 3 = 13 > 12
        self.assertEqual(pack(s, 1000), ["Aa bb. Cc dd. Ee ff gg. Hh."])
        for n in (1, 5, 6, 7, 20, 40):
            for c in pack(s, n):
                self.assertTrue(len(c) <= n or " " not in c, (n, c))   # over n only for one word longer than n

    def test_long_sentence_split_at_word_boundaries(self):
        pack = self.st.pack_sentences
        long = " ".join(f"w{i}" for i in range(50))
        got = pack(["Short.", long, "End."], 30)
        self.assertEqual(got[0], "Short.")                      # flushed before the long sentence
        self.assertEqual(got[-1], "End.")                       # the long sentence's pieces stay apart from the next
        self.assertEqual(" ".join(got[1:-1]).split(), long.split())   # every word, in order, none cut
        self.assertTrue(all(len(c) <= 30 for c in got))
        self.assertEqual(pack(["a " + "x" * 50 + " b"], 10), ["a", "x" * 50, "b"])   # an over-long word stays whole

    def test_empty_and_white_space_text(self):
        pack = self.st.pack_sentences
        for n in (0, 1, 50):
            self.assertEqual(pack([], n), [])
            self.assertEqual(pack(["", "  ", "\n\t"], n), [])
            self.assertEqual(pack(sentences_of(""), n), [])
            self.assertEqual(pack(sentences_of("   \n\t "), n), [])

    def test_matches_the_rule_on_many_texts(self):
        pack = self.st.pack_sentences
        rng = random.Random(7)
        words = ["a", "to", "the", "streaming", "Dr.", "ünïcödé", "ʃˈʊɹ", "end.", "now!", "why?", "é…", "ok."]
        for _ in range(300):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(0, 80)))
            sents = sentences_of(text)
            for n in (None, 1, 8, 25, 60, 400):
                for cap in (0, 40):
                    self.assertEqual(pack(sents, n or 0, cap), reference_pack(sents, n, cap), (text, n, cap))
                    self.assertEqual(pack(sents, n or 0, cap, True), reference_pack(sents, n, cap, True), (text, n, cap))

    def test_same_chunks_every_call(self):
        # Deterministic: nothing is measured, so the chunks do not depend on the machine, the time or earlier calls.
        pack = self.st.pack_sentences
        for t in TEXTS:
            for n in (0, 40, 120):
                first = pack(sentences_of(t), n)
                self.assertEqual([pack(sentences_of(t), n) for _ in range(5)], [first] * 5)

    def test_sentence_split_is_the_front_ends(self):
        # The C library cuts Kitten TTS 2 text with this port of the package's SENT_MULTI.
        for t in TEXTS + ["a. b", "a.b c", "ア。イ", "Wait… what?  Yes.\nNo!", "x。 y", "।a । b", "end."]:
            self.assertEqual(self.st.split_sentences(t), sentences_of(t), t)

    def test_abbreviations_kitten_tts_08(self):
        # Kitten TTS 0.8 sentences come from its own splitter (chunk_text), which keeps common abbreviations inside a
        # sentence; the chunker only packs them.
        from kitten_inference.tts08.preprocess import chunk_text
        sents = chunk_text("Dr. Smith met Mr. Jones at 5 p.m. today. They talked. It was fine.")
        self.assertEqual(len(sents), 3, sents)
        self.assertEqual(self.st.pack_sentences(sents), sents)
        self.assertEqual(self.st.pack_sentences(sents, 1000, self.st.TTS08_MAX_CHARS), [" ".join(sents)])

    def test_kitten_tts_08_cut_pieces_end_like_chunk_text(self):
        # A Kitten TTS 0.8 sentence cut at spaces: each piece ends with punctuation, as chunk_text's pieces do.
        got = self.st.pack_sentences(["One two three four five six seven."], 12, self.st.TTS08_MAX_CHARS, True)
        self.assertEqual(got, ["One two,", "three four,", "five six,", "seven."])
        self.assertEqual(self.st.pack_sentences(["One two three."], 7), ["One two", "three."])   # Kitten TTS 2: as cut

    def test_kitten_tts_08_cap_400_characters(self):
        st = self.st
        self.assertEqual(st.TTS08_MAX_CHARS, 400)
        sents = [("word " * 70).strip() + "."] * 4                 # 350 characters each
        for c in st.pack_sentences(sents, 5000, st.TTS08_MAX_CHARS):
            self.assertLessEqual(len(c), 400)

    def test_defaults_per_family(self):
        # Kitten TTS 2 mini/nano/pico: 380 characters (KittenTTS 2's chunk_chars), the same constant in the C library;
        # Kitten TTS 0.8: one sentence per chunk (its chunk_text). Python and C defaults come from one place each.
        import inspect
        from kitten_inference.tts08.cpp_model import KittenTTS_1_Cpp as T08Model
        from kitten_inference.tts2_tiny import model as tiny
        self.assertEqual(self.st.TINY_CHUNK_CHARS, 380)
        self.assertEqual(tiny.STREAM_CHUNK_CHARS, self.st.TINY_CHUNK_CHARS)
        # One meaning everywhere: None (0 in C) is the model's default; there is no other sentinel.
        for cls in (tiny.KittenTTS2Tiny, T08Model):
            sig = inspect.signature(cls.generate_stream).parameters
            self.assertEqual((sig["chunk_chars"].default, sig["trim_lead"].default), (None, True), cls)
            self.assertIsNone(inspect.signature(cls.stream_chunks).parameters["chunk_chars"].default, cls)
        h = (Path(__file__).resolve().parents[1] / "engines/lite/engine/include/kitten_tts08.h").read_text()
        self.assertIn("#define KT08_STREAM_TINY_CHUNK_CHARS 380", h)
        self.assertNotIn("ONE_SENTENCE", h)

    def test_lead_trim_drops_only_silence(self):
        st = self.st
        a = np.zeros(24000, dtype=np.float32)
        a[10000] = 0.004       # above -50 dBFS (0.00316)
        a[5000] = 0.003        # below
        self.assertEqual(st.lead_trim_offset(a), 10000 - 480)
        self.assertEqual(st.lead_trim_offset(np.zeros(100, dtype=np.float32)), 0)
        a[100] = 0.5
        self.assertEqual(st.lead_trim_offset(a), 0)  # less than 20 ms in: nothing to drop

    def test_end_trim_drops_only_silence(self):
        st = self.st
        rng = np.random.default_rng(0)
        a = np.concatenate([np.zeros(2400), rng.uniform(-0.5, 0.5, 24000), rng.uniform(-THR, THR, 9600) * 0.9,
                            np.zeros(4800)]).astype(np.float32)
        last = int(np.nonzero(np.abs(a) >= THR)[0][-1])
        self.assertEqual(st.TAIL_KEEP_MS, 30.0)
        k = st.tail_keep(a, 24000)
        self.assertEqual(k, last + 1 + 720)                     # 30 ms after the last audible sample
        self.assertTrue((np.abs(a[k:]) < THR).all())            # only silence dropped
        self.assertEqual(st.tail_keep(np.zeros(100, np.float32), 24000), 100)   # nothing audible: kept whole
        self.assertEqual(st.tail_keep(a, 24000, 0.0), len(a))   # 0: off


class OptionTest(unittest.TestCase):
    def test_chunk_chars_validation(self):
        self.assertIsNone(_family.check_chunk_chars(None))
        self.assertEqual(_family.check_chunk_chars(80), 80)
        self.assertEqual(_family.check_chunk_chars(np.int64(5)), 5)
        self.assertEqual(_family.check_chunk_chars(10 ** 12), 2 ** 31 - 1)   # past the chunker's int: the largest
        for bad in (0, -1, 2.5, "80", True, [80]):
            with self.assertRaisesRegex(ValueError, CHUNK_ERROR):
                _family.check_chunk_chars(bad)

    def test_driver_order_and_edges(self):
        # _stream.run: a chunk is phonemised only when it is next; first/last mark the stream's edges (a Kitten TTS 2
        # chunk past 350 phonemes gives several pieces); empty chunks are skipped. As kt08_stream_next.
        log = []

        def phonemize(c):
            log.append(("ph", c))
            return {"A": ["a"], "B": ["b1", "b2"], "C": [], "D": ["d"]}[c]

        def synth(ph, text, first, last):
            log.append(("syn", ph, first, last))
            return ph

        it = _stream.run(["A", "B", "C", "D"], phonemize, synth)
        self.assertEqual(next(it), "a")
        self.assertEqual(log, [("ph", "A"), ("syn", "a", True, False)])   # B not phonemised yet
        self.assertEqual(list(it), ["b1", "b2", "d"])
        self.assertEqual([x for x in log if x[0] == "syn"],
                         [("syn", "a", True, False), ("syn", "b1", False, False), ("syn", "b2", False, False),
                          ("syn", "d", False, True)])
        self.assertEqual(list(_stream.run([], phonemize, synth)), [])

    def test_trim_lead_validation(self):
        self.assertTrue(_stream.check_trim_lead(True))
        self.assertFalse(_stream.check_trim_lead(False))
        for bad in (None, 1, "yes"):
            with self.assertRaises(TypeError):
                _stream.check_trim_lead(bad)


@unittest.skipUnless(REPO, "KITTEN_TTS2_TINY_REPO is not set")
class TinyModelStreamTest(unittest.TestCase):
    def test_stream_follows_the_chunks(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        tts = KittenTTS2Tiny(REPO, backend="cpu")
        text = TEXTS[0]
        default = list(tts.generate_stream(text))
        none = list(tts.generate_stream(text, chunk_chars=None))          # None: the model's default, 380
        at380 = list(tts.generate_stream(text, chunk_chars=380))
        per = list(tts.generate_stream(text, chunk_chars=78))             # 78: one sentence per chunk for this text
        kept = list(tts.generate_stream(text, chunk_chars=78, trim_lead=False))
        packed = list(tts.generate_stream(text, chunk_chars=120))
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError, CHUNK_ERROR):
                list(tts.generate_stream(text, chunk_chars=bad))
        # None and 380 are the same request (the same chunks, checked below); the audio is compared within 1e-4, not
        # bit for bit, because the engine's rounding can depend on the calls made before in the same process (seen on
        # the previous commit too); the fresh-process identity runs check bit for bit.
        self.assertEqual([len(c) for c in none], [len(c) for c in default])
        self.assertEqual([len(c) for c in at380], [len(c) for c in default])
        for a, b, c in zip(default, none, at380):
            self.assertLess(max(float(np.max(np.abs(a - b))), float(np.max(np.abs(a - c)))), 1e-4)
        gen = tts.generate(text)
        # Default (380 characters): this text is one chunk, as in generate(); the stream is generate()'s audio without
        # its silent start (an exact suffix, nothing at or above -50 dBFS cut).
        self.assertEqual(len(tts.stream_chunks(text)), 1)
        self.assertEqual(len(default), 1)
        cut = len(gen) - len(default[0])
        self.assertGreater(cut, 0)
        self.assertTrue(np.array_equal(gen[cut:], default[0]))
        self.assertLess(float(np.max(np.abs(gen[:cut]))), THR)
        self.assertEqual(tts.stream_chunks(text, chunk_chars=None), tts.stream_chunks(text, chunk_chars=380))
        self.assertEqual(len(tts.stream_chunks(text, chunk_chars=78)), 4)   # one sentence per chunk
        self.assertEqual(len(per), 4)
        self.assertEqual(len(packed), len(tts.stream_chunks(text, chunk_chars=120)))
        self.assertLess(len(packed), len(per))
        for i, c in enumerate(per):
            self.assertEqual(int(np.argmax(np.abs(c) >= THR)), 480, i)     # every chunk starts 20 ms before its sound
            if i < len(per) - 1:                                           # 30 ms of end silence when another follows
                last = int(np.nonzero(np.abs(c) >= THR)[0][-1])
                self.assertLessEqual(len(c) - (last + 1), 720, i)
        self.assertGreater(int(np.argmax(np.abs(kept[0]) >= THR)), 480)   # trim_lead=False keeps the first start
        self.assertTrue(np.array_equal(kept[1], per[1]))                   # later chunks as with the trim

    def test_long_text_chunks(self):
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        tts = KittenTTS2Tiny(REPO, backend="cpu")
        text = " ".join([TEXTS[0]] * 10)
        ch = tts.stream_chunks(text, chunk_chars=78)   # every sentence alone: 78 holds the longest (75), not two
        self.assertEqual(len(ch), 40)
        self.assertTrue(ch[0].startswith("Sure, your table"))
        self.assertLessEqual(len(ch[0]), 100)
        ch = tts.stream_chunks(text)
        self.assertEqual(ch, tts.stream_chunks(text, chunk_chars=None))
        self.assertTrue(all(len(c) <= 380 for c in ch))
        self.assertEqual(" ".join(ch).split(), " ".join(tts.stream_chunks(text, chunk_chars=78)).split())

    def test_chunk_past_350_phonemes_is_cut_at_sentence_ends(self):
        # A chunk the model cannot take in one call is cut as generate() cuts long text (front_end: sentence ends,
        # then spaces, packed up to 350 phonemes), not in the middle of a sentence.
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        from kitten_inference.tts2_tiny.frontend import front_end
        tts = KittenTTS2Tiny(REPO, backend="cpu")
        lang = tts.default_language
        text = " ".join([TEXTS[0]] * 3)
        chunks, g2p, fold = tts._text_chunks(text, lang, 2000)
        self.assertEqual(len(chunks), 1)
        pieces = front_end(g2p, chunks[0], normalise=False)[0]
        pieces = [fold(p) for p in pieces] if fold else pieces
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 350 for p in pieces))
        self.assertEqual(pieces, tts.phonemize(text, lang))                # generate()'s pieces
        self.assertEqual(len(list(tts.generate_stream(text, chunk_chars=2000))), len(pieces))


@unittest.skipUnless(T08, "KITTEN_TTS08_STREAM_MODEL (e.g. KittenML/kitten-tts-nano-0.8) is not set")
class Tts08StreamTest(unittest.TestCase):
    def test_default_stream_is_generate_sentence_by_sentence(self):
        from kitten_inference.tts08 import KittenTTS as T
        from kitten_inference.tts08.preprocess import chunk_text
        tts = T(T08).model
        text = "Sure, your table for four is booked. Please arrive a few minutes early. We will hold it for you."
        new = list(tts.generate_stream(text))
        packed = list(tts.generate_stream(text, chunk_chars=200))
        ref = [tts.generate_single_chunk(c, "expr-voice-5-m", 1.0) for c in chunk_text(tts.preprocessor(text))]
        self.assertEqual(len(new), 3)
        self.assertEqual(len(ref), 3)
        cut = len(ref[0]) - len(new[0])
        self.assertGreater(cut, 0)
        self.assertTrue(np.array_equal(ref[0][cut:], new[0]))             # first: lead trimmed only
        self.assertLess(float(np.max(np.abs(ref[0][:cut]))), THR)
        for a, b in zip(ref[1:], new[1:]):
            self.assertTrue(np.array_equal(a, b))                          # later chunks: generate()'s
        self.assertEqual(len(packed), len(tts.stream_chunks(text, chunk_chars=200)))
        self.assertLess(len(packed), 3)
        for c in tts.stream_chunks("One two three four five six seven eight nine ten.", chunk_chars=20):
            self.assertIn(c[-1], ".,")                                     # cut pieces end like chunk_text's


def _clib():
    lib = ctypes.CDLL(LIB)
    P, S, F = ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)
    lib.kt08_last_error.restype = ctypes.c_char_p
    lib.kt08_free.argtypes = [P]
    lib.kt08_stream_options_init.argtypes = [ctypes.POINTER(StreamOptions)]
    lib.kt08_stream_open.argtypes = [P, P, P, ctypes.c_char_p, ctypes.c_char_p, F, S, ctypes.c_float,
                                     ctypes.POINTER(StreamOptions), ctypes.POINTER(P)]
    lib.kt08_stream_open_tts08.argtypes = [P, P, ctypes.c_char_p, F, S, ctypes.c_float, ctypes.POINTER(StreamOptions),
                                           ctypes.POINTER(P)]
    lib.kt08_stream_next.argtypes = [P, ctypes.POINTER(F), ctypes.POINTER(S), ctypes.POINTER(ctypes.c_int)]
    lib.kt08_stream_chunks.argtypes = [P, ctypes.POINTER(P)]
    lib.kt08_stream_free.argtypes = [P]
    lib.kt08_tiny_frontend_load.argtypes = [ctypes.c_char_p] * 4 + [ctypes.POINTER(P)]
    lib.kt08_tiny_frontend_free.argtypes = [P]
    lib.kt08_phonemizer_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(P)]
    lib.kt08_phonemizer_free.argtypes = [P]
    return lib


class StreamOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("chunk_chars", ctypes.c_int32), ("trim_first_pause", ctypes.c_int32)]


def _c_chunks(lib, st):
    out = ctypes.c_void_p()
    assert lib.kt08_stream_chunks(st, ctypes.byref(out)) == 0
    s = ctypes.string_at(out).decode()
    lib.kt08_free(out)
    return s.split("\n") if s else []


def _c_audio(lib, st):
    got, last = [], ctypes.c_int(0)
    out, n = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
    while not last.value:
        assert lib.kt08_stream_next(st, ctypes.byref(out), ctypes.byref(n), ctypes.byref(last)) == 0, lib.kt08_last_error()
        if n.value:
            got.append(np.ctypeslib.as_array(out, shape=(n.value,)).copy())
        lib.kt08_free(out)
    return got


@unittest.skipUnless(LIB and REPO and FE_DATA, "KITTEN_TTS08_CAPI_LIB, KITTEN_TTS2_TINY_REPO and KITTEN_TINY_FE_DATA are needed")
class CLibrarySameChunksTest(unittest.TestCase):
    def test_c_library_module_and_python_give_the_same_chunks(self):
        import json
        from kitten_inference.tts2_tiny import KittenTTS2Tiny
        lib = _clib()
        tts = KittenTTS2Tiny(REPO, backend="cpu")
        lang = tts.default_language
        cfg = json.loads((Path(REPO) / "config.json").read_text())
        tmp = tempfile.mkdtemp()
        try:
            g2p = Path(tmp) / "g2p"
            shutil.copytree(Path(REPO) / cfg.get("g2p_dir", "g2p"), g2p)
            for f in Path(FE_DATA).iterdir():
                shutil.copy(f, g2p / f.name)
            fe, model, v = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            vocab = str(Path(REPO) / "vocab.json").encode()
            self.assertEqual(lib.kt08_tiny_frontend_load(cfg.get("phonemizer", "v1").encode(), vocab, str(g2p).encode(),
                                                         None, ctypes.byref(fe)), 0, lib.kt08_last_error())
            arch, weights = tts._converted(tts.config["languages"][lang]["model"])
            self.assertEqual(lib.kt08_model_load(arch.encode(), weights.encode(), None, ctypes.byref(model)), 0)
            self.assertEqual(lib.kt08_vocab_load(vocab, ctypes.byref(v)), 0)
            _, vkey = tts.resolve_voice(None, lang)
            vp = os.path.join(tmp, "v.bin")
            tts._pack(lang, vkey).astype("<f4").tofile(vp)
            voice, rows = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
            self.assertEqual(lib.kt08_voice_load(vp.encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
            st_mod = engine().stream
            g2p_py = tts.frontend.g2p(lang)
            long_text = " ".join([TEXTS[0]] * 3)   # over 350 phonemes in one chunk at chunk_chars=2000
            for text in TEXTS + ["Dr. Smith paid $12.50 for 3 books on March 3rd, 2024. He was happy.", long_text]:
                norm = g2p_py._normalize(text) if hasattr(g2p_py, "_normalize") else text
                for n in (None, 1, 40, 120, 380, 1000, 2000):   # None: the model's default (0 in C, 380)
                    so = StreamOptions()
                    lib.kt08_stream_options_init(ctypes.byref(so))
                    self.assertEqual((so.chunk_chars, so.trim_first_pause), (0, 1))   # 0: the model's default
                    so.chunk_chars = 0 if n is None else n
                    py_n = n
                    st = ctypes.c_void_p()
                    self.assertEqual(lib.kt08_stream_open(model, v, fe, text.encode(), lang.encode(), voice, rows.value,
                                                          1.0, ctypes.byref(so), ctypes.byref(st)), 0,
                                     lib.kt08_last_error())
                    c = _c_chunks(lib, st)
                    if (n in (None, 120) and text == TEXTS[1]) or (n == 2000 and text == long_text):
                        audio = _c_audio(lib, st)
                        py_audio = list(tts.generate_stream(text, chunk_chars=py_n))
                        self.assertEqual(len(audio), len(py_audio))   # the same pieces on both paths
                        for i, a in enumerate(audio):
                            self.assertEqual(int(np.argmax(np.abs(a) >= THR)), 480)
                    lib.kt08_stream_free(st)
                    self.assertEqual(c, tts.stream_chunks(text, chunk_chars=py_n), (text, n))              # Python
                    self.assertEqual(c, st_mod.pack_sentences(sentences_of(norm), n or st_mod.TINY_CHUNK_CHARS),
                                     (text, n))                                                                # module
                    if n is None:
                        default_c = c
                    elif n == 380:
                        self.assertEqual(c, default_c, text)   # 0 is 380 for mini, nano and pico
            if os.environ.get("KITTEN_ESPEAK_DATA") and "es" in tts.languages:
                self._spanish(lib, tts, cfg, vocab, g2p, model, v, rows)
            for bad in (-1, -2):   # one meaning: 0 or a positive N; -1 is not a sentinel
                so.chunk_chars = bad
                st = ctypes.c_void_p()
                self.assertEqual(lib.kt08_stream_open(model, v, fe, b"Hi.", lang.encode(), voice, rows.value, 1.0,
                                                      ctypes.byref(so), ctypes.byref(st)), 1)     # KT08_ERR_ARGUMENT
                self.assertIsNone(st.value)
                self.assertEqual(lib.kt08_last_error().decode(), "chunk_chars must be 0 (the model's default) or a "
                                                                 "positive whole number of characters")
            so.chunk_chars = 0
            lib.kt08_free(voice)
            lib.kt08_vocab_free(v)
            lib.kt08_model_free(model)
            lib.kt08_tiny_frontend_free(fe)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _spanish(self, lib, tts, cfg, vocab, g2p, model, v, rows):
        # Another language, several sentences past 350 phonemes in one chunk: the C library and the Python package
        # give the same text chunks and the same pieces (both cut such a chunk at sentence ends, as generate()).
        try:
            tts.phonemize("Hola.", "es")
        except Exception as exc:  # the Python front end for Spanish is not installed here
            self.skipTest(f"Spanish front end unavailable in Python: {exc}")
        fe = ctypes.c_void_p()
        self.assertEqual(lib.kt08_tiny_frontend_load(cfg.get("phonemizer", "v1").encode(), vocab, str(g2p).encode(),
                                                     os.environ["KITTEN_ESPEAK_DATA"].encode(), ctypes.byref(fe)), 0,
                         lib.kt08_last_error())
        _, vkey = tts.resolve_voice(None, "es")
        with tempfile.TemporaryDirectory() as d:
            vp = os.path.join(d, "v.bin")
            tts._pack("es", vkey).astype("<f4").tofile(vp)
            voice, r = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
            self.assertEqual(lib.kt08_voice_load(vp.encode(), ctypes.byref(voice), ctypes.byref(r)), 0)
        text = " ".join(["Buenos días. La mesa para cuatro está reservada para esta noche a las siete y media.",
                         "Por favor, lleguen unos minutos antes, porque la cocina se llena después de las ocho.",
                         "Guardaremos la mesa quince minutos. Si cambian sus planes, llámennos."] * 2)
        for n in (None, 100, 2000):
            so = StreamOptions()
            lib.kt08_stream_options_init(ctypes.byref(so))
            so.chunk_chars = 0 if n is None else n
            st = ctypes.c_void_p()
            self.assertEqual(lib.kt08_stream_open(model, v, fe, text.encode(), b"es", voice, r.value, 1.0,
                                                  ctypes.byref(so), ctypes.byref(st)), 0, lib.kt08_last_error())
            c = _c_chunks(lib, st)
            audio = _c_audio(lib, st)
            lib.kt08_stream_free(st)
            self.assertEqual(c, tts.stream_chunks(text, language="es", chunk_chars=n), n)
            py = list(tts.generate_stream(text, voice=vkey, language="es", chunk_chars=n))
            self.assertEqual(len(audio), len(py), n)
        lib.kt08_free(voice)
        lib.kt08_tiny_frontend_free(fe)


@unittest.skipUnless(LIB and T08 and PH08,
                     "KITTEN_TTS08_CAPI_LIB, KITTEN_TTS08_STREAM_MODEL and KITTEN_TTS08_PHONEMIZER_DATA are needed")
class CLibraryTts08Test(unittest.TestCase):
    def test_c_stream_matches_the_python_stream(self):
        # kt08_stream_open_tts08: the same chunks as the Python package's stream() (its sentences, or packed with
        # chunk_chars), the first without its silent start; audio equal to the Python stream's chunk by chunk when the
        # C phonemiser gives the same phonemes (byte-identical with KITTEN_TTS08_CAPI_SAME_BUILD=1, else within 1e-4
        # on Linux; other systems build the library differently: lengths within 10%).
        from kitten_inference.tts08 import KittenTTS as T
        lib = _clib()
        tts = T(T08).model
        voice_name = tts._resolve_voice("expr-voice-2-f")
        text = "Sure, your table for four is booked. Please arrive a few minutes early. We will hold it for you."
        ph = ctypes.c_void_p()
        self.assertEqual(lib.kt08_phonemizer_load(os.path.join(PH08, "en_rules").encode(),
                                                  os.path.join(PH08, "en_list").encode(), ctypes.byref(ph)), 0)
        model = ctypes.c_void_p()
        self.assertEqual(lib.kt08_model_load(tts.arch_path.encode(), tts.weights_path.encode(), None,
                                             ctypes.byref(model)), 0, lib.kt08_last_error())
        pack = (tts._voice_packs[voice_name] if voice_name in tts._voice_packs
                else np.asarray(tts._voices_npz[voice_name], dtype=np.float32).reshape(-1, 256))
        with tempfile.TemporaryDirectory() as d:
            vp = os.path.join(d, "v.bin")
            pack.astype("<f4").tofile(vp)
            voice, rows = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
            self.assertEqual(lib.kt08_voice_load(vp.encode(), ctypes.byref(voice), ctypes.byref(rows)), 0)
        speed = tts._effective_speed(voice_name, 1.0)
        for bad in (-1, -2):   # one meaning: 0 or a positive N; -1 is not a sentinel
            so = StreamOptions()
            lib.kt08_stream_options_init(ctypes.byref(so))
            so.chunk_chars = bad
            st = ctypes.c_void_p()
            self.assertEqual(lib.kt08_stream_open_tts08(model, ph, text.encode(), voice, rows.value, speed,
                                                        ctypes.byref(so), ctypes.byref(st)), 1)   # KT08_ERR_ARGUMENT
            self.assertEqual(lib.kt08_last_error().decode(), "chunk_chars must be 0 (the model's default) or a "
                                                             "positive whole number of characters")
        for n in (None, 60, 1000):
            so = StreamOptions()
            lib.kt08_stream_options_init(ctypes.byref(so))
            so.chunk_chars = 0 if n is None else n   # 0: the model's default, one sentence per chunk for Kitten TTS 0.8
            st = ctypes.c_void_p()
            self.assertEqual(lib.kt08_stream_open_tts08(model, ph, text.encode(), voice, rows.value, speed,
                                                        ctypes.byref(so), ctypes.byref(st)), 0, lib.kt08_last_error())
            chunks = _c_chunks(lib, st)
            self.assertEqual(chunks, tts.stream_chunks(text, chunk_chars=n), n)
            if n is None:
                self.assertEqual(len(chunks), 3)   # one sentence per chunk
            got = _c_audio(lib, st)
            lib.kt08_stream_free(st)
            py = list(tts.generate_stream(text, voice=voice_name, clean_text=True, chunk_chars=n))
            self.assertEqual(len(got), len(chunks))
            self.assertEqual(len(py), len(got))
            same_ph = all(tts.phonemizer.phonemize([c])[0] == _c_phonemes(lib, ph, c) for c in chunks)
            same_build = os.environ.get("KITTEN_TTS08_CAPI_SAME_BUILD") == "1"
            for a, b in zip(got, py):
                if same_ph and sys.platform.startswith("linux"):
                    self.assertEqual(len(a), len(b))
                    if same_build:
                        self.assertTrue(np.array_equal(a, b))
                    else:
                        self.assertLess(float(np.max(np.abs(a - b))), 1e-4)
                else:
                    self.assertLess(abs(len(a) - len(b)), 0.1 * len(b))
        lib.kt08_free(voice)
        lib.kt08_model_free(model)
        lib.kt08_phonemizer_free(ph)


def _c_phonemes(lib, ph, chunk):
    n = ctypes.c_size_t()
    lib.kt08_chunk_to_phonemes.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_size_t)]
    lib.kt08_chunk_to_phonemes(ph, chunk.encode(), None, 0, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    assert lib.kt08_chunk_to_phonemes(ph, chunk.encode(), buf, n.value, ctypes.byref(n)) == 0
    return buf.value.decode()


if __name__ == "__main__":
    unittest.main()
