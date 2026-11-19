"""Turkish is not enabled (kept in the library, returns KT08_ERR_UNSUPPORTED). Turkish text front end of Kitten TTS 2 mini (kt08_tiny_frontend_*, phonemizer v2).

Source-only checks always run. The library check needs a shared build of the C library, the espeak-ng data files
(engines/lite/mobile/fetch_phonemizer_data.sh OUT tr) and a downloaded model repository for its vocab.json:
    KITTEN_TTS08_CAPI_LIB=build/libkitten_tts08.so KITTEN_ESPEAK_DATA=phonemizer-data \
    KITTEN_TTS2_TINY_REPO=<kitten-tts-2-mini folder> python -m pytest tests/tts2_tiny/test_tiny_lang_tr.py
The library check asserts that Turkish is refused.
"""
import ctypes
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FETCH = ROOT / "engines" / "lite" / "mobile" / "fetch_phonemizer_data.sh"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
DATA = os.environ.get("KITTEN_ESPEAK_DATA")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
TR = [
    "Bir gözlemci, sokaklarında çetelerinin dolaşarak mağazaları yağmalamasıyla Bişkek'in bir \"anarşi\" durumuna saplandığını söyledi.",
    "Paraguay, Asunción'un 1537'de kurulmasından bu yana yerli kimliğini ve karakterini büyük ölçüde korumayı başardı.",
    "Araç aynı gün takriben 12.00 GMT'de kaza yerinden kaldırıldı.",
    "Keçilerin ilk kez yaklaşık 10.000 yıl önce İran'ın Zagros Dağları'nda evcilleştirildikleri tahmin edilmektedir.",
    "Amazon Nehri'nin ana kolu 6,387 km'dir (3,980 mil). Binlerce küçük nehirden su toplar.",
    "Güney Afrika Ulusal Parklarının (SANParks) da amiral gemisi olarak görülür.",
    "Akşam yemeğini yedikten sonra ne yaptın?",
]


class SourceTest(unittest.TestCase):
    def test_fetch_script_pins_turkish_data(self):
        s = FETCH.read_text(encoding="utf-8")
        for f in ("tr_rules", "tr_list", "tr_listx", "phsource/ph_turkish"):
            self.assertRegex(s, re.escape(f) + r" [0-9a-f]{64}")


@unittest.skipUnless(LIB and DATA and REPO, "needs KITTEN_TTS08_CAPI_LIB, KITTEN_ESPEAK_DATA and KITTEN_TTS2_TINY_REPO")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib = ctypes.CDLL(LIB)
        P = ctypes.c_void_p
        lib.kt08_tiny_frontend_load.argtypes = [ctypes.c_char_p] * 4 + [ctypes.POINTER(P)]
        lib.kt08_tiny_frontend_free.argtypes = [P]
        lib.kt08_tiny_text_to_phonemes.argtypes = [P, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.kt08_free.argtypes = [ctypes.c_void_p]
        lib.kt08_last_error.restype = ctypes.c_char_p
        cls.lib = lib

    def _fe(self, version):
        fe = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_frontend_load(version.encode(), os.path.join(REPO, "vocab.json").encode(), None,
                                              DATA.encode(), ctypes.byref(fe))
        self.assertEqual(rc, 0, self.lib.kt08_last_error())
        return fe

    def _run(self, fe, text, lang):
        out = ctypes.c_void_p()
        rc = self.lib.kt08_tiny_text_to_phonemes(fe, text.encode(), lang.encode(), ctypes.byref(out))
        if rc:
            return rc, None
        s = ctypes.string_at(out.value).decode()
        self.lib.kt08_free(out)
        return 0, (s.split("\n") if s else [])

    def test_not_enabled(self):
        # Turkish is NOT switched on: on fresh text (WMT news + Wikipedia) it matched the package on 97.84 % only.
        # The rules stay in the library; the language returns KT08_ERR_UNSUPPORTED until it passes again.
        fe = self._fe("v2")
        try:
            for text in TR[:2]:
                rc, _ = self._run(fe, text, "tr")
                self.assertEqual(rc, 6, text)   # KT08_ERR_UNSUPPORTED
        finally:
            self.lib.kt08_tiny_frontend_free(fe)

    def test_not_for_nano_pico(self):
        fe = self._fe("v1")   # Turkish is a mini (phonemizer v2) language only
        try:
            rc, _ = self._run(fe, "Merhaba.", "tr")
            self.assertEqual(rc, 6)   # KT08_ERR_UNSUPPORTED
        finally:
            self.lib.kt08_tiny_frontend_free(fe)


if __name__ == "__main__":
    unittest.main()
