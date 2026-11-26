"""Text front end of Kitten TTS 2 for the languages other than English, in the C library (kt08_tiny_frontend_*).

Source-only checks always run. The library check needs a shared build of the C library, the espeak-ng data files
(engines/lite/mobile/fetch_phonemizer_data.sh DIR) and a downloaded model repository for its vocab.json:
    KITTEN_TTS08_CAPI_LIB=build/libkitten_tts08.so KITTEN_ESPEAK_DATA=phonemizer-data \
    KITTEN_TTS2_TINY_REPO=<kitten-tts-2-nano folder> python -m pytest tests/tts2_tiny/test_tiny_lang_frontend.py
It compares the C library with the Python package (espeak-ng through phonemizer) on a few sentences of each language
that this build supports: phonemizer v1 (nano, pico): es it pt fr; v2 (mini): es it pt fr de da hi ja ko sv.
(Japanese: tests/tts2_tiny/test_tiny_ja_frontend.py.)
"""
import ctypes
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engines" / "lite" / "engine" / "include" / "kitten_tts08.h"
FETCH = ROOT / "engines" / "lite" / "mobile" / "fetch_phonemizer_data.sh"
FRONTEND = ROOT / "engines" / "lite" / "engine" / "src" / "frontend"
LIB = os.environ.get("KITTEN_TTS08_CAPI_LIB")
DATA = os.environ.get("KITTEN_ESPEAK_DATA")
REPO = os.environ.get("KITTEN_TTS2_TINY_REPO")
ES = [
    "Esto parece tener sentido, ya que en la Tierra no se percibe su movimiento, ¿cierto?",
    "Las escenas se proyectan en las pirámides y todas ellas son iluminadas.",
    "Durante esta fase de la historia de Europa, la iglesia católica, que había acumulado riqueza y poder, fue cuestionada.",
    "Contiene 78 recomendaciones, de las cuales la número uno es la adopción de una nueva iniciativa.",
]
IT = [
    "L'incidente è avvenuto in alta montagna e si ritiene sia stato causato da fuoco nemico.",
    "Per la discesa a terra non sarà necessario un visto separato (a partire dal 2009) utilizzando le escursioni di bordo.",
    "Secondo una stima dell'USACE, 6 pollici di pioggia potrebbero rompere gli argini danneggiati in precedenza.",
]
SW = [
    "Tangu mwaka wa 1966 Sandabani zimekuwa mbuga za wanyama pori, na inakadiriwa kuwa sasa kuna chui 400 wa Royal "
    "Bengal na kulungu takribani 30,000 katika eneo hilo.",
    "Hakuna ilani ya tsunami iliyotolewa, kwani mtetemeko haukufikia ukubwa unaohitajika wa 6.5.",
    "Wasafiri wanaonuia kwenda ng'ambo kwa upasuaji lazima wahakikishe kuwa wamebeba stakabadhi halali na/au nakala.",
    "Kiwango cha 802.11n hufanya kazi katika masafa ya gigahezi 2.4 na gigahezi 5.0 (SANParks), saa 11:35 usiku.",
]
PT = [   # Brazilian Portuguese: numbers, ordinals, hyphenated words, an acronym
    "Logo após o início das hostilidades, a Grã-Bretanha começou um bloqueio naval à Alemanha.",
    "Era governado pelos franceses \"Vichy\". Eram franceses que fizeram as pazes com os alemães em 1940 e trabalharam com os invasores em vez de combatê-los.",
    "Várias TVs grandes foram instaladas em diversos lugares de Roma para que o povo pudesse assistir à cerimônia.",
    # a decimal number, quotation marks, a number with a leading zero, "R$", a short all-capital word
    "O ala de 1,98m disse 'obrigado' e ligou para 0800 555 111: pagou R$ 50 pelo SUV da Sainsbury's em 2,5 anos.",
]
FR = [
    "Le Vatican compte environ 800 habitants. C'est le plus petit pays indépendant au monde, et le moins peuplé.",
    "En 1976, trente pour cent du Machu Picchu avait été restauré et cette restauration se poursuit jusqu'à aujourd'hui.",
    "Parmi les sports populaires figurent le football, le basket-ball, le volley-ball, le water-polo, l'escrime, le rugby, "
    "le cyclisme, le hockey sur glace, le roller hockey et la Formule 1.",
    "La semaine dernière, le METI a annoncé qu’Apple l’avait informé de 34 incidents de surchauffe supplémentaires, que la "
    "société a qualifiés de « non graves ».",
]
KO = [   # mini only (v2); FLEURS ko_kr test sentences: English words, a spelled abbreviation, numbers, a percent sign
    "집권당인 서아프리카 인민 기구(SWAPO)도 의회 선거에서 과반수를 유지했습니다.",
    "어제 아침 터키 가지안테프에 있는 경찰서에서 폭발한 차량 폭탄으로 인해 경찰관 2명이 죽고 20명 이상의 다른 사람들이 다쳤습니다.",
    "불행히도, 운전자의 행동이 100% 확실하다고 예측할 수 없기 때문에 교통 흐름을 연구하기란 어렵다.",
    "재입국 충격은 신혼 단계가 없기 때문에 문화 충격보다 빠르게 발생하며, 더 오래 지속하고, 더 극심할 수 있습니다.",
]
FI = [   # mini only (phonemizer v2): secondary stress, a decimal point, doubling across words, "SANParks"
    "Suurin osa valtiossa työskentelevistä puhuu italiaa myös arkikielenään, mutta uskonnollisissa toimituksissa käytetään usein latinaa.",
    "Linja-autoja lähtee koko päivän ajan, etenkin kello 6.30 ja 7.30 välisenä aikana.",
    "Mitkä tehdasvalmisteiset tavarat ovat antiikkia? Niitä voi olla kiinni 10 000 vuotta.",
    "Se on yksi Etelä-Afrikan päänähtävyyksistä ja sitä pidetään Etelä-Afrikan kansallispuistojen (SANParks) lippulaivana.",
]
HI = [
    "स्कीइंग मार्ग को एक हाईकिंग (लंबी पैदल यात्रा) मार्ग जैसा ही सोचें।",
    "इस साल की शुरुआत में 53 साल के कुओमो ने अपना शासनकाल शुरू किया और पिछले महीने समलैंगिक विवाह को वैध बनाने वाले बिल पर हस्ताक्षर किए.",
    "मध्य युग के अंत में पश्चिमी यूरोप ने अपनी शैली विकसित करना शुरू कर दिया. धर्मयुद्ध के परिणामस्वरूप उस समय के सबसे बड़े विकास में से एक लोगों ने कपड़ों को बाँधने के लिए बटन का उपयोग करना शुरू किया",
]
SV = [   # mini only (v2)
    "Sent på söndagen meddelade USA:s president Donald Trump via sin pressekreterare att amerikansk trupp ska lämna Syrien.",
    "Avbokningspolicyerna med anledning av coronaviruset ser olika ut, men i slutet av mars sträcker de flesta sig inte "
    "så långt som till juli 2020, då de olympiska spelen skulle ha börjat.",
    "Om du reser med en laptop eller en surfplatta, se till att lagra en kopia på dess minne eller hårddisk "
    "(tillgängligt utan internet).",
    "Priset steg med 2,5 procent till 40 000 kronor år 1989, och 5% av dem betalade $ 100.",
    # English words inside Swedish ("design", "of the"), "rrs", a sign before a number, "=", capitals
    "Lord President of the Council designade en kostymdesign åt en gitarrspelare i Bălți.",
    "Det var -56 grader i juli 1971, och 3%-räntan = 0 enligt AGES Casting AB och NBA.",
]

DE = [
    "Ziegen wurden anscheinend bereits vor ungefähr 10.000 Jahren im Zāgros-Gebirge im Iran domestiziert.",
    "Zwischen 22:00 und 11:00 Uhr MDT wurde von den Insassen im Hof ein Feuer gelegt.",
    "Sein Ruhm als Epizentrum des Luxus begann ungefähr um 400 n. Chr. und dauerte bis ungefähr 1100 n. Chr. an.",
    "In einigen Gebieten genügt es, Wasser eine Minute lang abzukochen, in anderen müssen es mehrere Minuten sein.",
]

DA = [   # mini only (v2): FLEURS da test sentences: numbers, a hyphenated name, an abbreviation, a loan word
    "Men efter at have mistet kaptajnens gærde scorede Indien kun 36 løb og havde tabt 7 gærder da inningen var slut.",
    "I 2002 blev Goma ødelagt af lava fra Nyiragongo-vulkanen, som begravede de fleste af byens gader i særdeleshed i byens centrum.",
    "Et nyt provisorisk valgråd (CEP) på ni medlemmer blev i går svoret ind af Martelly.",
    "Som i alle sydafrikanske nationalparker er der daglige bevarelses- og adgangsgebyrer til parken.",
    "Dette er en vigtig måde at skelne mellem nogle verber og objektiver",
    "Jo mindre Rossby-tallet er, desto mindre aktiv er stjernen med hensyn til magnetiske vendinger.",
    "I 1987 kostede huset 12500 kroner, og i 2019 havde det en værdi på 3250000 kroner.",
]

V1 = ("es", "it", "pt", "fr", "de")
V2 = ("es", "it", "pt", "fr", "de", "da", "hi", "ko", "sv")
# Enabled, with known differences from the package on some sentences (numbers, amounts, ~1 in 5 Dutch sentences):
# checked for phonemes, not for an exact match
OTHERS = {
    "fi": "Hyvää huomenta, tänään on kaunis päivä.",
    "nl": "Goedemorgen, het is vandaag een mooie dag.",
    "ru": "Доброе утро, сегодня прекрасный день.",
    "sw": "Habari za asubuhi, leo ni siku nzuri.",
    "tr": "Günaydın, bugün güzel bir gün.",
    "hi": HI[0],
}
OTHERS_V1 = ("hi", "ru")
OTHERS_V2 = ("fi", "nl", "ru", "sw", "tr")
SENTENCES = {"da": DA, "de": DE, "es": ES, "it": IT, "pt": PT, "fr": FR, "hi": HI, "ko": KO, "sv": SV}


class SourceTest(unittest.TestCase):
    def test_header_declares_front_end(self):
        h = HEADER.read_text(encoding="utf-8")
        for name in ("kt08_tiny_frontend_load", "kt08_tiny_frontend_free", "kt08_tiny_text_to_phonemes"):
            self.assertIn(name + "(", h)

    def test_fetch_script_pins_data(self):
        s = FETCH.read_text(encoding="utf-8")
        # one pin for every espeak-ng data file: the 1.52.0 release, the version the Python package runs
        self.assertIn("COMMIT=4870adfa25b1a32b4361592f1be8a40337c58d6c", s)
        for f in ("en_rules", "en_list", "es_rules", "es_list", "it_rules", "it_list", "pt_rules", "pt_list", "fr_rules",
                  "fr_list", "sw_rules", "tr_rules", "sv_rules", "fi_rules", "ko_rules", "hi_rules", "phsource/phonemes",
                  "phsource/ph_spanish", "phsource/ph_pt_brazil", "phsource/ph_swahili", "phsource/ph_swedish",
                  "phsource/ph_finnish", "phsource/ph_korean", "phsource/ph_hindi", "da_rules", "da_list", "phsource/ph_danish", "de_rules", "de_list", "de_emoji", "phsource/ph_german",
                  "nl_rules", "nl_list", "phsource/ph_dutch", "phsource/ph_afrikaans", "ru_rules", "ru_list", "ru_listx", "phsource/ph_russian",
                  "phsource/ph_polish"):
            self.assertRegex(s, re.escape(f) + r" [0-9a-f]{64}")

    def test_no_espeak_data_in_repo(self):
        # espeak-ng's data files are GPL-3.0: downloaded by the app, never committed
        names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        bad = [n for n in names if re.search(r"(^|/)([a-z]{2,3}_(rules|list|listx|extra)|ph_[a-z_]+|phonemes)$", n)]
        self.assertEqual(bad, [])

    def test_front_end_has_no_espeak_code(self):
        for p in list(FRONTEND.glob("*.cpp")) + list(FRONTEND.glob("lang/*.[ch]pp")):
            text = p.read_text(encoding="utf-8")
            self.assertIn("Apache License", text)
            self.assertNotIn("espeak_ng.h", text)


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

    def _check(self, version, langs):
        from kitten_inference.tts2_tiny.frontend import Frontend
        vocab = json.load(open(os.path.join(REPO, "vocab.json"), encoding="utf-8"))
        py = Frontend(version, vocab, None)
        fe = self._fe(version)
        try:
            for lang in langs:
                if not os.path.exists(os.path.join(DATA, lang + "_rules")):
                    continue   # that language's data was not fetched
                for text in SENTENCES[lang]:
                    rc, chunks = self._run(fe, text, lang)
                    self.assertEqual(rc, 0, self.lib.kt08_last_error())
                    self.assertEqual(chunks, py(text, lang)[0], (lang, text))
        finally:
            self.lib.kt08_tiny_frontend_free(fe)

    def test_matches_package_v1(self):
        self._check("v1", V1)

    def test_matches_package_v2(self):
        self._check("v2", V2)

    def test_french_many_dashes_do_not_overflow_the_stack(self):
        # " – " (space, en dash, space) ends a French clause; thousands of them must not recurse once each
        if not os.path.exists(os.path.join(DATA, "fr_rules")):
            self.skipTest("fr data not fetched")
        fe = self._fe("v1")
        try:
            rc, chunks = self._run(fe, "a" + " \u2013 a" * 20000, "fr")
            self.assertEqual(rc, 0, self.lib.kt08_last_error())
            self.assertTrue(chunks)
        finally:
            self.lib.kt08_tiny_frontend_free(fe)

    def test_other_languages_give_phonemes(self):
        for version, langs in (("v1", OTHERS_V1), ("v2", OTHERS_V2)):
            fe = self._fe(version)
            try:
                for lang in langs:
                    if not os.path.exists(os.path.join(DATA, lang + "_rules")):
                        continue   # that language's data was not fetched
                    rc, chunks = self._run(fe, OTHERS[lang], lang)
                    self.assertEqual(rc, 0, (version, lang, self.lib.kt08_last_error()))
                    self.assertTrue(chunks and all(chunks), (version, lang))
            finally:
                self.lib.kt08_tiny_frontend_free(fe)

    def test_unsupported_language(self):
        # KT08_ERR_UNSUPPORTED (6): languages the package's phonemizer has no data for (nano / pico: fi ko sv sw tr ja da nl)
        for version, langs in (("v1", ("fi", "ko", "sv", "sw", "tr", "ja", "da", "nl", "zh", "xx")),
                               ("v2", ("zh", "xx"))):
            fe = self._fe(version)
            try:
                for lang in langs:
                    rc, _ = self._run(fe, "Hello.", lang)
                    self.assertEqual(rc, 6, (version, lang))
            finally:
                self.lib.kt08_tiny_frontend_free(fe)

if __name__ == "__main__":
    unittest.main()
