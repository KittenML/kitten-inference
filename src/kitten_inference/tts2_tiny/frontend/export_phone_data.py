# Part of kitten-inference (Apache-2.0).
"""Writes the data files the phone library's English text front end (kt08_tiny_frontend_*) reads at run time.

    python -m kitten_inference.tts2_tiny.frontend.export_phone_data OUT_DIR

OUT_DIR gets:
  us_gold.json, us_silver.json   misaki 0.9.4's American English lexicons (Apache-2.0), copied from this package
  spacy_en_exceptions.json       the tokenizer special cases of the vendored spaCy model en_core_web_sm 3.8.0 (MIT):
                                 {"don't": ["do", "n't"], ...}, the only part of spaCy the phone front end uses
Put them next to the model repository's g2p/ files (cmu_misaki_us.json, g2p_bart_en_us/), or pass their folder as the
front end's data folder. Needs spaCy (installed with the package on CPython 3.10+).
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def export(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in ("us_gold.json", "us_silver.json"):
        shutil.copyfile(os.path.join(HERE, "misaki", "data", name), os.path.join(out_dir, name))
    import spacy
    nlp = spacy.load(os.path.join(HERE, "en_core_web_sm"), enable=[])
    orth = spacy.attrs.ORTH
    rules = {k: [t[orth] for t in v] for k, v in sorted(nlp.tokenizer.rules.items())}
    with open(os.path.join(out_dir, "spacy_en_exceptions.json"), "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=0, sort_keys=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    export(sys.argv[1])
