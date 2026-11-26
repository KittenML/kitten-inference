// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Finnish-specific parts of the espeak-data reader (espeak_lang.cpp). Original code: the behaviour is described from
// espeak-ng's documented voice options (espeak-ng-data/lang/urj/fi: stressRule 0, stressOpt 1 2 4 6) and checked
// against the Python package's output; no espeak-ng source code.
#pragma once

#include <vector>

namespace kt_fe {
namespace lang_fi {

// Secondary stress of one Finnish word. `stress` has one entry per syllable (vowel): -1 = not decided yet,
// 0..1 = unstressed, 2..3 = secondary, 4 = primary. `heavy` says whether each syllable is heavy (long vowel or
// diphthong, or closed by a consonant). After the first syllable, a syllable whose neighbours are both unstressed
// gets secondary stress (so stress alternates), except:
//   - the last syllable never gets it;
//   - a light syllable is passed over when a heavy syllable follows later in the word, or when the next syllable is
//     the last one and is heavy (the stress then falls on the heavier syllable, or on none).
// Heavy: a long vowel or diphthong, or two or more consonants before the next vowel; the last syllable is heavy
// when it is long or ends in a consonant.
// Undecided syllables stay -1 (the caller makes them unstressed).
void secondary_stress(const std::vector<bool>& heavy, std::vector<int>& stress);

}  // namespace lang_fi
}  // namespace kt_fe
