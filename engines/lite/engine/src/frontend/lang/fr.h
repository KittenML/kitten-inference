// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// French (fr-fr) helpers of the text front end for Kitten TTS 2 mini / nano / pico.
//
// English words inside French text: the French rules hand many names and loan words to English ("Washington",
// "football"). Those words are read by CEPhonemizer (KittenML's Apache-2.0 English rule engine, already part of this
// library) in its British mode, which is what the Python package's reference does for a French voice. This file is
// the small adapter around it: which words go to it, and the spelling of its output (ties, British vowels).
#pragma once

#include <memory>
#include <string>

namespace kt_fe {

class FrEnglishWords {
public:
    // data_dir holds en_rules and en_list (the same folder as the French files). nullptr when they cannot be read.
    static std::unique_ptr<FrEnglishWords> load(const std::string& data_dir);
    virtual ~FrEnglishWords() = default;
    // One written word ("Washington", "d'Ellsworth") -> IPA with stress marks and '^' ties, in the form the generic
    // reader prints. Empty when the caller should use the generic reader (function words, letters outside ASCII).
    virtual std::string word(const std::string& utf8_word) const = 0;
};

}  // namespace kt_fe
