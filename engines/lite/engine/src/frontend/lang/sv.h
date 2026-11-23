// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Swedish pieces of the espeak-data reader (espeak_lang.cpp): how a run of digits is read, as the Python package's
// espeak-ng 1.52.0 reads it (measured against it word by word; no espeak-ng code).
#pragma once

#include <functional>
#include <string>

namespace kt_fe {
namespace sv {

// digits: ASCII digits only (separators already removed). ph(key): the word list's phonemes for "_7", "_2X", "_0C",
// "_0M1", "_0M2", "_1M2", ... Returns phoneme mnemonics; "||" separates words.
//   25 -> ,tjugo'fem (the tens lose their primary)   125 -> etthundra tjugofem   1000 -> ett tusen
//   1940 -> nittonhundra fyrtio (19xx only)   2020 -> två tusen tjugo   2000000 -> två miljoner   07 -> noll sju
// year_group: the digits were written as a four-digit group, a space, then groups of three ("1971 621", "2000 300"):
//   the four digits are read first, as a number of thousands or millions: nittonhundra sjuttioett tusen sexhundra
//   tjugoett; två miljoner tusen trehundra.
// grouped: written in groups ("1 909"): never read like a year (ett tusen niohundra nio).
std::string number(const std::string& digits, const std::function<std::string(const std::string&)>& ph, bool year_group = false,
                   bool grouped = false);

}  // namespace sv
}  // namespace kt_fe
