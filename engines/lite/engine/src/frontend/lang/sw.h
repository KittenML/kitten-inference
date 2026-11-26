// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
//
// Swahili number words for the espeak-ng data reader (espeak_lang.cpp). The words themselves come from the
// downloaded word list (sw_list: _1 .. _9, _1X .. _9X, _1C .. _9C, _1M1 .. _9M1, _0M1, _1M2 .. _9M2, _0M3, _0and);
// this file only decides how they are put together, as measured against espeak-ng's own output.
#pragma once

#include <functional>
#include <string>

namespace kt_fe {

// digits: ASCII '0'..'9', no leading zero, at most 12 digits. entry(key) returns the list entry's phonemes, has(key)
// whether the entry exists. The result joins the words with "||" (a new word), "\x02" (a new stress domain without a
// space) and "\x0b" (a space inside one stress domain: "na" and the word after it are stressed together).
// Returns "" for anything else (the caller then reads the digits one by one).
std::string sw_number(const std::string& digits, const std::function<std::string(const std::string&)>& entry,
                      const std::function<bool(const std::string&)>& has);

}  // namespace kt_fe
