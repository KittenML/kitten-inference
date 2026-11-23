// Copyright 2026 KittenML. Licensed under the Apache License, Version 2.0.
#include "frontend/lang/fi.h"

namespace kt_fe {
namespace lang_fi {

void secondary_stress(const std::vector<bool>& heavy, std::vector<int>& stress) {
    const int n = (int)stress.size();
    auto weak = [&](int k) { return k < 0 || k >= n || stress[k] <= 1; };   // -1 (undecided) counts as weak
    bool primary = false;
    for (int s : stress) if (s >= 4) primary = true;
    if (n > 0 && stress[0] < 0 && primary) stress[0] = 2;   // the primary is later (numbers): the first syllable
    for (int k = 1; k + 1 < n; k++) {                                        // never the last syllable
        if (stress[k] >= 0) continue;
        if (!weak(k - 1) || !weak(k + 1)) continue;
        if (k > 1 && !heavy[k]) {
            // a light syllable gives way to a heavy one later in the word (the last syllable only when it is next)
            bool heavier_ahead = heavy[k + 1];
            for (int j = k + 1; j + 1 < n && !heavier_ahead; j++) heavier_ahead = heavy[j];
            if (heavier_ahead) continue;
        }
        stress[k] = 2;
    }
}

}  // namespace lang_fi
}  // namespace kt_fe
