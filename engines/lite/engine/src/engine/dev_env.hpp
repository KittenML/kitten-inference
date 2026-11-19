// src/engine/dev_env.hpp — engine development switches read from the environment.
//
// The engine has a few A/B toggles (kernel variants, in-place plans, LSTM split, Winograd block size), statistics and
// profiling outputs that are useful while developing or measuring a change. They are compiled in only when the engine
// is configured with -DKITTEN_ENGINE_DEV_ENV=ON; every other build (the Python wheels, the C library, the phone
// builds) ignores them and always runs the default path, so a stray environment variable cannot change the audio or
// the speed of a release build.
//
// The settings a user may need stay readable in every build (see the engine README, "Environment variables").
#pragma once

#include <cstdlib>

inline const char* kt_dev_getenv(const char* name) {
#if defined(KITTEN_ENGINE_DEV_ENV)
    return std::getenv(name);
#else
    (void)name;
    return nullptr;
#endif
}
