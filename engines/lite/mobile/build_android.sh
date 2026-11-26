#!/usr/bin/env bash
# Builds the Kitten TTS 0.8 C library for Android arm64-v8a with the Android NDK (r26 or newer).
#
#   ANDROID_NDK=$HOME/Android/Sdk/ndk/27.2.12479018 engines/lite/mobile/build_android.sh [OUT_DIR]
#
# Output: OUT_DIR/lib/libkitten_tts08.so (one library for every arm64 phone: the int8 kernels pick the CPU's
# instruction set at run time), OUT_DIR/include/kitten_tts08.h and OUT_DIR/bin/kt08_example.
# Environment: ANDROID_API (default 24), KT08_SHARED=OFF for a static library, KT08_STATIC_EXAMPLE=1 for a fully
# static example program (it also runs on ARM Linux, which the tests use to compare phones with Linux; static
# executables need ANDROID_API=29 or newer).
# KITTEN_QNN_SDK=<Qualcomm AI Runtime SDK folder> also builds the optional Qualcomm NPU path (kt08_npu_*, README.md);
# only the SDK's headers are used: Qualcomm's libraries are loaded at run time and are not part of the output.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$PWD/kitten-tts08-android-arm64}
: "${ANDROID_NDK:?set ANDROID_NDK to the Android NDK folder}"
BUILD=${BUILD_DIR:-$OUT/build}
GEN=$(command -v ninja >/dev/null 2>&1 && echo Ninja || echo "Unix Makefiles")
EXTRA=()
if [ "${KT08_STATIC_EXAMPLE:-0}" = 1 ]; then EXTRA+=(-DKITTEN_C_EXAMPLE_STATIC=ON); fi
cmake -S "$HERE/../engine" -B "$BUILD" -G "$GEN" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM="android-${ANDROID_API:-24}" -DANDROID_STL=c++_static \
  -DBUILD_SHARED_LIBS="${KT08_SHARED:-ON}" -DKITTEN_BUILD_C_LIBRARY=ON -DKITTEN_BUILD_C_EXAMPLE=ON \
  -DKITTEN_BUILD_PYTHON_MODULE=OFF -DENABLE_CUDA=OFF -DENABLE_NATIVE_ARCH=OFF -DENABLE_ARM_ISA_DISPATCH=ON \
  -DCMAKE_INSTALL_PREFIX="$OUT" -DKITTEN_QNN_SDK="${KITTEN_QNN_SDK:-}" "${EXTRA[@]}"
cmake --build "$BUILD" --parallel
cmake --install "$BUILD" --component c_library --strip
# The NDK's toolchain file adds -g; --strip covers the .so and the program but not a static archive (34 MB with
# debug info). Strip the archive's debug sections; its code is unchanged.
if [ -f "$OUT/lib/libkitten_tts08.a" ]; then
  "$(ls -d "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin | head -1)/llvm-strip" --strip-debug "$OUT/lib/libkitten_tts08.a"
fi
echo "built $OUT"
