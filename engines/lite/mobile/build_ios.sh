#!/usr/bin/env bash
# Builds the Kitten TTS 0.8 C library for iOS (arm64 iPhone, iOS 15 or newer) and packs it as an XCFramework.
# Needs macOS with Xcode and CMake 3.21 or newer.
#
#   engines/lite/mobile/build_ios.sh [OUT_DIR]
#
# Output: OUT_DIR/KittenTTS08.xcframework (static library + kitten_tts08.h) and OUT_DIR/device/ (the plain static
# library). Link the app with Accelerate (-framework Accelerate) and libc++.
# Environment: IOS_DEPLOYMENT_TARGET (default 15.0), KT08_SIMULATOR=1 to add an arm64 simulator slice.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$PWD/kitten-tts08-ios}
TARGET=${IOS_DEPLOYMENT_TARGET:-15.0}
GEN=$(command -v ninja >/dev/null 2>&1 && echo Ninja || echo "Unix Makefiles")
build() {  # build SYSROOT DIR
  cmake -S "$HERE/../engine" -B "$OUT/build-$2" -G "$GEN" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT="$1" -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$TARGET" -DBUILD_SHARED_LIBS=OFF -DKITTEN_BUILD_C_LIBRARY=ON \
    -DKITTEN_BUILD_PYTHON_MODULE=OFF -DENABLE_CUDA=OFF -DENABLE_NATIVE_ARCH=OFF -DENABLE_OPENMP=OFF \
    -DKITTEN_ACCELERATE=ON -DKITTEN_ACCELERATE_DEFAULT=ON -DCMAKE_INSTALL_PREFIX="$OUT/$2"
  cmake --build "$OUT/build-$2" --parallel
  cmake --install "$OUT/build-$2" --component c_library
}
build iphoneos device
ARGS=(-library "$OUT/device/lib/libkitten_tts08.a" -headers "$OUT/device/include")
if [ "${KT08_SIMULATOR:-0}" = 1 ]; then
  build iphonesimulator simulator
  ARGS+=(-library "$OUT/simulator/lib/libkitten_tts08.a" -headers "$OUT/simulator/include")
fi
rm -rf "$OUT/KittenTTS08.xcframework"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$OUT/KittenTTS08.xcframework"
echo "built $OUT/KittenTTS08.xcframework"
