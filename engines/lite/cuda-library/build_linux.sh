#!/bin/bash
# Build the NVIDIA GPU library of Kitten TTS 2 mini, nano and pico (libkt08_cuda.so: the engine with its CUDA backend,
# the CUDA 12.8 runtime linked in, no cuBLAS) in the manylinux_2_28 container, the same image and CUDA as the Kitten TTS
# 2 engine (engines/tts2/packaging/build_wheel_linux.sh), whose CUDA payload wheel ships it in
# kitten_inference/_tts2_cuda/lib/:
#   engines/lite/cuda-library/build_linux.sh [OUT_DIR]     (inside the container)
#   engines/lite/cuda-library/build_linux.sh --docker [OUT_DIR]   (runs itself in the container)
# Environment: JOBS (default: the CPUs available), KITTEN_CUDA_ARCHS (default below: Pascal to Blackwell, plus PTX for
# newer GPUs). Prints the library's size, its dynamic dependencies (only the C/C++ system libraries the manylinux_2_28
# policy allows; the driver's libcuda is opened at run time) and its exported symbols (kt08g_* only).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:2026.10.24-1}
if [ "${1:-}" = --docker ]; then
  shift
  OUT=$(mkdir -p "${1:-$ROOT/engines/lite/dist}" && cd "${1:-$ROOT/engines/lite/dist}" && pwd)
  exec docker run --rm -e JOBS="${JOBS:-$(nproc)}" -e KITTEN_CUDA_ARCHS="${KITTEN_CUDA_ARCHS:-}" \
    -v "$ROOT":/root/src -v "$OUT":/out -w /root/src "$IMAGE" bash engines/lite/cuda-library/build_linux.sh /out
fi
OUT=${1:-$ROOT/engines/lite/dist}
JOBS=${JOBS:-$(nproc)}
ARCHS=${KITTEN_CUDA_ARCHS:-"61-real;75-real;80-real;86-real;89-real;90-real;100-real;120"}
sh "$ROOT/engines/tts2/packaging/install_cuda_linux.sh"
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDACXX=/usr/local/cuda-12.8/bin/nvcc
BUILD=${KT08_BUILD_DIR:-/tmp/kt08-cuda-library}   # outside the source tree
rm -rf "$BUILD" "$BUILD-stage"
cmake -S "$ROOT/engines/lite/engine" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DKITTEN_BUILD_CUDA_LIBRARY=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DKITTEN_BUILD_PYTHON_MODULE=OFF -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_AVX2=ON -DENABLE_ARM_DOTPROD=OFF \
  -DENABLE_OPENMP=OFF -DKITTEN_BUILD_EXTRAS=OFF
cmake --build "$BUILD" --target kt08_cuda --parallel "$JOBS"
cmake --install "$BUILD" --component cuda_library --prefix "$BUILD-stage"
LIB="$BUILD-stage/lib/libkt08_cuda.so"
strip --strip-unneeded "$LIB"
mkdir -p "$OUT"
cp "$LIB" "$OUT/libkt08_cuda.so"
echo "--- dynamic dependencies"; readelf -d "$OUT/libkt08_cuda.so" | grep NEEDED
bad=$(readelf -d "$OUT/libkt08_cuda.so" | grep NEEDED | grep -Ev '\[(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s)\.so|\[ld-linux-x86-64\.so' || true)
[ -z "$bad" ] || { echo "unexpected dependency: $bad"; exit 1; }
echo "--- exported functions"; nm -D --defined-only "$OUT/libkt08_cuda.so" | awk '$2 == "T" {print $3}' | sort | tee "$OUT/exports.txt"
grep -qv '^kt08g_' "$OUT/exports.txt" && { echo "symbols other than kt08g_* are exported"; exit 1; }
echo "--- GPU code"; cuobjdump --list-elf "$OUT/libkt08_cuda.so" | sed 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/' | sort | uniq -c
cuobjdump --list-ptx "$OUT/libkt08_cuda.so" | sed 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/' | sort | uniq -c | sed 's/^/PTX /'
ls -l "$OUT/libkt08_cuda.so"
