#!/bin/bash
# Build the Linux wheel in the manylinux_2_28 container with Docker (what the CI job runs):
#   engines/tts2/packaging/build_linux_docker.sh [OUT_DIR]   (default engines/tts2/dist)
# The source is mounted at /root/src/port, the path of the build that packaging/sass-reference.txt was made from: nvcc
# puts the source file's path into the names of GPU functions in anonymous namespaces, so the same path gives the
# same names.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$SRC/dist}" && cd "${1:-$SRC/dist}" && pwd)
IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:2026.10.24-1}
# KT08_CUDA_LIB: a built libkt08_cuda.so (engines/lite/cuda-library/build_linux.sh) to add to the wheel.
lib=()
if [ -n "${KT08_CUDA_LIB:-}" ]; then
  lib=(-v "$(cd "$(dirname "$KT08_CUDA_LIB")" && pwd)/$(basename "$KT08_CUDA_LIB")":/kt08/libkt08_cuda.so:ro -e KT08_CUDA_LIB=/kt08/libkt08_cuda.so)
fi
docker run --rm -e JOBS="${JOBS:-$(nproc)}" -e WHEEL_VERSION="${WHEEL_VERSION:-}" -e KT2N_SASS_CHECK="${KT2N_SASS_CHECK:-1}" \
  "${lib[@]}" -v "$SRC":/root/src/port -v "$OUT":/out -w /root/src/port "$IMAGE" \
  bash packaging/build_wheel_linux.sh /out
