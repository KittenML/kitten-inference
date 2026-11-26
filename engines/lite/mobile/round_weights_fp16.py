"""Writes a copy of a Kitten TTS 0.8 weights file (W1X1) whose float32 values are rounded to the nearest 16-bit float.

    python engines/lite/mobile/round_weights_fp16.py kitten_fp32_80m.bin kitten_fp32_80m_fp16w.bin

LOSSY: the rounded weights change the audio slightly. With this file and the C library's fp16_weights option
(kt08_options.fp16_weights = 1), the engine keeps the weights in 16 bits, which saves about 40% of the weight memory.
The file stays float32 on disk; only the values change. Needs numpy.
"""
import struct
import sys

import numpy as np


def round_file(src_path, dst_path):
    src = open(src_path, "rb").read()
    if src[:4] != b"W1X1":
        raise SystemExit(f"{src_path} is not a W1X1 weights file")
    _version, count = struct.unpack_from("<II", src, 4)
    out = bytearray(src)
    rounded = total = 0
    for i in range(count):
        entry = src[12 + 80 * i: 12 + 80 * (i + 1)]
        offset, nbytes = struct.unpack_from("<QI", entry, 64)
        dtype, _flags, ndim = struct.unpack_from("<BBH", src, offset)
        shape = struct.unpack_from("<%di" % ndim, src, offset + 4)
        data = offset + 4 + 4 * ndim
        values = int(np.prod(shape)) if ndim else 1
        if dtype != 1 or values == 0 or (nbytes - 4 - 4 * ndim) // values != 4:
            continue  # not a float32 tensor
        a = np.frombuffer(src, np.float32, values, data)
        b = a.astype(np.float16).astype(np.float32)
        rounded += int((a != b).sum())
        total += values
        out[data:data + 4 * values] = b.tobytes()
    with open(dst_path, "wb") as f:
        f.write(out)
    print(f"{dst_path}: {rounded} of {total} float32 values rounded to 16 bits")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    round_file(sys.argv[1], sys.argv[2])
