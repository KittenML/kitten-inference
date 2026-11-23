"""The number formats (precisions) of the decoder's weights that each backend offers, and why others are not.

In addition to the locally converted formats, small-enhanced supplies pre-quantized int4 and int8 weights. This module imports nothing heavy, so the checks run before PyTorch
or the CPU library is loaded.
"""
from __future__ import annotations

BACKENDS = ("cuda", "cpu", "metal")
# The first of each is the backend's default.
#   NVIDIA GPU: bf16 = the published weights; fp16 = 16-bit floats; int8 = 8-bit weights, 16-bit activations.
#   CPU and Mac GPU: q8_0 = 8-bit weights, one scale per 32 weights; f16 = 16-bit floats; bf16 = the published weights.
GPU = ("bf16", "fp16", "int8", "int4")
CPU = ("q8_0", "f16", "bf16", "int8", "int4")
ENCODER_CPU = ("f16", "q8_0", "bf16")
NOT_OFFERED_GPU = {
    "fp8": "8-bit float weights and activations raised the error rate by more than 0.5 points in several "
           "languages and were not faster than 'int8' for one stream on any GPU tested",
}
NOT_OFFERED_CPU = {
    "q6_k": "6-bit weights were less accurate than 'q8_0' (worse by more than 0.5 points in 5 of 30 languages "
            "with small-enhanced) and no speed gain was measured",
    "q5_k_m": "5-bit weights were less accurate than 'q8_0' (tiny: worse by more than 0.5 points in Russian) and "
              "no speed gain was measured",
    "q4_k_m": "4-bit weights raised the error rate by more than 0.5 points in 13 of 30 languages with "
              "small-enhanced; a decode step ran at 0.94 to 1.31 times the speed of 'q8_0' (Xeon Gold 6430, Arm GB10)",
    "q4_0": "4-bit weights raised the error rate by more than 0.5 points in 6 of the 7 languages tiny supports; "
            "a decode step was 1.14 times faster than 'q8_0' (Xeon Gold 6430)",
    "f32": "32-bit floats were not tested for accuracy; 'bf16' holds the published weights exactly in half the "
           "memory",
}
_WHERE = {"cuda": "an NVIDIA GPU (backend='cuda')", "cpu": "the CPU (backend='cpu')",
          "metal": "the Mac GPU (backend='metal')"}


def offered(backend: str) -> tuple:
    """The precisions `backend` offers; the first is its default."""
    return GPU if backend == "cuda" else CPU


def check_backend(backend):
    """None (automatic) or one of BACKENDS in lower case; ValueError otherwise."""
    if backend is None:
        return None
    key = backend.strip().lower() if isinstance(backend, str) else None
    if key not in BACKENDS:
        raise ValueError("backend must be 'cuda' (NVIDIA GPU), 'cpu' or 'metal' (Mac GPU), or left out to pick "
                         f"automatically, not {backend!r}")
    return key


def check(backend: str, precision) -> str:
    """`precision`, or the default of `backend` for None. ValueError, naming what to use, for a precision this
    backend does not offer."""
    choices = offered(backend)
    if precision is None:
        return choices[0]
    if isinstance(precision, str) and precision in choices:
        return precision
    use = f"Use one of {', '.join(choices)}"
    reason = (NOT_OFFERED_GPU if backend == "cuda" else NOT_OFFERED_CPU).get(precision) \
        if isinstance(precision, str) else None
    if reason:
        raise ValueError(f"precision {precision!r} is not offered: {reason}. {use}.")
    if isinstance(precision, str) and precision in (CPU if backend == "cuda" else GPU):
        other = "the CPU and the Mac GPU" if backend == "cuda" else "NVIDIA GPUs (backend='cuda')"
        raise ValueError(f"precision {precision!r} is for {other}; this model is loading on {_WHERE[backend]}. "
                         f"{use}, or leave precision out for the default ({choices[0]}).")
    raise ValueError(f"precision must be one of {', '.join(choices)} on {_WHERE[backend]}, not {precision!r}")
