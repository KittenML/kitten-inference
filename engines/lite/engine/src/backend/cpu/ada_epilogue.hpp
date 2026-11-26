#pragma once
// elementwise activation fused into the AdaIN1d final pass (see ada_in1d_epi_fp32 in ops/norm.cpp).
// kind 0 = none, 1 = Snake1D (alpha / inv_alpha [C], may be null = 1), 2 = LeakyReLU (slope).
struct AdaEpilogue {
    int kind = 0;
    const float* alpha = nullptr;
    const float* inv_alpha = nullptr;
    float slope = 0.f;
};
