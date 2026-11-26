// ops/norm.cu — CUDA upsample and adaptive norm ops
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_upsample_nearest1d_fp32(const float* in, float* out, int T, int C, int sf, StreamHandle /*stream*/) {
    dim3 block(256);
    dim3 grid((T*sf + block.x - 1) / block.x, 1, C);
    k_upsample_nearest1d<<<grid, block>>>(in, out, T, C, sf);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_upsample_bilinear2d_fp32(const float* in, float* out,
    int C, int iH, int iW, int oH, int oW, StreamHandle /*stream*/)
{
    dim3 block(16, 16);
    dim3 grid((oW + block.x - 1) / block.x, (oH + block.y - 1) / block.y, C);
    k_upsample_bilinear2d<<<grid, block>>>(in, out, C, iH, iW, oH, oW);
    CUDA_CHECK(cudaGetLastError());
}

static void ada_in1d_impl(const float* feat, float* out, const float* gamma, const float* beta, int T, int C, float eps,
                          const float* norm_weight, const float* norm_bias, const float* sa, const float* si);

void cuda_ada_in1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle /*stream*/,
    const float* norm_weight, const float* norm_bias)
{
    ada_in1d_impl(feat, out, gamma, beta, T, C, eps, norm_weight, norm_bias, nullptr, nullptr);
}

// AdaIN followed by Snake1D (alpha, inv_alpha: device) in the same output pass: one [T, C] write and read fewer
void cuda_ada_in1d_snake_fp32(const float* feat, float* out, const float* gamma, const float* beta, int T, int C,
                              float eps, const float* norm_weight, const float* norm_bias, const float* alpha,
                              const float* inv_alpha, StreamHandle /*stream*/)
{
    ada_in1d_impl(feat, out, gamma, beta, T, C, eps, norm_weight, norm_bias, alpha, inv_alpha);
}

static void ada_in1d_impl(const float* feat, float* out, const float* gamma, const float* beta, int T, int C, float eps,
                          const float* norm_weight, const float* norm_bias, const float* sa, const float* si)
{
    // The InstanceNorm weights are constant: uploaded once (no copy and wait per call).
    float* d_nw = norm_weight ? g_const_cache.get(norm_weight, C) : nullptr;
    float* d_nb = norm_bias   ? g_const_cache.get(norm_bias,   C) : nullptr;
    const int tiles = (C + 31) / 32;
    if (tiles >= 64 || T < 1024) {   // enough blocks already (or a short axis): one block per 32 channels
        k_ada_in1d<<<tiles, dim3(32, 32)>>>(feat, out, gamma, beta, T, C, eps, d_nw, d_nb, sa, si);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Few channels, long time axis: split the time axis so that about 256 blocks share the sums.
    const int S = std::min((T + 255) / 256, std::max(1, 256 / tiles));
    const int rows = (T + S - 1) / S;
    auto mark = g_scratch.save();
    float* part = (float*)g_scratch.alloc((size_t)S * C * sizeof(float));
    float* stat = (float*)g_scratch.alloc((size_t)2 * C * sizeof(float));   // mean, 1 / std
    k_adain_psum<<<dim3(tiles, S), dim3(32, 8)>>>(feat, T, C, rows, nullptr, part);
    k_adain_reduce<<<(C + 127) / 128, 128>>>(part, S, C, T, eps, 0, stat);
    k_adain_psum<<<dim3(tiles, S), dim3(32, 8)>>>(feat, T, C, rows, stat, part);
    k_adain_reduce<<<(C + 127) / 128, 128>>>(part, S, C, T, eps, 1, stat + C);
    const size_t n = (size_t)T * C;
    k_adain_apply<<<(unsigned)((n + 255) / 256), 256>>>(feat, out, stat, stat + C, gamma, beta, d_nw, d_nb, T, C, sa, si);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}

void cuda_ada_layer_norm1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle /*stream*/)
{
    // Use parallel reduction: up to 256 threads per row
    int bsz = 32;                       // the tree reduction halves the block: it must be a power of two
    while (bsz < C && bsz < 256) bsz <<= 1;
    k_ada_layernorm1d<<<T, bsz, bsz * sizeof(float)>>>(feat, out, gamma, beta, T, C, eps);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_snake1d_fp32(const float* in, float* out,
    const float* alpha, const float* inv_alpha,
    int T, int C, StreamHandle /*stream*/)
{
    int n = T * C;
    k_snake1d<<<(n + 255) / 256, 256>>>(in, out, alpha, inv_alpha, T, C);
    CUDA_CHECK(cudaGetLastError());
}
