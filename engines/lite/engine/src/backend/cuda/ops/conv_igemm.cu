// ops/conv_igemm.cu — strict fp32 1-D convolution as an implicit GEMM: each block reads its input windows in place
// from the [T][C_in] activation (no im2col copy, no cuBLAS), zero padding handled by the loads.
//   out[t][co] = bias[co] + sum_{k < K, c < C_in} w[co][k][c] * x[t + k*d - pad][c]      (x outside [0, T) is 0)
// Ported from the kitten-tts-2-cpp ggml engine's igemm_conv_kernel (128 x 128 tile, 8 x 8 per thread, k-slices of 16,
// the next slice loaded into registers while the current one is multiplied from shared memory), generalised to
// smaller tiles for layers whose 128 x 128 grid is less than one wave, and with the zero padding moved into the loads.
// f32 fused multiply-adds, no TF32: equal precision with strict fp32 PyTorch.
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

namespace {

constexpr int IG_THREADS = 256;

template <int BT, int BC, int BK, bool GEN>
__global__ void __launch_bounds__(IG_THREADS, 2)
k_conv1d_igemm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
               const float* __restrict__ resid, int T, int Cin, int K, int d, int pl, int stride, int Cout, int Tout,
               float* __restrict__ out, int splits)
{
    // splits > 1 (split-K): block z of gridDim.z reduces slices [z*nk/splits, (z+1)*nk/splits) and writes its partial
    // sums to out + z * Tout * Cout (bias and resid are null then; k_igemm_reduce adds the partials in order).
    // GEN = false: Cin % BK == 0, Cout % 4 == 0, float4 loads and stores. GEN = true: any Cin / Cout, scalar loads
    // with bounds (a slice may straddle two taps), the reduction length padded to a multiple of BK with zeros.
    constexpr int PAD = 4;
    constexpr int TT = BT / 16, TC = BC / 16;          // outputs per thread: TT time steps x TC channels
    constexpr int GT = TT / 4, GC = TC / 4;            // float4 groups per thread
    constexpr int TPR_A = IG_THREADS / BC, TPR_B = IG_THREADS / BT;   // threads per loaded row
    constexpr int NLA = BK / (4 * TPR_A), NLB = BK / (4 * TPR_B);    // float4 loads per thread per slice
    static_assert(NLA >= 1 && NLB >= 1, "tile too large for the slice");
    __shared__ __align__(16) float As[2][BK][BC + PAD];   // [k][co]
    __shared__ __align__(16) float Bs[2][BK][BT + PAD];   // [k][t]
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.x * BT, co0 = blockIdx.y * BC;
    const int KC = Cin * K, nk_all = (KC + BK - 1) / BK;
    const int it0 = (int)((long long)blockIdx.z * nk_all / splits), it1 = (int)((long long)(blockIdx.z + 1) * nk_all / splits);
    out += (size_t)blockIdx.z * Tout * Cout;
    const int la = tid / TPR_A, lha = (tid % TPR_A) * 4;
    const int lb = tid / TPR_B, lhb = (tid % TPR_B) * 4;
    const bool a_ok = co0 + la < Cout;
    const float* wrow = w + (size_t)(co0 + la) * KC + lha;
    const int trow = (t0 + lb) * stride - pl;          // input row of tap 0 for this thread's loaded row
    const bool b_live = t0 + lb < Tout;
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);

    float acc[TT][TC];
#pragma unroll
    for (int i = 0; i < TT; i++)
#pragma unroll
        for (int j = 0; j < TC; j++) acc[i][j] = 0.0f;

    float4 ra[NLA], rb[NLB];
    int tap = (it0 * BK) / Cin, c0 = (it0 * BK) - tap * Cin;   // GEN = false: the current slice starts at tap, c0
    auto load = [&](int it) {
        if constexpr (!GEN) {
            const int r = trow + tap * d;
            const bool bok = b_live && r >= 0 && r < T;
            const float* wn = wrow + (size_t)it * BK;
            const float* pn = x + (size_t)r * Cin + c0 + lhb;
#pragma unroll
            for (int q = 0; q < NLA; q++) ra[q] = a_ok ? __ldg((const float4*)(wn + q * 4 * TPR_A)) : z4;
#pragma unroll
            for (int q = 0; q < NLB; q++) rb[q] = bok ? __ldg((const float4*)(pn + q * 4 * TPR_B)) : z4;
        } else {
            const int k0 = it * BK;
#pragma unroll
            for (int q = 0; q < NLA; q++) {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kk = k0 + q * 4 * TPR_A + lha + e;
                    v[e] = (a_ok && kk < KC) ? __ldg(w + (size_t)(co0 + la) * KC + kk) : 0.f;
                }
                ra[q] = make_float4(v[0], v[1], v[2], v[3]);
            }
#pragma unroll
            for (int q = 0; q < NLB; q++) {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kk = k0 + q * 4 * TPR_B + lhb + e;
                    float y = 0.f;
                    if (b_live && kk < KC) {
                        const int tp = kk / Cin, c = kk - tp * Cin;
                        const int r = trow + tp * d;
                        if (r >= 0 && r < T) y = __ldg(x + (size_t)r * Cin + c);
                    }
                    v[e] = y;
                }
                rb[q] = make_float4(v[0], v[1], v[2], v[3]);
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int q = 0; q < NLA; q++) {
            const int kk = q * 4 * TPR_A + lha;
            As[buf][kk + 0][la] = ra[q].x; As[buf][kk + 1][la] = ra[q].y; As[buf][kk + 2][la] = ra[q].z; As[buf][kk + 3][la] = ra[q].w;
        }
#pragma unroll
        for (int q = 0; q < NLB; q++) {
            const int kk = q * 4 * TPR_B + lhb;
            Bs[buf][kk + 0][lb] = rb[q].x; Bs[buf][kk + 1][lb] = rb[q].y; Bs[buf][kk + 2][lb] = rb[q].z; Bs[buf][kk + 3][lb] = rb[q].w;
        }
    };
    load(it0);
    store(0);
    __syncthreads();

    for (int it = it0; it < it1; it++) {
        const int buf = (it - it0) & 1;
        const bool more = it + 1 < it1;
        if (more) {
            if constexpr (!GEN) { c0 += BK; if (c0 == Cin) { c0 = 0; tap++; } }
            load(it + 1);
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TC], b[TT];
#pragma unroll
            for (int g = 0; g < GC; g++) {
                const float4 v = *(const float4*)&As[buf][kk][g * (BC / GC) + tx * 4];
                a[g * 4 + 0] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < GT; g++) {
                const float4 v = *(const float4*)&Bs[buf][kk][g * (BT / GT) + ty * 4];
                b[g * 4 + 0] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TT; i++)
#pragma unroll
                for (int j = 0; j < TC; j++) acc[i][j] = fmaf(a[j], b[i], acc[i][j]);
        }
        if (more) store(buf ^ 1);
        __syncthreads();
    }

    // acc[i][j]: time t0 + (i / 4) * (BT / GT) + ty * 4 + i % 4, channel co0 + (j / 4) * (BC / GC) + tx * 4 + j % 4
#pragma unroll
    for (int i = 0; i < TT; i++) {
        const int t = t0 + (i / 4) * (BT / GT) + ty * 4 + (i % 4);
        if (t >= Tout) continue;
#pragma unroll
        for (int g = 0; g < GC; g++) {
            const int co = co0 + g * (BC / GC) + tx * 4;
            if (co >= Cout) continue;
            if constexpr (!GEN) {
                float4 v = make_float4(acc[i][g * 4], acc[i][g * 4 + 1], acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
                if (bias) {
                    const float4 bb = __ldg((const float4*)(bias + co));
                    v.x = __fadd_rn(v.x, bb.x); v.y = __fadd_rn(v.y, bb.y); v.z = __fadd_rn(v.z, bb.z); v.w = __fadd_rn(v.w, bb.w);
                }
                if (resid) {   // (conv + bias) + resid: the same two roundings as the separate add node
                    const float4 rr = __ldg((const float4*)(resid + (size_t)t * Cout + co));
                    v.x = __fadd_rn(v.x, rr.x); v.y = __fadd_rn(v.y, rr.y); v.z = __fadd_rn(v.z, rr.z); v.w = __fadd_rn(v.w, rr.w);
                }
                *(float4*)(out + (size_t)t * Cout + co) = v;
            } else {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (co + e >= Cout) break;
                    float v = acc[i][g * 4 + e];
                    if (bias) v = __fadd_rn(v, __ldg(bias + co + e));
                    if (resid) v = __fadd_rn(v, __ldg(resid + (size_t)t * Cout + co + e));
                    out[(size_t)t * Cout + co + e] = v;
                }
            }
        }
    }
}

// out[t][co] = ((sum over z of part[z][t][co], z in order) + bias[co]) + resid[t][co]
__global__ void k_igemm_reduce(const float* __restrict__ part, int splits, const float* __restrict__ bias,
                               const float* __restrict__ resid, float* __restrict__ out, int Tout, int Cout)
{
    const size_t n = (size_t)Tout * Cout;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = part[i];
        for (int z = 1; z < splits; ++z) v = __fadd_rn(v, part[(size_t)z * n + i]);
        if (bias) v = __fadd_rn(v, bias[i % Cout]);
        if (resid) v = __fadd_rn(v, resid[i]);
        out[i] = v;
    }
}

int sm_count() {
    static int n = 0;
    if (!n) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 1;
    }
    return n;
}

} // namespace

bool cuda_conv1d_igemm_supported(int C_in, int C_out, int stride)
{
    static const bool off = [] { const char* e = getenv("KITTEN_CUDA_IGEMM"); return e && e[0] == '0'; }();
    return !off && stride >= 1 && C_in >= 1 && C_out >= 1;
}

template <bool GEN>
static void launch_igemm(int best, dim3 grid, const float* x, const float* w, const float* bias, const float* resid,
                         float* out, int T, int C_in, int C_out, int K, int pad, int dilation, int stride, int T_out,
                         int splits)
{
    if (best == 0)
        k_conv1d_igemm<128, 128, 16, GEN><<<grid, IG_THREADS>>>(x, w, bias, resid, T, C_in, K, dilation, pad, stride, C_out, T_out, out, splits);
    else if (best == 1)
        k_conv1d_igemm<128, 64, 16, GEN><<<grid, IG_THREADS>>>(x, w, bias, resid, T, C_in, K, dilation, pad, stride, C_out, T_out, out, splits);
    else
        k_conv1d_igemm<64, 64, 16, GEN><<<grid, IG_THREADS>>>(x, w, bias, resid, T, C_in, K, dilation, pad, stride, C_out, T_out, out, splits);
}

// x [T][C_in], w [C_out][K][C_in] (device), bias [C_out] or null (device), resid [T_out][C_out] or null (device)
void cuda_conv1d_igemm_fp32(const float* x, const float* w, const float* bias, const float* resid, float* out,
                            int T, int C_in, int C_out, int K, int pad, int dilation, int stride, int T_out)
{
    if (T_out <= 0) return;
    // Pick the tile whose grid fills the GPU best (2 blocks per SM): the 128 x 128 tile does the most work per byte
    // loaded, the smaller ones avoid a mostly idle last wave on short or narrow layers.
    const int slots = 2 * sm_count();
    struct Cand { int bt, bc; double eff; };
    // Per-tile efficiency weights, measured on the RTX 4090 (forced-tile runs, results/optimize/experiments.jsonl t1):
    // a 128 x 64 tile does at least as much work per SM-second as 128 x 128 (they were weighted 0.85 and 1.0).
    const Cand cands[3] = {{128, 128, 0.95}, {128, 64, 1.00}, {64, 64, 0.70}};
    int best = 0; double best_score = -1;
    for (int i = 0; i < 3; ++i) {
        const long long blocks = (long long)((T_out + cands[i].bt - 1) / cands[i].bt) * ((C_out + cands[i].bc - 1) / cands[i].bc);
        const long long waves = (blocks + slots - 1) / slots;
        const double useful = (double)T_out * C_out / ((double)waves * slots * cands[i].bt * cands[i].bc);
        const double score = useful * cands[i].eff;
        if (score > best_score + 1e-9) { best_score = score; best = i; }
    }
    static const int force = [] { const char* e = getenv("KITTEN_CUDA_IGEMM_TILE"); return e ? atoi(e) : -1; }();
    if (force >= 0 && force < 3) best = force;
    const int bt = cands[best].bt, bc = cands[best].bc;
    dim3 grid((T_out + bt - 1) / bt, (C_out + bc - 1) / bc);
    const bool vec = stride >= 1 && C_in % 16 == 0 && C_out % 4 == 0;
    // Split-K when the grid is under half the GPU's block slots: the reduction is cut into `splits` ranges of at least
    // 16 slices, each block writes partial sums, and one more kernel adds them in a fixed order (deterministic, unlike
    // atomics) with the bias and the residual.
    const int nk = (K * C_in + 15) / 16;
    const long long blocks = (long long)grid.x * grid.y;
    int splits = 1;
    static const bool no_split = [] { const char* e = getenv("KITTEN_CUDA_IGEMM_SPLITK"); return e && e[0] == '0'; }();
    if (!no_split && blocks * 2 <= slots)
        splits = (int)std::max(1LL, std::min({(long long)slots / blocks, (long long)nk / 16, 16LL}));
    if (splits > 1) {
        grid.z = splits;
        auto mark = g_scratch.save();
        float* part = (float*)g_scratch.alloc((size_t)splits * T_out * C_out * sizeof(float));
        if (vec) launch_igemm<false>(best, grid, x, w, nullptr, nullptr, part, T, C_in, C_out, K, pad, dilation, stride, T_out, splits);
        else     launch_igemm<true>(best, grid, x, w, nullptr, nullptr, part, T, C_in, C_out, K, pad, dilation, stride, T_out, splits);
        CUDA_CHECK(cudaGetLastError());
        const size_t n = (size_t)T_out * C_out;
        k_igemm_reduce<<<(unsigned)std::min<size_t>((n + 255) / 256, 4096), 256>>>(part, splits, bias, resid, out, T_out, C_out);
        CUDA_CHECK(cudaGetLastError());
        g_scratch.restore(mark);
        return;
    }
    if (vec) launch_igemm<false>(best, grid, x, w, bias, resid, out, T, C_in, C_out, K, pad, dilation, stride, T_out, 1);
    else     launch_igemm<true>(best, grid, x, w, bias, resid, out, T, C_in, C_out, K, pad, dilation, stride, T_out, 1);
    CUDA_CHECK(cudaGetLastError());
}
