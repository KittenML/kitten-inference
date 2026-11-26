// ops/lstm.cu — LSTM (one or two directions) on CUDA: input projection with cuBLAS, recurrence in one kernel.
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

// gx[t][d][r] += b_ih[r] + b_hh[r] (either bias may be missing)
static __global__ void k_lstm_bias(float* gx, const float* b_ih, const float* b_hh, int T, int D, int G, int d)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (r >= G || t >= T) return;
    float b = 0.f;
    if (b_ih) b += b_ih[r];
    if (b_hh) b += b_hh[r];
    gx[((size_t)t * D + d) * G + r] += b;
}

// One block per direction. Thread r computes gate row r: sum over k of W_hh[r][k] * h[k] (k in order) plus the input
// projection, then threads k < H update the cell (gate order i, f, g, o as PyTorch). W_hh is stored transposed,
// [H][4H], so a warp reads 32 neighbouring gate rows for each k. The reverse direction walks t = T-1 ... 0.
static __global__ void k_lstm_scan(const float* __restrict__ gx, int D, const float* __restrict__ wt0,
                                   const float* __restrict__ wt1, float* __restrict__ out, int H, int T)
{
    extern __shared__ float sm[];
    const int G = 4 * H;
    float* h  = sm;          // [H]
    float* c  = sm + H;      // [H]
    float* gt = sm + 2 * H;  // [G]
    const int d = blockIdx.x;
    const float* W = d == 0 ? wt0 : wt1;
    for (int k = threadIdx.x; k < H; k += blockDim.x) { h[k] = 0.f; c[k] = 0.f; }
    __syncthreads();
    for (int st = 0; st < T; ++st) {
        const int t = d == 0 ? st : T - 1 - st;
        const float* g0 = gx + ((size_t)t * D + d) * G;
        for (int r = threadIdx.x; r < G; r += blockDim.x) {
            float acc = 0.f;
            const float* w = W + r;
#pragma unroll 8
            for (int k = 0; k < H; ++k) acc += w[(size_t)k * G] * h[k];
            gt[r] = acc + g0[r];
        }
        __syncthreads();
        for (int k = threadIdx.x; k < H; k += blockDim.x) {
            const float ig = 1.f / (1.f + expf(-gt[k]));
            const float fg = 1.f / (1.f + expf(-gt[H + k]));
            const float gg = tanhf(gt[2 * H + k]);
            const float og = 1.f / (1.f + expf(-gt[3 * H + k]));
            const float cn = fg * c[k] + ig * gg;
            c[k] = cn;
            const float hn = og * tanhf(cn);
            h[k] = hn;
            out[(size_t)t * D * H + (size_t)d * H + k] = hn;
        }
        __syncthreads();
    }
}


// Multi-block recurrence (ported from the kitten-tts-2-cpp ggml engine's lstm_mb_kernel): each direction is split over
// NB blocks, block b owning hidden units [b*U, (b+1)*U) and their four gate rows, whose W_hh slice stays in shared
// memory for the whole sequence (the one-block kernel re-read all of W_hh from L2 every step). After each step a block
// publishes its part of h in global memory and waits on a per-direction step counter, so all 2 * NB blocks must be
// resident at once: a cooperative launch guarantees it (or fails, and the one-block kernel runs). Each gate's dot
// product is split over TPR threads and reduced with warp shuffles.
template <int KPT>
static __global__ void k_lstm_scan_mb(const float* __restrict__ gx, int D, const float* __restrict__ wt0,
                                      const float* __restrict__ wt1, float* __restrict__ out, int H, int T, int NB,
                                      int TPR, unsigned long long* hx)
{
    // hx: [2 parities][D][H] 64-bit words, (step tag << 32) | float bits of h. A value of step st's output is written
    // with tag st + 1 into parity (st + 1) & 1; the readers of step st + 1 spin on each word until its tag matches. The
    // data and its tag travel in one 64-bit store, so no fence or atomic counter is needed between the blocks (the
    // earlier version paid a fence, an atomic add and a spin on a separate counter per step). Two parities suffice:
    // a block can only overwrite parity p with tag st + 2 after it has read every block's tag st + 1 values, which
    // each block wrote after it had finished reading parity p at step st.
    // KPT > 0: each thread keeps its KPT = H / TPR weights of its gate row in registers (k = lane + TPR * i).
    // KPT = 0: the block's W_hh slice in shared memory, rows padded to H + 4 floats (with a row stride that is a
    // multiple of 32 floats, the 8 rows a warp reads hit the same banks: an 8-way conflict on every load).
    extern __shared__ float sm[];
    const int U = H / NB, R = 4 * U, G = 4 * H;
    const int HS = KPT > 0 ? 0 : H + 4;
    const int d = blockIdx.x / NB, b = blockIdx.x % NB;
    float* ws = sm;                         // KPT = 0: [R][HS], row g*U + j = gate g of unit b*U + j
    float* hs = ws + (size_t)R * HS;        // [H]
    float* gs = hs + H;                     // [R]
    const float* W = d == 0 ? wt0 : wt1;    // transposed, [H][4H]
    const int row = threadIdx.x / TPR, lane = threadIdx.x % TPR;
    float wreg[KPT > 0 ? KPT : 1];
    if constexpr (KPT > 0) {
        const int g = row / U, j = row % U;
#pragma unroll
        for (int i = 0; i < KPT; i++)
            wreg[i] = row < R ? W[(size_t)(lane + TPR * i) * G + g * H + b * U + j] : 0.f;
    } else {
        for (int i = threadIdx.x; i < R * H; i += blockDim.x) {
            const int rw = i / H, k = i % H, g = rw / U, j = rw % U;
            ws[(size_t)rw * HS + k] = W[(size_t)k * G + g * H + b * U + j];
        }
    }
    float c = 0.0f;                         // cell state of unit b*U + threadIdx.x (threads < U)
    for (int st = 0; st < T; st++) {
        const int t = d == 0 ? st : T - 1 - st;
        if (st == 0) {
            for (int k = threadIdx.x; k < H; k += blockDim.x) hs[k] = 0.0f;
        } else {
            const volatile unsigned long long* hp = hx + ((size_t)(st & 1) * D + d) * H;
            for (int k = threadIdx.x; k < H; k += blockDim.x) {
                unsigned long long v;
                do { v = hp[k]; } while ((unsigned)(v >> 32) != (unsigned)st);
                hs[k] = __uint_as_float((unsigned)v);
            }
        }
        __syncthreads();
        {
            float acc = 0.0f;
            if constexpr (KPT > 0) {
#pragma unroll
                for (int i = 0; i < KPT; i++) acc = fmaf(wreg[i], hs[lane + TPR * i], acc);
            } else if (row < R) {
                const float* wr = ws + (size_t)row * HS;
                for (int k = lane; k < H; k += TPR) acc = fmaf(wr[k], hs[k], acc);
            }
            for (int o = TPR / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, TPR);
            if (row < R && lane == 0) {
                const int g = row / U, j = row % U;
                gs[row] = acc + gx[((size_t)t * D + d) * G + g * H + b * U + j];
            }
        }
        __syncthreads();
        if (threadIdx.x < U) {
            const int j = threadIdx.x;
            const float ig = 1.f / (1.f + expf(-gs[j]));
            const float fg = 1.f / (1.f + expf(-gs[U + j]));
            const float gg = tanhf(gs[2 * U + j]);
            const float og = 1.f / (1.f + expf(-gs[3 * U + j]));
            c = fg * c + ig * gg;
            const float hn = og * tanhf(c);
            out[(size_t)t * D * H + (size_t)d * H + b * U + j] = hn;
            volatile unsigned long long* hn_out = hx + ((size_t)((st + 1) & 1) * D + d) * H;
            hn_out[b * U + j] = ((unsigned long long)(unsigned)(st + 1) << 32) | __float_as_uint(hn);
        }
    }
}

// NB blocks per direction such that the block's W_hh slice fits in shared memory; false: use the one-block kernel
static bool lstm_mb_shape(int H, int& NB, int& TPR, int& threads, size_t& shm, int& kpt)
{
    static int max_shm = -1, coop = 0;
    if (max_shm < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_shm, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    }
    static const bool off = [] { const char* e = getenv("KITTEN_CUDA_LSTM_MB"); return e && e[0] == '0'; }();
    static const bool no_reg = [] { const char* e = getenv("KITTEN_CUDA_LSTM_REG"); return e && e[0] == '0'; }();
    if (off || !coop) return false;
    NB = 0;
    for (int nb : {8, 16, 32}) {
        if (H % nb) continue;
        const int U = H / nb;
        const size_t need = ((size_t)4 * U * (H + 4) + H + 4 * U) * sizeof(float);
        if (need <= (size_t)max_shm && 4 * U <= 256) { NB = nb; shm = need; break; }
    }
    if (!NB) return false;
    const int R = 4 * (H / NB);
    TPR = 1;
    while (TPR < 32 && R * TPR * 2 <= 256) TPR *= 2;
    threads = std::max(32, ((R * TPR + 31) / 32) * 32);
    kpt = 0;
    if (!no_reg && H % TPR == 0) {
        const int k = H / TPR;
        if (k == 16 || k == 24 || k == 32 || k == 48 || k == 64) {
            kpt = k;
            shm = ((size_t)H + 4 * (H / NB)) * sizeof(float);
        }
    }
    return true;
}

static const void* lstm_mb_kernel(int kpt)
{
    switch (kpt) {
        case 16: return (const void*)k_lstm_scan_mb<16>;
        case 24: return (const void*)k_lstm_scan_mb<24>;
        case 32: return (const void*)k_lstm_scan_mb<32>;
        case 48: return (const void*)k_lstm_scan_mb<48>;
        case 64: return (const void*)k_lstm_scan_mb<64>;
        default: return (const void*)k_lstm_scan_mb<0>;
    }
}

void cuda_lstm_fp32(
    const float* input, const float* w_ih, const float* b_ih, const float* w_hh_t, const float* b_hh,
    const float* w_ih_r, const float* b_ih_r, const float* w_hh_t_r, const float* b_hh_r,
    float* output, int T, int C_in, int H, bool bidirectional, StreamHandle /*stream*/)
{
    if (T <= 0) return;
    const int D = bidirectional ? 2 : 1, G = 4 * H;
    auto mark = g_scratch.save();
    float* gx = (float*)g_scratch.alloc((size_t)T * D * G * sizeof(float));
    const float alpha = 1.f, beta = 0.f;
    for (int d = 0; d < D; ++d) {
        // column-major: gx_d^T (G x T, ld D*G) = W_ih_d^T (G x C_in) * X^T (C_in x T)
        CUBLAS_CHECK(cublasSgemm(get_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, G, T, C_in, &alpha,
                                 d == 0 ? w_ih : w_ih_r, C_in, input, C_in, &beta, gx + (size_t)d * G, D * G));
        const float* bi = d == 0 ? b_ih : b_ih_r;
        const float* bh = d == 0 ? b_hh : b_hh_r;
        if (bi || bh) {
            k_lstm_bias<<<dim3((G + 255) / 256, T), 256>>>(gx, bi, bh, T, D, G, d);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    int NB = 0, TPR = 1, mb_threads = 0, kpt = 0;
    size_t mb_shm = 0;
    if (lstm_mb_shape(H, NB, TPR, mb_threads, mb_shm, kpt)) {
        const void* kern = lstm_mb_kernel(kpt);
        if (mb_shm > 48 * 1024) {
            static size_t s_shm_set = 0;
            if (mb_shm > s_shm_set) {
                CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)mb_shm));
                s_shm_set = mb_shm;
            }
        }
        const size_t hx_bytes = (size_t)2 * D * H * sizeof(unsigned long long);
        auto* hx = (unsigned long long*)g_scratch.alloc(hx_bytes);
        CUDA_CHECK(cudaMemsetAsync(hx, 0, hx_bytes, 0));   // tag 0 never matches a step (tags start at 1)
        const float* wt1 = bidirectional ? w_hh_t_r : w_hh_t;
        int Di = D, Hi = H, Ti = T;
        void* args[] = {(void*)&gx, &Di, (void*)&w_hh_t, (void*)&wt1, (void*)&output, &Hi, &Ti, &NB, &TPR, (void*)&hx};
        const cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(D * NB), dim3(mb_threads),
                                                          args, mb_shm, 0);
        if (e == cudaSuccess) { g_scratch.restore(mark); return; }
        cudaGetLastError();   // e.g. not all blocks can be resident at once: one block per direction below
    }
    const int threads = std::min(1024, ((G + 31) / 32) * 32);
    const size_t smem = (size_t)(2 * H + G) * sizeof(float);
    k_lstm_scan<<<D, threads, smem>>>(gx, D, w_hh_t, bidirectional ? w_hh_t_r : w_hh_t, output, H, T);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}
