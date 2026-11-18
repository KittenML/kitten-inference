// ops/kblas.cu — the engine's own GEMM kernels behind the cuBLAS subset of ops/kblas.cuh (KITTEN_CUDA_OWN_BLAS).
//
// One tiled kernel, templated on the input/accumulator types, the two transposes and the tile size:
//   C = alpha * op(A) * op(B) + beta * C, column-major, as cuBLAS.
// Each block computes a BM x BN tile of C (BM = 16*TM, BN = 16*TN) with 256 threads, each thread TM x TN outputs;
// the K dimension goes through shared memory BK = 16 at a time. Global loads are coalesced for both layouts of each
// operand (threads walk the contiguous dimension). Each output is one fp32 (or int32) sum in k order, so results are
// the same from run to run. 64 x 64 tiles when they fill the GPU, 32 x 32 tiles for the small GEMMs that would
// otherwise leave most multiprocessors idle.
#include "kblas.cuh"

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kBK      = 16;
constexpr int kThreads = 256;

template <typename Tin, typename Tacc, typename Tout, bool TA, bool TB, int TM, int TN>
__global__ void __launch_bounds__(kThreads)
k_gemm(int m, int n, int k, Tacc alpha, const Tin* __restrict__ A, int lda, long long sA,
       const Tin* __restrict__ B, int ldb, long long sB, Tacc beta, Tout* __restrict__ C, int ldc, long long sC)
{
    constexpr int BM = 16 * TM, BN = 16 * TN;
    __shared__ Tacc As[kBK][BM + 1];
    __shared__ Tacc Bs[kBK][BN + 1];

    const long long z = blockIdx.z;
    A += z * sA;
    B += z * sB;
    C += z * sC;
    const int i0 = blockIdx.x * BM, j0 = blockIdx.y * BN;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;

    Tacc acc[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r)
#pragma unroll
        for (int c = 0; c < TN; ++c) acc[r][c] = Tacc(0);

    for (int p0 = 0; p0 < k; p0 += kBK) {
        // op(A)(i, p): A[p + i*lda] when transposed (contiguous in p), else A[i + p*lda] (contiguous in i).
#pragma unroll
        for (int e = t; e < BM * kBK; e += kThreads) {
            const int i = TA ? e / kBK : e % BM;
            const int p = TA ? e % kBK : e / BM;
            const int gi = i0 + i, gp = p0 + p;
            Tacc v = Tacc(0);
            if (gi < m && gp < k)
                v = (Tacc)(TA ? A[gp + (size_t)gi * lda] : A[gi + (size_t)gp * lda]);
            As[p][i] = v;
        }
        // op(B)(p, j): B[j + p*ldb] when transposed (contiguous in j), else B[p + j*ldb] (contiguous in p).
#pragma unroll
        for (int e = t; e < BN * kBK; e += kThreads) {
            const int j = TB ? e % BN : e / kBK;
            const int p = TB ? e / BN : e % kBK;
            const int gj = j0 + j, gp = p0 + p;
            Tacc v = Tacc(0);
            if (gj < n && gp < k)
                v = (Tacc)(TB ? B[gj + (size_t)gp * ldb] : B[gp + (size_t)gj * ldb]);
            Bs[p][j] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            Tacc a[TM], b[TN];
#pragma unroll
            for (int r = 0; r < TM; ++r) a[r] = As[kk][tx + 16 * r];
#pragma unroll
            for (int c = 0; c < TN; ++c) b[c] = Bs[kk][ty + 16 * c];
#pragma unroll
            for (int r = 0; r < TM; ++r)
#pragma unroll
                for (int c = 0; c < TN; ++c) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }

    // Rows of C go to tx: a warp's stores are contiguous in column-major C.
#pragma unroll
    for (int c = 0; c < TN; ++c) {
        const int j = j0 + ty + 16 * c;
        if (j >= n) continue;
#pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int i = i0 + tx + 16 * r;
            if (i >= m) continue;
            Tout* dst = C + i + (size_t)j * ldc;
            Tacc v = alpha * acc[r][c];
            if (beta != Tacc(0)) v += beta * (Tacc)(*dst);   // beta 0: C is not read (it may hold NaN)
            *dst = (Tout)v;
        }
    }
}

// C = alpha * op(A) * x + beta * C for n == 1 with A transposed (rows of op(A) contiguous in memory): one warp per
// output, the lanes stride over k and add up with shuffles. x(p) = B[p] (B not transposed) or B[p * ldb].
template <bool TB>
__global__ void __launch_bounds__(kThreads)
k_gemv_t(int m, int k, float alpha, const float* __restrict__ A, int lda, long long sA, const float* __restrict__ B,
         int ldb, long long sB, float beta, float* __restrict__ C, long long sC)
{
    const long long z = blockIdx.y;
    A += z * sA; B += z * sB; C += z * sC;
    const int i = blockIdx.x * (kThreads / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= m) return;
    const float* a = A + (size_t)i * lda;
    float acc = 0.f;
    for (int p = lane; p < k; p += 32) acc += a[p] * (TB ? B[(size_t)p * ldb] : B[p]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) C[i] = beta != 0.f ? alpha * acc + beta * C[i] : alpha * acc;
}

// fp32 "TN" GEMM (A transposed, B not: both operands contiguous along k, the layout of nearly every call in the
// engine): BM x 64 tiles (BM = 128 or 64), 256 threads with TM x 4 outputs each, BK = 16, the next tile prefetched into
// registers while the current one is used, 16-byte loads along k (the caller checks alignment: lda, ldb and the batch
// strides multiples of 4, 16-byte aligned bases).
constexpr int kTnBN = 64;
template <int TM>
__global__ void __launch_bounds__(kThreads)
k_sgemm_tn(int m, int n, int k, float alpha, const float* __restrict__ A, int lda, long long sA,
           const float* __restrict__ B, int ldb, long long sB, float beta, float* __restrict__ C, int ldc, long long sC)
{
    constexpr int BM = 16 * TM, AL = BM * kBK / 4 / kThreads;   // float4 loads of A per thread (2 or 1)
    __shared__ float As[kBK][BM + 4];
    __shared__ float Bs[kBK][kTnBN + 4];
    const long long z = blockIdx.z;
    A += z * sA; B += z * sB; C += z * sC;
    const int i0 = blockIdx.x * BM, j0 = blockIdx.y * kTnBN;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int lr = t / 4, lkq = (t % 4) * 4;   // load row (+ 64 per extra A load) and k offset
    float4 ra[AL], rb;
    auto load4 = [&](const float* base, int row, int rows, int ld, int p) -> float4 {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row >= rows) return v;
        const float* q = base + (size_t)row * ld + p;
        if (p + 3 < k) return *reinterpret_cast<const float4*>(q);
        if (p < k) v.x = q[0];
        if (p + 1 < k) v.y = q[1];
        if (p + 2 < k) v.z = q[2];
        return v;
    };
    auto fetch = [&](int p0) {
#pragma unroll
        for (int l = 0; l < AL; ++l) ra[l] = load4(A, i0 + lr + 64 * l, m, lda, p0 + lkq);
        rb = load4(B, j0 + lr, n, ldb, p0 + lkq);
    };
    float acc[TM][4];
#pragma unroll
    for (int r = 0; r < TM; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.f;
    fetch(0);
    for (int p0 = 0; p0 < k; p0 += kBK) {
#pragma unroll
        for (int l = 0; l < AL; ++l) {
            As[lkq + 0][lr + 64 * l] = ra[l].x; As[lkq + 1][lr + 64 * l] = ra[l].y;
            As[lkq + 2][lr + 64 * l] = ra[l].z; As[lkq + 3][lr + 64 * l] = ra[l].w;
        }
        Bs[lkq + 0][lr] = rb.x; Bs[lkq + 1][lr] = rb.y; Bs[lkq + 2][lr] = rb.z; Bs[lkq + 3][lr] = rb.w;
        __syncthreads();
        if (p0 + kBK < k) fetch(p0 + kBK);   // global loads of the next tile overlap the math below
        // Thread (tx, ty) owns rows (r / 4) * 64 + 4 * tx + r % 4 and columns 4 * ty + c: its operands are 16-byte
        // shared-memory loads (2 for A at TM = 8, 1 for B per k step, not 12 single loads), without bank conflicts
        // (8 threads of a quarter warp read 32 consecutive words of A; B is a broadcast).
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a[TM], b[4];
#pragma unroll
            for (int h = 0; h < TM / 4; ++h) {
                const float4 v = *reinterpret_cast<const float4*>(&As[kk][64 * h + 4 * tx]);
                a[4 * h + 0] = v.x; a[4 * h + 1] = v.y; a[4 * h + 2] = v.z; a[4 * h + 3] = v.w;
            }
            const float4 w = *reinterpret_cast<const float4*>(&Bs[kk][4 * ty]);
            b[0] = w.x; b[1] = w.y; b[2] = w.z; b[3] = w.w;
#pragma unroll
            for (int r = 0; r < TM; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int j = j0 + 4 * ty + c;
        if (j >= n) continue;
#pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int i = i0 + 64 * (r / 4) + 4 * tx + r % 4;
            if (i >= m) continue;
            float* dst = C + i + (size_t)j * ldc;
            *dst = beta != 0.f ? alpha * acc[r][c] + beta * *dst : alpha * acc[r][c];
        }
    }
}

int sm_count() {
    static int count = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess || n <= 0)
            n = 1;
        return n;
    }();
    return count;
}

template <typename Tin, typename Tacc, typename Tout, bool TA, bool TB, int T>
void launch_tile(int m, int n, int k, Tacc alpha, const Tin* A, int lda, long long sA, const Tin* B, int ldb,
                 long long sB, Tacc beta, Tout* C, int ldc, long long sC, int batch)
{
    dim3 grid((m + 16 * T - 1) / (16 * T), (n + 16 * T - 1) / (16 * T), batch);
    k_gemm<Tin, Tacc, Tout, TA, TB, T, T><<<grid, kThreads>>>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C,
                                                               ldc, sC);
}

template <typename Tin, typename Tacc, typename Tout, bool TA, bool TB>
void launch_ops(int m, int n, int k, Tacc alpha, const Tin* A, int lda, long long sA, const Tin* B, int ldb,
                long long sB, Tacc beta, Tout* C, int ldc, long long sC, int batch)
{
    const long long big = (long long)((m + 63) / 64) * ((n + 63) / 64) * batch;
    if (big >= sm_count())
        launch_tile<Tin, Tacc, Tout, TA, TB, 4>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    else
        launch_tile<Tin, Tacc, Tout, TA, TB, 2>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
}

template <typename Tin, typename Tacc, typename Tout>
cublasStatus_t gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, Tacc alpha, const Tin* A,
                    int lda, long long sA, const Tin* B, int ldb, long long sB, Tacc beta, Tout* C, int ldc,
                    long long sC, int batch)
{
    if (m < 0 || n < 0 || k < 0 || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
    const bool TA = ta == CUBLAS_OP_T, TB = tb == CUBLAS_OP_T;
    if (k == 0) alpha = Tacc(0);   // C = beta * C; the kernel's k loop does not run
    if constexpr (std::is_same<Tin, float>::value) {   // fp32: GEMV and 128 x 64 TN fast paths
        if (TA && n == 1 && k > 0) {
            dim3 grid((m + kThreads / 32 - 1) / (kThreads / 32), batch);
            if (TB) k_gemv_t<true><<<grid, kThreads>>>(m, k, alpha, A, lda, sA, B, ldb, sB, beta, C, sC);
            else    k_gemv_t<false><<<grid, kThreads>>>(m, k, alpha, A, lda, sA, B, ldb, sB, beta, C, sC);
            return cudaPeekAtLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
        }
        const bool aligned = lda % 4 == 0 && ldb % 4 == 0 && sA % 4 == 0 && sB % 4 == 0 &&
                             reinterpret_cast<uintptr_t>(A) % 16 == 0 && reinterpret_cast<uintptr_t>(B) % 16 == 0;
        if (TA && !TB && aligned && k > 0) {
            // 128 x 64 tiles when they keep at least half the multiprocessors busy, else 64 x 64.
            const long long tiles = (long long)((m + 127) / 128) * ((n + kTnBN - 1) / kTnBN) * batch;
            const bool big = tiles >= sm_count() / 2;
            const int bm = big ? 128 : 64;
            dim3 grid((m + bm - 1) / bm, (n + kTnBN - 1) / kTnBN, batch);
            if (big) k_sgemm_tn<8><<<grid, kThreads>>>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC);
            else     k_sgemm_tn<4><<<grid, kThreads>>>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC);
            return cudaPeekAtLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (TA && TB)        launch_ops<Tin, Tacc, Tout, true, true>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    else if (TA)         launch_ops<Tin, Tacc, Tout, true, false>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    else if (TB)         launch_ops<Tin, Tacc, Tout, false, true>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    else                 launch_ops<Tin, Tacc, Tout, false, false>(m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
    return cudaPeekAtLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t gemm_ex(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha,
                       const void* A, cudaDataType Atype, int lda, long long sA, const void* B, cudaDataType Btype,
                       int ldb, long long sB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long sC,
                       int batch, cublasComputeType_t compute)
{
    if ((ta != CUBLAS_OP_N && ta != CUBLAS_OP_T) || (tb != CUBLAS_OP_N && tb != CUBLAS_OP_T))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F && compute == CUBLAS_COMPUTE_32F)
        return gemm<float, float, float>(ta, tb, m, n, k, *(const float*)alpha, (const float*)A, lda, sA,
                                         (const float*)B, ldb, sB, *(const float*)beta, (float*)C, ldc, sC, batch);
    if (Atype == CUDA_R_8I && Btype == CUDA_R_8I && Ctype == CUDA_R_32I && compute == CUBLAS_COMPUTE_32I)
        return gemm<int8_t, int32_t, int32_t>(ta, tb, m, n, k, *(const int32_t*)alpha, (const int8_t*)A, lda, sA,
                                              (const int8_t*)B, ldb, sB, *(const int32_t*)beta, (int32_t*)C, ldc,
                                              sC, batch);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

}  // namespace

cublasStatus_t kblas_create(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    static int token;
    *handle = reinterpret_cast<cublasHandle_t>(&token);   // stateless: the default stream, host-side scalars
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t kblas_set_math_mode(cublasHandle_t, cublasMath_t mode) {
    return mode == CUBLAS_DEFAULT_MATH ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t kblas_gemm_ex(cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                             const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                             cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                             cublasComputeType_t compute, cublasGemmAlgo_t)
{
    return gemm_ex(transa, transb, m, n, k, alpha, A, Atype, lda, 0, B, Btype, ldb, 0, beta, C, Ctype, ldc, 0, 1,
                   compute);
}

cublasStatus_t kblas_gemm_strided_batched_ex(cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb,
                                             int m, int n, int k, const void* alpha, const void* A,
                                             cudaDataType Atype, int lda, long long strideA, const void* B,
                                             cudaDataType Btype, int ldb, long long strideB, const void* beta,
                                             void* C, cudaDataType Ctype, int ldc, long long strideC, int batch,
                                             cublasComputeType_t compute, cublasGemmAlgo_t)
{
    return gemm_ex(transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype,
                   ldc, strideC, batch, compute);
}

cublasStatus_t kblas_sgemm(cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                           const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta,
                           float* C, int ldc)
{
    return gemm<float, float, float>(transa, transb, m, n, k, *alpha, A, lda, 0, B, ldb, 0, *beta, C, ldc, 0, 1);
}

cublasStatus_t kblas_sgemm_strided_batched(cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb,
                                           int m, int n, int k, const float* alpha, const float* A, int lda,
                                           long long strideA, const float* B, int ldb, long long strideB,
                                           const float* beta, float* C, int ldc, long long strideC, int batch)
{
    return gemm<float, float, float>(transa, transb, m, n, k, *alpha, A, lda, strideA, B, ldb, strideB, *beta, C,
                                     ldc, strideC, batch);
}
