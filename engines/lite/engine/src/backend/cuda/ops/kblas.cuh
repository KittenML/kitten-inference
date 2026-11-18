// ops/kblas.cuh — the part of the cuBLAS API the CUDA backend uses, implemented with the engine's own kernels
// (ops/kblas.cu), so a build with KITTEN_CUDA_OWN_BLAS needs no cuBLAS library (libcublas + libcublasLt are about
// 600 MB; the engine's GEMMs are small and plain fp32, which these kernels cover).
//
// Same meaning as cuBLAS: column-major matrices, host-side alpha/beta, the default (legacy) stream, no TF32 (fp32
// products are computed in fp32, as cuBLAS does with CUBLAS_DEFAULT_MATH and CUBLAS_COMPUTE_32F). The symbols are
// renamed (kblas_*), so this never clashes with a real cuBLAS in the same process.
#pragma once
#include <cuda_runtime.h>
#include <library_types.h>

typedef struct kblas_context* cublasHandle_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15,
} cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0 } cublasMath_t;
typedef enum { CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32I = 72 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 } cublasGemmAlgo_t;

#define cublasCreate               kblas_create
#define cublasSetMathMode          kblas_set_math_mode
#define cublasGemmEx               kblas_gemm_ex
#define cublasGemmStridedBatchedEx kblas_gemm_strided_batched_ex
#define cublasSgemm                kblas_sgemm
#define cublasSgemmStridedBatched  kblas_sgemm_strided_batched

cublasStatus_t kblas_create(cublasHandle_t* handle);
cublasStatus_t kblas_set_math_mode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t kblas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                             int m, int n, int k, const void* alpha,
                             const void* A, cudaDataType Atype, int lda,
                             const void* B, cudaDataType Btype, int ldb, const void* beta,
                             void* C, cudaDataType Ctype, int ldc,
                             cublasComputeType_t compute, cublasGemmAlgo_t algo);
cublasStatus_t kblas_gemm_strided_batched_ex(cublasHandle_t handle, cublasOperation_t transa,
                                             cublasOperation_t transb, int m, int n, int k, const void* alpha,
                                             const void* A, cudaDataType Atype, int lda, long long strideA,
                                             const void* B, cudaDataType Btype, int ldb, long long strideB,
                                             const void* beta, void* C, cudaDataType Ctype, int ldc,
                                             long long strideC, int batch, cublasComputeType_t compute,
                                             cublasGemmAlgo_t algo);
cublasStatus_t kblas_sgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t kblas_sgemm_strided_batched(cublasHandle_t handle, cublasOperation_t transa,
                                           cublasOperation_t transb, int m, int n, int k, const float* alpha,
                                           const float* A, int lda, long long strideA,
                                           const float* B, int ldb, long long strideB, const float* beta,
                                           float* C, int ldc, long long strideC, int batch);
