// conv1d.cpp — Conv1dFp32 and ConvTranspose1dFp32 layer implementations
//
// Layout convention: 1-D tensors use shape {C, T, 1}.
// Buffer memory: [T, C] (NHWC with H=T, W=1).
//
// Conv1dFp32 reuses conv2d_fp32_nhwc with kH=kernel_size, kW=1.
// ConvTranspose1dFp32 uses GEMM-per-kernel-position + scatter-add (groups==1)
// or scalar scatter-add fallback (groups > 1).

#include "engine/dev_env.hpp"
#include "conv1d.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include "backend/cpu/ops_neon.hpp"
#include "backend/cpu/arm_isa.hpp"
#include "backend/cpu/x86_isa.hpp"
#include "engine/load_options.hpp"
#include "backend/cpu/ops/profile_internal.hpp"
#include "backend/backend.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// the fp32 conv input pad copy was a single-threaded memcpy of the whole [T, C_in] input (2.4 MB per audio
// second at C=128 in the generator's last stage) in front of every K-split conv. Split it over the team (pure copy).
static void par_memcpy_fp32(float* dst, const float* src, size_t n)
{
#ifdef _OPENMP
    static const bool par = [] { const char* v = kt_dev_getenv("KITTEN_FP32_PAR_PAD"); return !(v && v[0] == '0'); }();
    const int nt = omp_get_max_threads();
    if (par && nt > 1 && !omp_in_parallel() && n >= (size_t)1 << 16) {
#pragma omp parallel num_threads(nt)
        {
            const size_t t = (size_t)omp_get_thread_num(), k = (size_t)omp_get_num_threads();
            const size_t per = (n + k - 1) / k, b = std::min(n, t * per), e = std::min(n, b + per);
            if (e > b) memcpy(dst + b, src + b, (e - b) * sizeof(float));
        }
        return;
    }
#endif
    memcpy(dst, src, n * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────
// helpers — use shared read_f32_tensor() from layer.hpp
// ─────────────────────────────────────────────────────────────────
static inline bool read_f32(const ReadTensorFn& r, const std::string& k,
                             std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

template <typename T>
static void release_vector(std::vector<T>& v)
{
    kitten_hint_vector(v);  // Apple low-memory: drop the pages from phys_footprint before the free
    std::vector<T>().swap(v);
}

static float* conv1d_float_scratch(BufferView view,
                                   std::vector<float>& fallback,
                                   size_t elems)
{
    const size_t bytes = elems * sizeof(float);
    if (view.valid() && view.bytes >= bytes) return view.as<float>();
    if (fallback.size() < elems) fallback.resize(elems);
    return fallback.data();
}

// KT08_LOAD_PAR=0 turns off the multi-threaded weight packing at load time (default on). Read once.
static bool load_par_on()
{
    static const bool on = [] { const char* e = kt_dev_getenv("KT08_LOAD_PAR"); return !(e && e[0] == '0'); }();
    return on;
}

// KT08_LOAD_PROFILE=1: where Conv1dFp32::load_weights spends its time (ms, summed over layers), printed at exit.
namespace {
struct ConvLoadStats {
    bool on = false;
    double read = 0, reorder = 0, wino = 0;
    ConvLoadStats() { const char* e = kt_dev_getenv("KT08_LOAD_PROFILE"); on = e && e[0] == '1'; }
    ~ConvLoadStats() { if (on) std::fprintf(stderr, "CONVLOAD read=%.1f ms reorder=%.1f ms wino_pack=%.1f ms\n", read, reorder, wino); }
    static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};
ConvLoadStats g_conv_load;
}  // namespace

static inline bool load_in_parallel()
{
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

// rows per block of the streamed in-place K-fused conv (0 = off: full padded copy, as before).
static int conv1d_stream_rows()
{
    static const int rows = [] {
        const char* v = kt_dev_getenv("KITTEN_FP32_CONV_STREAM_ROWS");
        return (v && v[0]) ? std::max(0, std::atoi(v)) : 2048;
    }();
    return rows;
}

// set by InferenceModel before it loads weights (true for the low-memory and extreme-low-memory
// modes). The fp32 kernel-3 convs choose their kernel when their weights are packed, from this hint.
static std::atomic<bool> g_conv1d_low_memory_hint{false};
void kitten_conv1d_set_low_memory_hint(bool low_memory) { g_conv1d_low_memory_hint.store(low_memory); }

// Kitten TTS 2 tiny models (set by InferenceModel from the arch name, before the weights are loaded).
static std::atomic<bool> g_conv1d_tiny_hint{false};
void kitten_conv1d_set_tiny_hint(bool tiny)
{
    g_conv1d_tiny_hint.store(tiny);
    kt_x86_isa_set_tiny(tiny);  // x86_64: the int8 weights packed from now on use the AVX-512 + VNNI kernels when the CPU has them
}
static std::atomic<bool> g_conv1d_extreme_hint{false};
void kitten_conv1d_set_extreme_memory_hint(bool extreme) { g_conv1d_extreme_hint.store(extreme); }

// Blocked Winograd F(4,r) for k = 3 / 7 / 11 (any dilation): tiny models in the latency and low-memory modes (the extreme-low-memory mode keeps
// the direct kernel: its weights are 1.3-1.4x larger), ARM NEON and x86 AVX2+FMA. KITTEN_FP32_WINO2=0 turns it off, =1 turns it on in every memory mode.
// Not when sgemm runs on Accelerate's matrix unit (iOS, A13 and newer): there the direct kernel's GEMMs go to the matrix unit and are about
// 2 times faster than Winograd's NEON GEMMs (iPhone 13 mini, latency mode, 2 threads: RTF 0.231 with Winograd, 0.122 without).
bool kitten_accel_sgemm_active();
static bool conv1d_wino2_wanted(int kernel_size, int c_in_g, int c_out)
{
    if (!wino2_supported(kernel_size, c_in_g, c_out)) return false;
    const char* e = kt_dev_getenv("KITTEN_FP32_WINO2");
    if (e && e[0] == '0') return false;
    if (e && e[0] == '1') return true;
    return g_conv1d_tiny_hint.load() && !g_conv1d_extreme_hint.load() && !kitten_accel_sgemm_active();
}

static bool conv1d_direct_by_default()
{
#if defined(__AVX2__) || defined(__AVX512F__)
    return true;
#else
    // ARM: Winograd F(4,3) keeps 6 transformed copies of each kernel-3 weight (+131 MiB after load for mini, peak
    // 820 vs 370 MiB on a set of English sentences, Cortex-X925, 1 thread, measured) and is faster (direct conv +16% CPU
    // time on GB10, +26-28% RTF on the Mac). So the low-memory modes use direct conv; latency mode keeps Winograd.
    // KITTEN_FP32_CONV1D_DIRECT still overrides both.
    return g_conv1d_low_memory_hint.load();
#endif
}

// (LOSSY, opt-in): fp16 compute for the K-fused direct conv1d (backend/cpu/isa/conv_fp16.cpp) when
// KITTEN_FP16_COMPUTE=1 and the CPU has FEAT_FP16 (not A53: then the fp32 kernels run). Forces direct conv (no
// Winograd) for kernel_size > 1 and keeps those weights only in fp16. Read once.
static bool conv1d_fp16_compute_on()
{
#if defined(KT_ARM_ISA_DISPATCH)
    static const bool env_on = [] {
        const char* e = kt_dev_getenv("KITTEN_FP16_COMPUTE");
        return e && e[0] == '1';
    }();
    const int o = kt_load_overrides().fp16_compute;  // C API option for this load, if set
    return (o >= 0 ? o == 1 : env_on) && kt_arm_has_fp16();
#else
    return false;
#endif
}

static bool conv1d_use_direct_kernel(int kernel_size)
{
    const char* v = std::getenv("KITTEN_FP32_CONV1D_DIRECT");
    if (!v || !v[0]) return conv1d_direct_by_default();
    if (v[0] == '0' || !std::strcmp(v, "wino")) return false;
    if (!std::strcmp(v, "1") || !std::strcmp(v, "all")) return true;

    const char* p = v;
    while (*p) {
        while (*p == ',' || *p == ':' || *p == ';' || *p == ' ') ++p;
        char* end = nullptr;
        const long k = std::strtol(p, &end, 10);
        if (end == p) break;
        if (k == kernel_size) return true;
        p = end;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────
// Conv1dFp32
// ─────────────────────────────────────────────────────────────────

void Conv1dFp32::load_weights(const ReadTensorFn& read)
{
    const double cl0 = g_conv_load.on ? ConvLoadStats::now() : 0.0;
    std::vector<float> raw_weight;
    read_f32(read, name + ".weight", raw_weight);
    read_f32(read, name + ".bias",   bias_fp32);
    const double cl1 = g_conv_load.on ? ConvLoadStats::now() : 0.0;

    const int C_in_g = C_in / std::max(groups, 1);
    const int K      = kernel_size * C_in_g;
    if (groups > 1 && C_in % groups != 0) bad_weights(name + ": C_in is not a multiple of groups");
    need_values(raw_weight, layer_elems(name, {C_out, C_in_g, kernel_size}), name + ".weight");
    allow_values(bias_fp32, (size_t)C_out, name + ".bias");

    // Reorder NCHW [C_out, C_in/g, kH] → NHWC [C_out, kH, C_in/g]
    weight_fp32.resize((size_t)C_out * K);
    // Pure copy, one output channel per iteration: split over the threads at load time (KT08_LOAD_PAR=0: one thread).
    #pragma omp parallel for schedule(static) if(load_par_on() && !omp_in_parallel() && (size_t)C_out * K >= 65536)
    for (int co = 0; co < C_out; ++co)
        for (int kh = 0; kh < kernel_size; ++kh)
            for (int ci = 0; ci < C_in_g; ++ci)
                weight_fp32[(size_t)co * K + kh * C_in_g + ci] =
                    raw_weight[(size_t)co * C_in_g * kernel_size + ci * kernel_size + kh];
    release_vector(raw_weight);  // no longer needed; give its pages back before packing
    const double cl2 = g_conv_load.on ? ConvLoadStats::now() : 0.0;
    if (g_conv_load.on) { g_conv_load.read += cl1 - cl0; g_conv_load.reorder += cl2 - cl1; }

    if (groups == 1 && stride == 1) {
        std::vector<float> w_k((size_t)C_out * C_in_g);
        const size_t nb_per_pack = packed_f32_elems(C_out, C_in_g) * sizeof(float);

        const bool fp16c = kernel_size > 1 && conv1d_fp16_compute_on();
        const bool use_direct = fp16c || conv1d_use_direct_kernel(kernel_size);

        if ((kernel_size == 3 || kernel_size == 7 || kernel_size == 11) && !fp16c && conv1d_wino2_wanted(kernel_size, C_in_g, C_out)) {
            // Blocked Winograd F(4,r), r = 3 / 7 / 11: alpha packed panels of G g (filter transform in double), see fp32_conv1d_wino2.cpp
            const int a = wino2_points(kernel_size);
            float* panels[14];
            const double cw0 = g_conv_load.on ? ConvLoadStats::now() : 0.0;
            wino2_pack_weights(weight_fp32.data(), C_out, C_in_g, kernel_size, panels);
            if (g_conv_load.on) g_conv_load.wino += ConvLoadStats::now() - cw0;
            w_wino2_packed_.resize(a);
            for (int p = 0; p < a; ++p)
                w_wino2_packed_[p] = Buffer(panels[p], nb_per_pack, MemSpace::Host,
                    [](void* q){ kitten_delete_floats(q); });
            wino2_r_ = kernel_size;
        } else if (kernel_size == 3 && dilation == 1 && !use_direct) {
            // Winograd F(4,3) filter transform G (6×3)
            static const float G[6][3] = {
                { 1.f/4,    0.f,      0.f   },
                {-1.f/6,  -1.f/6,  -1.f/6  },
                {-1.f/6,   1.f/6,  -1.f/6  },
                { 1.f/24,  1.f/12,  1.f/6  },
                { 1.f/24, -1.f/12,  1.f/6  },
                {   0.f,    0.f,     1.f   },
            };
            w_wino_packed_.resize(6);
            for (int p = 0; p < 6; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float g0 = weight_fp32[(size_t)co*K + 0*C_in_g + ci];
                        float g1 = weight_fp32[(size_t)co*K + 1*C_in_g + ci];
                        float g2 = weight_fp32[(size_t)co*K + 2*C_in_g + ci];
                        w_k[co * C_in_g + ci] = G[p][0]*g0 + G[p][1]*g1 + G[p][2]*g2;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ kitten_delete_floats(p); });
            }
        } else if (kernel_size == 7 && dilation == 1 && !use_direct) {
            // Winograd F(2,7) filter transform G (8×7)
            static const float G[8][7] = {
                {-1.f/36,      0.f,      0.f,      0.f,      0.f,       0.f,     0.f},
                { 1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48,  1.f/48},
                { 1.f/48, -1.f/48,  1.f/48, -1.f/48,  1.f/48, -1.f/48,  1.f/48},
                {-1.f/120,-1.f/60, -1.f/30, -1.f/15, -2.f/15, -4.f/15, -8.f/15},
                {-1.f/120, 1.f/60, -1.f/30,  1.f/15, -2.f/15,  4.f/15, -8.f/15},
                { 1.f/720, 1.f/240, 1.f/80,  3.f/80,  9.f/80, 27.f/80, 81.f/80},
                { 1.f/720,-1.f/240, 1.f/80, -3.f/80,  9.f/80,-27.f/80, 81.f/80},
                {      0.f,     0.f,     0.f,     0.f,     0.f,      0.f,    1.f},
            };
            w_wino_packed_.resize(8);
            for (int p = 0; p < 8; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float acc = 0.f;
                        for (int k = 0; k < 7; ++k)
                            acc += G[p][k] * weight_fp32[(size_t)co*K + k*C_in_g + ci];
                        w_k[co * C_in_g + ci] = acc;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ kitten_delete_floats(p); });
            }
        } else if (kernel_size == 11 && dilation == 1 && !use_direct) {
            // Winograd F(2,11) filter transform G (12×11)
            static const float G[12][11] = {
                {-1.f/14400,      0.f,      0.f,      0.f,      0.f,       0.f,       0.f,       0.f,       0.f,       0.f,       0.f},
                { 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280, 1.f/17280},
                { 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280,-1.f/17280, 1.f/17280},
                {-1.f/30240,-2.f/30240,-4.f/30240,-8.f/30240,-16.f/30240,-32.f/30240,-64.f/30240,-128.f/30240,-256.f/30240,-512.f/30240,-1024.f/30240},
                {-1.f/30240, 2.f/30240,-4.f/30240, 8.f/30240,-16.f/30240, 32.f/30240,-64.f/30240, 128.f/30240,-256.f/30240, 512.f/30240,-1024.f/30240},
                { 1.f/80640, 3.f/80640, 9.f/80640,27.f/80640,81.f/80640,243.f/80640,729.f/80640,2187.f/80640,6561.f/80640,19683.f/80640,59049.f/80640},
                { 1.f/80640,-3.f/80640, 9.f/80640,-27.f/80640,81.f/80640,-243.f/80640,729.f/80640,-2187.f/80640,6561.f/80640,-19683.f/80640,59049.f/80640},
                {-1.f/362880,-4.f/362880,-16.f/362880,-64.f/362880,-256.f/362880,-1024.f/362880,-4096.f/362880,-16384.f/362880,-65536.f/362880,-262144.f/362880,-1048576.f/362880},
                {-1.f/362880, 4.f/362880,-16.f/362880,64.f/362880,-256.f/362880,1024.f/362880,-4096.f/362880,16384.f/362880,-65536.f/362880,262144.f/362880,-1048576.f/362880},
                { 1.f/3628800, 5.f/3628800,25.f/3628800,125.f/3628800,625.f/3628800,3125.f/3628800,15625.f/3628800,78125.f/3628800,390625.f/3628800,1953125.f/3628800,9765625.f/3628800},
                { 1.f/3628800,-5.f/3628800,25.f/3628800,-125.f/3628800,625.f/3628800,-3125.f/3628800,15625.f/3628800,-78125.f/3628800,390625.f/3628800,-1953125.f/3628800,9765625.f/3628800},
                {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f},
            };
            w_wino_packed_.resize(12);
            for (int p = 0; p < 12; ++p) {
                for (int co = 0; co < C_out; ++co) {
                    for (int ci = 0; ci < C_in_g; ++ci) {
                        float acc = 0.f;
                        for (int k = 0; k < 11; ++k)
                            acc += G[p][k] * weight_fp32[(size_t)co*K + k*C_in_g + ci];
                        w_k[co * C_in_g + ci] = acc;
                    }
                }
                float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
                w_wino_packed_[p] = Buffer(pk, nb_per_pack, MemSpace::Host,
                    [](void* p){ kitten_delete_floats(p); });
            }
        } else if (kernel_size == 1) {
            // Single kernel: use per-k pack (fast path via sgemm_f32).
            w_per_k_packed_.resize(1);
            for (int co = 0; co < C_out; ++co)
                memcpy(w_k.data() + (size_t)co * C_in_g,
                       weight_fp32.data() + (size_t)co * K,
                       C_in_g * sizeof(float));
            float* pk = pack_weights_f32(w_k.data(), C_out, C_in_g);
            w_per_k_packed_[0] = Buffer(pk, nb_per_pack, MemSpace::Host,
                [](void* p){ kitten_delete_floats(p); });
        } else {
            // K-split path: merged pack [Co_t, kernel_size, C_in_g, TILE] for kfused GEMM.
            kfused_path = true;
#if defined(KT_ARM_ISA_DISPATCH)
            if (fp16c) {
                size_t hb = 0;
                uint16_t* ph = kt_pack_kmerged_f16(weight_fp32.data(), C_out, C_in_g, kernel_size, &hb);
                w_kmerged_h_ = Buffer(ph, hb, MemSpace::Host, [](void* p){ kt_free_f16(static_cast<uint16_t*>(p)); });
            } else
#endif
            {
            float* pk = pack_merged_weights_f32(weight_fp32.data(), C_out, C_in_g, kernel_size);
            const size_t nb = packed_f32_elems(C_out, C_in_g) * kernel_size * sizeof(float);
            w_kmerged_packed_ = Buffer(pk, nb, MemSpace::Host,
                [](void* p){ kitten_delete_floats(p); });
            }
        }

        kitten_hint_vector(w_k);  // w_k is freed at the end of this block
        // Keep weight_fp32 alive for CUDA cuDNN path (passed as raw weight pointer)
        if (active_backend()->mem_space == MemSpace::Host) {
            release_vector(weight_fp32);
        }
    } else {
        // Fallback path (stride>1, dilation>1, or groups>1): full-K pack
        // Always pack using CPU format: this layer always runs on CPU (required_mem_space=Host).
        float* packed = pack_weights_f32(weight_fp32.data(), C_out, K);
        size_t nb = (size_t)((C_out+3)/4) * K * 4 * sizeof(float);
        w_packed_f32 = Buffer(packed, nb, MemSpace::Host,
            [](void* p){ kitten_delete_floats(p); });
        if (active_backend()->mem_space == MemSpace::Host && groups == 1) {
            release_vector(weight_fp32);
        }
    }
}

void Conv1dFp32::upload_weights(Allocator& /*alloc*/)
{
    // GPU path uses weight_fp32 (CPU buffer) directly; cuDNN uploads it via get_weight_gpu().
    // No persistent GPU copy needed here.
}

void Conv1dFp32::release_runtime_scratch()
{
    release_vector(pad_buf_);
    release_vector(tmp_buf_);
}

Shape Conv1dFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const bool _profile_conv1d_fp32 = g_ops_profile_enabled;
    const double _t0_conv1d_fp32 = _profile_conv1d_fp32 ? now_ms() : 0.0;
    const Shape out_sh_early = output_shape(sh);
    if (_profile_conv1d_fp32)
        g_conv1d_fp32_gmac += 1e-9 * (double)out_sh_early.d1 * C_out * (C_in / std::max(groups, 1)) * kernel_size;

    // ── CUDA: implicit-GEMM convolution, with the planned residual add (fused_residual) in its epilogue ──
    if (in1.where == MemSpace::CUDA && sh.d2 == 1 && active_backend()->conv1d_fp32_cuda) {
        const float* resid = relu ? nullptr : fused_residual;   // relu(conv) + resid: the model adds it after
        active_backend()->conv1d_fp32_cuda(
            in1.as<float>(), weight_fp32.data(), bias_fp32.empty() ? nullptr : bias_fp32.data(), resid,
            out.as<float>(), sh.d0, sh.d1, C_out, kernel_size, stride, padding, dilation, relu, sc.stream);
        if (resid) fused_residual_done = true;
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh_early;
    }

    // ── GPU path: delegate to backend conv2d_fp32_nhwc (cuDNN / CUDA only) ──
    if (in1.where == MemSpace::CUDA || in1.where == MemSpace::HIP) {
        active_backend()->conv2d_fp32_nhwc(
            in1.as<float>(), weight_fp32.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, sh.d2,   // C_in, H=T, W=1
            C_out, kernel_size, 1,
            stride, 1, padding, 0,
            relu,
            nullptr,            // w_packed unused by cuDNN path
            sc.buffers[Scratch::F32_A].as<float>(),
            dilation, 1, sc.stream);
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh_early;
    }

    // ── Metal GPU path: im2col + MPS MatMul via conv1d_fp32_nhwc ─────────────
    // groups > 1 not supported by metal_conv1d_fp32; falls through to CPU path.
    if (in1.where == MemSpace::Metal && groups == 1 && active_backend()->conv1d_fp32_nhwc) {
        active_backend()->conv1d_fp32_nhwc(
            in1.as<float>(), weight_fp32.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, C_out,  // C_in, T, C_out
            kernel_size, stride, padding, dilation,
            relu, sc.stream);
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh_early;
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();
    const Shape out_sh = output_shape(sh);

    // ── Blocked Winograd F(4,7) / F(4,11), any dilation ──
    if (wino2_r_ != 0) {
        const float* ptrs[14];
        for (size_t p = 0; p < w_wino2_packed_.size(); ++p) ptrs[p] = w_wino2_packed_[p].as<float>();
        conv1d_wino2_fp32(in_buf, ptrs, bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                          sh.d1, out_sh.d1, C_in, C_out, padding, dilation, wino2_r_, out_buf, fused_residual);
        if (fused_residual) fused_residual_done = true;
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Winograd F(4,3) path (groups=1, stride=1, kernel_size=3, dilation=1) ──
    if (w_wino_packed_.size() == 6) {
        const float* ptrs[6];
        for (int p = 0; p < 6; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f43_fp32(in_buf, ptrs,
                             bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                             sh.d1, out_sh.d1, C_in, C_out, padding,
                             out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Winograd F(2,7) path (groups=1, stride=1, kernel_size=7, dilation=1) ──
    if (w_wino_packed_.size() == 8) {
        const float* ptrs[8];
        for (int p = 0; p < 8; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f27_fp32(in_buf, ptrs,
                             bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                             sh.d1, out_sh.d1, C_in, C_out, padding,
                             out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Winograd F(2,11) path (groups=1, stride=1, kernel_size=11, dilation=1) ──
    if (w_wino_packed_.size() == 12) {
        const float* ptrs[12];
        for (int p = 0; p < 12; ++p) ptrs[p] = w_wino_packed_[p].as<float>();
        conv1d_wino_f211_fp32(in_buf, ptrs,
                              bias_fp32.empty() ? nullptr : bias_fp32.data(), relu,
                              sh.d1, out_sh.d1, C_in, C_out, padding,
                              out_buf, sc.buffers[Scratch::F32_A].as<float>());
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── K-split path (groups=1, stride=1): kernel_size==1 or kfused GEMM ──
    if (!w_per_k_packed_.empty() || w_kmerged_packed_.valid() || w_kmerged_h_.valid()) {
        // the K-fused GEMM in fp32, or in fp16 (KITTEN_FP16_COMPUTE=1, LOSSY) when packed so.
        auto kfused = [&](const float* A, float* Cp, int Mrows, const float* res) {
#if defined(KT_ARM_ISA_DISPATCH)
            if (w_kmerged_h_.valid()) {
                conv1d_kfused_hgemm(A, w_kmerged_h_.as<uint16_t>(), bias_fp32.empty() ? nullptr : bias_fp32.data(), Cp,
                                    relu, Mrows, C_in, C_out, kernel_size, dilation, res, omp_get_max_threads());
                return;
            }
#endif
            conv1d_kfused_sgemm_f32(A, w_kmerged_packed_.as<float>(), bias_fp32.empty() ? nullptr : bias_fp32.data(),
                                    Cp, relu, Mrows, C_in, C_out, kernel_size, dilation, false, res);
        };
        const int T      = sh.d1;
        const int T_out  = out_sh.d1;
        const int C_in_g = C_in;

        // the planner put this conv in place (output slot == input slot, T_out == T, C_out <= C_in;
        // model.cpp plan_inplace_convs). Instead of a full padded copy of the input, stream it through a window of
        // B + halo rows: block [a, b) reads input rows [a - padding, b + padding); the `halo` rows it shares with the next
        // block are carried in the window, so later blocks never read rows that earlier blocks' outputs overwrote. Same
        // kernel, same per-element accumulation (checked bit-identical end to end). KITTEN_FP32_CONV_STREAM_ROWS=0 = off.
        const int stream_rows = conv1d_stream_rows();
        if (padding > 0 && (w_kmerged_packed_.valid() || w_kmerged_h_.valid()) && (const void*)in_buf == (const void*)out_buf && stream_rows > 0 &&
            T_out == T && C_out <= C_in_g) {
            const int halo = dilation * (kernel_size - 1);
            // B <= T_out keeps the window (B + halo rows) within the scratch the planner sized for a full padded copy,
            // so short inputs never fall back to the layer's own pad_buf_ (which low-memory mode keeps per layer).
            const int B = std::max(1, std::min(T_out, std::max(stream_rows, halo)));
            float* win = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, (size_t)(B + halo) * C_in_g);
            const size_t rowb = (size_t)C_in_g * sizeof(float);
            int carry = 0;
            for (int a = 0; a < T_out; a += B) {
                const int b = std::min(T_out, a + B), m = b - a;
                // window row j <-> input row a - padding + j, j in [0, m + halo)
                for (int j = carry; j < m + halo; ) {
                    const int r = a - padding + j;
                    if (r < 0 || r >= T) { std::memset(win + (size_t)j * C_in_g, 0, rowb); ++j; continue; }
                    const int n = std::min(m + halo - j, T - r);   // contiguous valid rows
                    std::memcpy(win + (size_t)j * C_in_g, in_buf + (size_t)r * C_in_g, (size_t)n * rowb);
                    j += n;
                }
                kfused(win, out_buf + (size_t)a * C_out, m, fused_residual ? fused_residual + (size_t)a * C_out : nullptr);
                std::memmove(win, win + (size_t)m * C_in_g, (size_t)halo * rowb);  // input rows [b - padding, b + padding)
                carry = halo;
            }
            if (fused_residual) fused_residual_done = true;
            if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
            return out_sh;
        }

        // Pad input once (zero-pad left and right by `padding` time steps)
        const float* A_base;
        if (padding > 0) {
            const int T_padded = T + 2 * padding;
            const size_t pad_need = (size_t)T_padded * C_in_g;
            float* pad_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, pad_need);
            memset(pad_buf, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            par_memcpy_fp32(pad_buf + (size_t)padding * C_in_g, in_buf, (size_t)T * C_in_g);
            memset(pad_buf + (size_t)(T + padding) * C_in_g, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            A_base = pad_buf;
        } else {
            A_base = in_buf;
        }

        if (!w_per_k_packed_.empty()) {
            // kernel_size == 1: single sgemm with bias+relu fused
            sgemm_f32(A_base, w_per_k_packed_[0].as<float>(),
                      bias_fp32.empty() ? nullptr : bias_fp32.data(),
                      out_buf, relu, T_out, C_in_g, C_out);
        } else {
            // kernel_size > 1: fused K-split GEMM (accumulates all kp in one pass)
            kfused(A_base, out_buf, T_out, fused_residual);
            if (fused_residual) fused_residual_done = true;
        }
        if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
        return out_sh;
    }

    // ── Fallback: im2col + single SGEMM (stride>1 or groups>1) ──────────────
    const int T_out  = out_sh.d1;
    const int C_in_g = C_in / std::max(groups, 1);
    const size_t col_elems = (size_t)T_out * kernel_size * C_in_g;
    float* col_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, col_elems);
    conv2d_fp32_nhwc(
        in_buf, weight_fp32.data(),
        bias_fp32.empty() ? nullptr : bias_fp32.data(),
        out_buf,
        sh.d0, sh.d1, sh.d2, C_out,
        kernel_size, 1, stride, 1, padding, 0, relu,
        w_packed_f32.as<float>(), col_buf,
        dilation, 1);

    if (_profile_conv1d_fp32) g_conv1d_fp32_ms += now_ms() - _t0_conv1d_fp32;
    return out_sh;
}

// ─────────────────────────────────────────────────────────────────
// Conv1dInt8
// ─────────────────────────────────────────────────────────────────

static int8_t* conv1d_int8_scratch(BufferView view,
                                   std::vector<int8_t>& fallback,
                                   size_t elems)
{
    if (view.valid() && view.bytes >= elems) return view.as<int8_t>();
    if (fallback.size() < elems) fallback.resize(elems);
    return fallback.data();
}

// Frees weights packed by pack_weights_sdot / pack_weights_c1i8. The x86_64 dispatch build allocates them with a header
// in front (backend/cpu/x86_isa.hpp), so they go back through free_packed there.
static void free_int8_pack(void* p)
{
#if defined(KT_X86_ISA_DISPATCH)
    free_packed(static_cast<int8_t*>(p));
#else
    delete[] static_cast<int8_t*>(p);
#endif
}

// fused int8 conv1d on ARM (ops_neon.hpp conv1d_int8_fused), on by default;
// KITTEN_INT8_FUSED=0 restores the per-tap path (kept for A/B and the bit-identity test).
static bool conv1d_int8_fused_on()
{
#ifdef __ARM_NEON
    static const bool on = [] { const char* e = kt_dev_getenv("KITTEN_INT8_FUSED"); return !(e && e[0] == '0'); }();
    return on;
#else
    return false;
#endif
}

void Conv1dInt8::scratch_needed(Shape in, size_t out[Scratch::N]) const
{
    if (active_backend()->mem_space != MemSpace::Host) return;

    const int T = std::max(in.d1, 1);
    const int T_out = std::max(output_shape(in).d1, 1);
    const int C_in_g = C_in / std::max(groups, 1);
    if (C_in_g <= 0 || C_out <= 0 || kernel_size <= 0) return;

    if (!w_per_k_packed_.empty()) {
        const int T_padded = T + 2 * padding;
        if (conv1d_int8_fused_on()) {   // quantized padded input (+64 bytes of read slack) and row scales only
            out[Scratch::I8_A] = std::max(out[Scratch::I8_A], (size_t)T_padded * C_in_g + 64);
            out[Scratch::F32_C] = std::max(out[Scratch::F32_C], (size_t)T_padded * sizeof(float));
            return;
        }
        if (padding > 0 && !int8_rows_split()) {   // the tiny models' x86 path quantizes without a padded copy
            out[Scratch::F32_A] = std::max(
                out[Scratch::F32_A],
                (size_t)T_padded * C_in_g * sizeof(float));
        }
        out[Scratch::I8_A] = std::max(
            out[Scratch::I8_A],
            (size_t)T_padded * C_in_g);
        if (kernel_size > 1 && !int8_rows_split()) {
            out[Scratch::F32_B] = std::max(
                out[Scratch::F32_B],
                (size_t)T_out * C_out * sizeof(float));
        }
        out[Scratch::F32_C] = std::max(
            out[Scratch::F32_C],
            (size_t)T_padded * sizeof(float));
        return;
    }

    const int K = kernel_size * C_in_g;
    const size_t im2col_need = (size_t)T_out * K;
    out[Scratch::F32_A] = std::max(out[Scratch::F32_A],
                                   im2col_need * sizeof(float));
    out[Scratch::I8_A] = std::max(out[Scratch::I8_A], im2col_need);
    out[Scratch::F32_C] = std::max(out[Scratch::F32_C],
                                   (size_t)C_out * sizeof(float));
}

void Conv1dInt8::load_weights(const ReadTensorFn& read)
{
    std::vector<int32_t> shape;
    std::vector<uint8_t> data;
    std::vector<int8_t>  weight_i8;

    if (read(name + ".weight_i8", shape, data)) tensor_values(data, weight_i8, name + ".weight_i8");
    if (read(name + ".w_scales", shape, data)) tensor_values(data, w_scales, name + ".w_scales");
    if (read(name + ".bias", shape, data)) tensor_values(data, bias_fp32, name + ".bias");

    const int C_in_g = C_in / std::max(groups, 1);
    const int K      = kernel_size * C_in_g;
    if (groups > 1 && C_in % groups != 0) bad_weights(name + ": C_in is not a multiple of groups");
    need_values(weight_i8, layer_elems(name, {C_out, C_in_g, kernel_size}), name + ".weight_i8");
    need_values(w_scales, (size_t)C_out, name + ".w_scales");
    allow_values(bias_fp32, (size_t)C_out, name + ".bias");
    if (!weight_i8.empty()) {
        if (stride == 1 && groups <= 1) {
            // Fast path: pack one [C_out, C_in_g] sub-matrix per kernel position.
            // weight_i8 layout: [C_out, K] with K = kernel_size * C_in_g (NHWC order).
            w_per_k_packed_.resize(kernel_size);
            w_per_k_row_sums_.resize(kernel_size);
            std::vector<int8_t> w_k((size_t)C_out * C_in_g);
            for (int k = 0; k < kernel_size; ++k) {
                w_per_k_row_sums_[k].assign(C_out, 0);
                for (int co = 0; co < C_out; ++co)
                    memcpy(w_k.data() + (size_t)co * C_in_g,
                           weight_i8.data() + (size_t)co * K + (size_t)k * C_in_g,
                           C_in_g * sizeof(int8_t));
                for (int co = 0; co < C_out; ++co) {
                    int32_t s = 0;
                    for (int ci = 0; ci < C_in_g; ++ci)
                        s += w_k[(size_t)co * C_in_g + ci];
                    w_per_k_row_sums_[k][co] = s;
                }
#ifdef __ARM_NEON
                // the fused kernel's layout (differs from the SDOT layout only on Armv8.0)
                const bool fz = conv1d_int8_fused_on();
                int8_t* pk = fz ? pack_weights_c1i8(w_k.data(), C_out, C_in_g)
                                : pack_weights_sdot(w_k.data(), C_out, C_in_g);
                size_t nb = fz ? kt_c1i8_packed_bytes(C_out, C_in_g) : kt_int8_packed_bytes(C_out, C_in_g);
#else
                int8_t* pk = pack_weights_sdot(w_k.data(), C_out, C_in_g);
                size_t nb = kt_int8_packed_bytes(C_out, C_in_g);
#endif
                w_per_k_packed_[k] = Buffer(pk, nb, MemSpace::Host, free_int8_pack);
            }
        } else {
            // Fallback: pack full [C_out, K] matrix.
            w_full_row_sums_.assign(C_out, 0);
            for (int co = 0; co < C_out; ++co) {
                int32_t s = 0;
                for (int ki = 0; ki < K; ++ki)
                    s += weight_i8[(size_t)co * K + ki];
                w_full_row_sums_[co] = s;
            }
            int8_t* pk = pack_weights_sdot(weight_i8.data(), C_out, K);
            size_t nb = kt_int8_packed_bytes(C_out, K);
            w_packed_full_ = Buffer(pk, nb, MemSpace::Host, free_int8_pack);
        }

        if (active_backend()->mem_space != MemSpace::Host) {
            // GPU path: dense [C_out, K_pad] row-major INT8.
            // K_pad = ceil4(K) to satisfy GPU GEMM leading-dimension requirements.
            const int K_pad = (K + 3) / 4 * 4;
            std::vector<int8_t> w_dense((size_t)C_out * K_pad, 0);
            for (int co = 0; co < C_out; ++co)
                memcpy(w_dense.data() + (size_t)co * K_pad,
                       weight_i8.data() + (size_t)co * K,
                       K * sizeof(int8_t));
            int8_t* pd = new int8_t[(size_t)C_out * K_pad];
            memcpy(pd, w_dense.data(), (size_t)C_out * K_pad);
            w_gpu_packed_ = Buffer(pd, (size_t)C_out * K_pad, MemSpace::Host,
                [](void* p){ delete[] static_cast<int8_t*>(p); });
        }
    }
    eff_zeros.assign(C_out, 0LL);
    tiny_ = g_conv1d_tiny_hint.load();
}

void Conv1dInt8::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::Host) return;
    if (!w_gpu_packed_.valid()) return;
    Buffer d = alloc.make_buffer(w_gpu_packed_.bytes());
    alloc.copy_h2d(d.ptr(), w_gpu_packed_.ptr(), w_gpu_packed_.bytes());
    w_gpu_packed_ = std::move(d);
}

void Conv1dInt8::release_runtime_scratch()
{
    release_vector(im2col_buf_);
    release_vector(pad_buf_);
    release_vector(tmp_buf_);
    release_vector(inp_scratch_);
    release_vector(req_scratch_);
    release_vector(row_scale_buf_);
}

Shape Conv1dInt8::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const Shape out_sh = output_shape(sh);

    // ── GPU path ──────────────────────────────────────────────────────────────
    if (in1.is_gpu()) {
        active_backend()->conv1d_int8_nhwc(
            in1.as<float>(), w_gpu_packed_.as<int8_t>(),
            w_scales.data(), bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d1, sh.d0, C_out,
            kernel_size, stride, padding, dilation,
            sc.stream);
        return out_sh;
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    const bool _profile_conv1d_int8 = g_ops_profile_enabled;
    const double _t0_conv1d_int8 = _profile_conv1d_int8 ? now_ms() : 0.0;
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();
    const int T      = sh.d1;
    const int T_out  = out_sh.d1;
    const int C_in_g = C_in / std::max(groups, 1);
    if (_profile_conv1d_int8) g_conv1d_int8_gmac += 1e-9 * (double)T_out * C_out * C_in_g * kernel_size;

    // ── Fast K-split path (stride=1, groups=1) ───────────────────────────────
    // Per-row activation quantization: each time-step [1, C_in_g] gets its own
    // scale = max|row| / 127.  This avoids the "outlier inflation" issue with
    // per-tensor quantization, improving accuracy by ~5 dB SNR at negligible cost.
    //
    // GEMM is called once per kernel position k with req_scale = w_scales (no
    // in_scale multiplier).  After GEMM, each row t of the output is multiplied
    // by row_scale[t] to dequantize.
#ifdef __ARM_NEON
    if (!w_per_k_packed_.empty() && conv1d_int8_fused_on()) {
        const int T_padded = T + 2 * padding;
        int8_t* q = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, (size_t)T_padded * C_in_g + 64);
        float* row_scale = conv1d_float_scratch(sc.buffers[Scratch::F32_C], row_scale_buf_, T_padded);
        const int8_t* k_ptrs_stack[32];
        std::vector<const int8_t*> k_ptrs_heap;
        const int8_t** k_ptrs = k_ptrs_stack;
        if (kernel_size > 32) { k_ptrs_heap.resize(kernel_size); k_ptrs = k_ptrs_heap.data(); }
        for (int k = 0; k < kernel_size; ++k) k_ptrs[k] = w_per_k_packed_[k].as<int8_t>();
        conv1d_int8_fused(in_buf, q, row_scale, k_ptrs, w_scales.data(),
                          bias_fp32.empty() ? nullptr : bias_fp32.data(), out_buf,
                          T, padding, T_out, C_in_g, C_out, kernel_size, dilation, omp_get_max_threads());
        if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
        return out_sh;
    }
#endif
#if (defined(__x86_64__) || defined(_M_X64)) && defined(_OPENMP)
    // ── Kitten TTS 2 tiny models on x86-64 (any thread count) ──────────────────
    // Quantize every padded input row once (threads split the rows; padding rows are zero), then each thread takes
    // whole blocks of 64 output rows and runs every tap on one block before the next (gemm_int8_dequant_accum, the
    // one-thread kernel). The block's float sums (64 x C_out) stay in the L2 cache across the taps. The per-tap path
    // below copies the padded input, re-quantizes it for each tap, writes and re-reads a T_out x C_out float buffer per
    // tap and passes over the whole output in memory once per tap, with three barriers per tap: int8 scaled poorly with
    // threads, and 16 single-thread processes saturated the memory bus. Each element still adds the taps in the same
    // order through the same kernel tile (blocks start at multiples of 64 rows), so the one-thread output is unchanged
    // and the layer's output does not depend on the thread count. Kitten TTS 0.8 keeps the per-tap path (its audio
    // must not change).
    if (!w_per_k_packed_.empty() && int8_rows_split()) {
        const int nth = omp_get_max_threads();
        const int T_padded = T + 2 * padding;
        int8_t* q = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, (size_t)T_padded * C_in_g);
        float* row_scale = conv1d_float_scratch(sc.buffers[Scratch::F32_C], row_scale_buf_, T_padded);
        const float* bias_ptr = bias_fp32.empty() ? nullptr : bias_fp32.data();
        const int blocks = (T_out + 63) / 64;
#pragma omp parallel num_threads(nth) if(nth > 1)
        {
#pragma omp for schedule(static)
            for (int t = 0; t < T_padded; ++t) {
                const int ts = t - padding;
                if (ts < 0 || ts >= T) {   // a zero row, as quantize_row_fp32_to_int8 quantizes it
                    std::memset(q + (size_t)t * C_in_g, 0, (size_t)C_in_g);
                    row_scale[t] = std::max(0.f / 127.f, 1e-8f);
                } else {
                    quantize_row_fp32_to_int8(in_buf + (size_t)ts * C_in_g, q + (size_t)t * C_in_g, row_scale[t], C_in_g);
                }
            }
            const int tid = omp_get_thread_num(), nT = omp_get_num_threads();
            const int r0 = (int)((long)blocks * tid / nT) * 64;
            const int r1 = std::min(T_out, (int)((long)blocks * (tid + 1) / nT) * 64);
            constexpr int kTapBlock = 64;
            for (int b0 = r0; b0 < r1; b0 += kTapBlock) {
                const int b1 = std::min(r1, b0 + kTapBlock);
                for (int k = 0; k < kernel_size; ++k) {
                    const int tap_row = k * dilation;
                    gemm_int8_dequant_accum(q + (size_t)(tap_row + b0) * C_in_g, w_per_k_packed_[k].as<int8_t>(),
                                            out_buf + (size_t)b0 * C_out, b1 - b0, C_in_g, C_out, w_scales.data(),
                                            row_scale + tap_row + b0, bias_ptr, k != 0,
                                            w_per_k_row_sums_[k].empty() ? nullptr : w_per_k_row_sums_[k].data(), nullptr);
                }
            }
        }
        if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
        return out_sh;
    }
#endif
    if (!w_per_k_packed_.empty()) {
        // Pad input (single-threaded, outside OMP region)
        const float* A_base;
        const int T_padded = T + 2 * padding;
        if (padding > 0) {
            const size_t pad_need = (size_t)T_padded * C_in_g;
            float* pad_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, pad_need);
            memset(pad_buf, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            memcpy(pad_buf + (size_t)padding * C_in_g, in_buf,
                   (size_t)T * C_in_g * sizeof(float));
            memset(pad_buf + (size_t)(T + padding) * C_in_g, 0,
                   (size_t)padding * C_in_g * sizeof(float));
            A_base = pad_buf;
        } else {
            A_base = in_buf;
        }

        const size_t inp_need = (size_t)T_padded * C_in_g;
        int8_t* inp_scratch = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, inp_need);
        const float* req_scale = w_scales.data();
        float* row_scale = conv1d_float_scratch(sc.buffers[Scratch::F32_C], row_scale_buf_, T_padded);
        float* tmp_buf = nullptr;
        if (kernel_size > 1 && !int8_rows_split()) {
            const size_t tmp_need = (size_t)T_out * C_out;
            tmp_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_B], tmp_buf_, tmp_need);
        }

        const int nthreads = omp_get_max_threads();

        // ── Single-thread path ────────────────────────────────────────────────
        if (nthreads <= 1) {
            const float* bias_ptr = bias_fp32.empty() ? nullptr : bias_fp32.data();
            for (int t = 0; t < T_padded; ++t)
                quantize_row_fp32_to_int8(A_base + (size_t)t * C_in_g,
                                         inp_scratch + (size_t)t * C_in_g,
                                         row_scale[t], C_in_g);

            for (int k = 0; k < kernel_size; ++k) {
                const int tap_row = k * dilation;
                const int8_t* q_src = inp_scratch + (size_t)tap_row * C_in_g;
                const float* tap_row_scale = row_scale + tap_row;

                // Fused INT8 GEMM + dequantization. Conv1dInt8 always runs on
                // CPU (required_mem_space=Host), so call the CPU helper directly.
                gemm_int8_dequant_accum(q_src, w_per_k_packed_[k].as<int8_t>(),
                                        out_buf,
                                        T_out, C_in_g, C_out,
                                        w_scales.data(), tap_row_scale,
                                        bias_ptr, k != 0,
                                        w_per_k_row_sums_[k].empty() ? nullptr : w_per_k_row_sums_[k].data(),
                                        nullptr);
            }
            if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
            return out_sh;
        }

        // ── Multi-thread path ─────────────────────────────────────────────────
        {
            const int8_t* k_ptrs_stack[32];
            std::vector<const int8_t*> k_ptrs_heap;
            const int8_t** k_ptrs = k_ptrs_stack;
            if (kernel_size > (int)(sizeof(k_ptrs_stack) / sizeof(k_ptrs_stack[0]))) {
                k_ptrs_heap.resize(kernel_size);
                k_ptrs = k_ptrs_heap.data();
            }
            for (int k = 0; k < kernel_size; ++k)
                k_ptrs[k] = w_per_k_packed_[k].as<int8_t>();
            const float* bias_ptr = bias_fp32.empty() ? nullptr : bias_fp32.data();
            conv1d_int8_ksplit(
                A_base, k_ptrs, eff_zeros.data(), req_scale,
                bias_ptr, out_buf, tmp_buf,
                inp_scratch, row_scale,
                T_out, C_in_g, C_out, kernel_size, dilation, nthreads);
        }

        if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
        return out_sh;
    }

    // ── Fallback: stride>1 or groups>1 — im2col + single seqgemm_int8 ────────
    const int K = kernel_size * C_in_g;
    const size_t im2col_need = (size_t)T_out * K;
    float* im2col_buf = conv1d_float_scratch(sc.buffers[Scratch::F32_A], im2col_buf_, im2col_need);
    for (int t = 0; t < T_out; ++t) {
        float* row = im2col_buf + (size_t)t * K;
        for (int k = 0; k < kernel_size; ++k) {
            const int t_in = t * stride + k * dilation - padding;
            float* dst = row + (size_t)k * C_in_g;
            if (t_in >= 0 && t_in < T)
                memcpy(dst, in_buf + (size_t)t_in * C_in_g, C_in_g * sizeof(float));
            else
                memset(dst, 0, C_in_g * sizeof(float));
        }
    }
    int8_t* inp_scratch = conv1d_int8_scratch(sc.buffers[Scratch::I8_A], inp_scratch_, im2col_need);
    float* req_scratch = conv1d_float_scratch(sc.buffers[Scratch::F32_C], req_scratch_, C_out);
    // Use CPU seqgemm_int8 directly: Conv1dInt8 always runs on CPU.
    seqgemm_int8_with_rows(im2col_buf, w_packed_full_.as<int8_t>(),
                           w_scales.data(), eff_zeros.data(),
                           bias_fp32.empty() ? nullptr : bias_fp32.data(),
                           inp_scratch, req_scratch, out_buf,
                           T_out, K, C_out, false,
                           w_full_row_sums_.empty() ? nullptr : w_full_row_sums_.data(),
                           nullptr);
    if (_profile_conv1d_int8) g_conv1d_int8_ms += now_ms() - _t0_conv1d_int8;
    return out_sh;
}

// ─────────────────────────────────────────────────────────────────
// ConvTranspose1dFp32
// ─────────────────────────────────────────────────────────────────

void ConvTranspose1dFp32::load_weights(const ReadTensorFn& read)
{
    // PyTorch ConvTranspose1d weight: [C_in, C_out/groups, kernel_size]
    read_f32(read, name + ".weight", weight_fp32);
    read_f32(read, name + ".bias",   bias_fp32);

    // Pre-pack one [C_out_g, C_in_g] GEMM matrix per (group, kernel position)
    const int C_in_g  = C_in  / (groups > 0 ? groups : 1);
    const int C_out_g = C_out / (groups > 0 ? groups : 1);
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0)
        bad_weights(name + ": C_in and C_out must be multiples of groups");
    need_values(weight_fp32, layer_elems(name, {C_in, C_out_g, kernel_size}), name + ".weight");
    allow_values(bias_fp32, (size_t)C_out, name + ".bias");
    w_packed_k.resize((size_t)groups * kernel_size);
    // One (group, kernel position) matrix per iteration, each with its own scratch and its own output: they run on
    // different threads at load time (KT08_LOAD_PAR=0: one thread); the packed values do not depend on who computes them.
    const int n_mats = groups * kernel_size;
    const bool par_load = load_par_on() && !load_in_parallel() && n_mats > 1 && (size_t)n_mats * C_out_g * C_in_g >= 65536;
    #pragma omp parallel if(par_load)
    {
        std::vector<float> tmp((size_t)C_out_g * C_in_g);
        #pragma omp for schedule(dynamic, 1)
        for (int gk = 0; gk < n_mats; ++gk) {
            const int g = gk / kernel_size, k = gk % kernel_size;
            for (int co = 0; co < C_out_g; ++co)
                for (int ci = 0; ci < C_in_g; ++ci)
                    tmp[(size_t)co * C_in_g + ci] =
                        weight_fp32[((size_t)(g*C_in_g + ci) * C_out_g + co) * kernel_size + k];
            float* packed = pack_weights_f32(tmp.data(), C_out_g, C_in_g);
            size_t nb = (size_t)(((C_out_g+3)/4)) * C_in_g * 4 * sizeof(float);
            w_packed_k[g * kernel_size + k] = Buffer(packed, nb, MemSpace::Host,
                [](void* p){ kitten_delete_floats(p); });
        }
        release_vector(tmp);
    }

    if (active_backend()->mem_space == MemSpace::Host) {
        if (groups == 1) release_vector(weight_fp32);
    } else {
        // GPU weight packing:
        // groups == 1: [kernel_size, C_out, C_in] row-major for MPS matmul.
        //   W_gpu[k, c_out, c_in] = weight_fp32[(c_in * C_out + c_out) * kernel_size + k]
        // depthwise (groups == C_in == C_out): [C, kernel_size] row-major.
        //   W_gpu[c, k] = weight_fp32[c * kernel_size + k]  (PyTorch [C,1,ksz] flattened)
        const bool is_depthwise = (groups > 1 && C_in_g == 1 && C_out_g == 1);
        if (groups == 1) {
            w_gpu_data.resize((size_t)kernel_size * C_out_g * C_in_g);
            for (int k = 0; k < kernel_size; ++k)
                for (int co = 0; co < C_out_g; ++co)
                    for (int ci = 0; ci < C_in_g; ++ci)
                        w_gpu_data[(size_t)k * C_out_g * C_in_g + (size_t)co * C_in_g + ci] =
                            weight_fp32[((size_t)ci * C_out_g + co) * kernel_size + k];
        } else if (is_depthwise) {
            // PyTorch layout: [C_in, C_out/groups, kernel_size] = [C, 1, ksz]
            // GPU layout: [C, ksz] — same order, just drop the trivial middle dim
            w_gpu_data.assign(weight_fp32.begin(), weight_fp32.end());
        }
    }
}

void ConvTranspose1dFp32::release_runtime_scratch()
{
    release_vector(pad_buf_);
}

Shape ConvTranspose1dFp32::forward(const TensorView* inputs, int n_inputs,
                                       BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const Shape out_sh = output_shape(sh);

    // ── Metal GPU path ────────────────────────────────────────────────────────
    // Supports groups==1 (GEMM+scatter) and depthwise (groups==C_in==C_out).
    const int C_in_g_  = C_in  / (groups > 0 ? groups : 1);
    const int C_out_g_ = C_out / (groups > 0 ? groups : 1);
    const bool is_depthwise_ = (groups > 1 && C_in_g_ == 1 && C_out_g_ == 1);
    if ((in1.where == MemSpace::Metal || in1.where == MemSpace::CUDA) &&
        (groups == 1 || is_depthwise_) &&
        !w_gpu_data.empty() &&
        active_backend()->conv_transpose1d_fp32_nhwc) {
        active_backend()->conv_transpose1d_fp32_nhwc(
            in1.as<float>(), w_gpu_data.data(),
            bias_fp32.empty() ? nullptr : bias_fp32.data(),
            out.as<float>(),
            sh.d0, sh.d1, C_out,
            kernel_size, stride, padding, output_padding,
            groups,
            sc.stream);
        return out_sh;
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    const bool _profile_conv_transpose = g_ops_profile_enabled;
    const double _t0_conv_transpose = _profile_conv_transpose ? now_ms() : 0.0;
    const float* in_buf  = in1.as<float>();
    float*       out_buf = out.as<float>();

    const int T     = sh.d1;
    const int C_in_ = sh.d0;
    const int T_out = out_sh.d1;
    const int C_out_g = C_out / (groups > 0 ? groups : 1);
    const int C_in_g  = C_in_ / (groups > 0 ? groups : 1);

    memset(out_buf, 0, (size_t)C_out * T_out * sizeof(float));

    if (groups == 1 && !w_packed_k.empty()) {
        // GEMM path: one sgemm per kernel position, then scatter-add.
        const size_t scratch_need = (size_t)T * C_out_g;
        float* scratch = conv1d_float_scratch(sc.buffers[Scratch::F32_A], pad_buf_, scratch_need);

        for (int k = 0; k < kernel_size; ++k) {
            sgemm_f32(in_buf, w_packed_k[k].as<float>(), nullptr, scratch,
                      false, T, C_in_g, C_out_g);

            // for a fixed k the rows t map to distinct rows t_out, so the scatter-add runs on all threads
            // (was serial: gen_up1 66 ms per long text at 16 threads on a 16-core x86 server); the k order is unchanged.
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(!omp_in_parallel() && (size_t)T * C_out_g >= 65536)
#endif
            for (int t = 0; t < T; ++t) {
                int t_out = t * stride + k - padding;
                if (t_out < 0 || t_out >= T_out) continue;
                float*       y = out_buf + (size_t)t_out * C_out;
                const float* s = scratch  + (size_t)t   * C_out_g;
                add_vectors_fp32(y, s, C_out_g);
            }
        }
    } else {
        conv_transpose1d_grouped_fp32(
            in_buf, out_buf, weight_fp32.data(),
            T, T_out, C_in_, C_out, C_in_g, C_out_g,
            groups, stride, padding, kernel_size);
    }

    if (!bias_fp32.empty())
        bias_add_rows_fp32(out_buf, bias_fp32.data(), T_out, C_out);

    if (_profile_conv_transpose) g_conv_transpose_ms += now_ms() - _t0_conv_transpose;
    return out_sh;
}
