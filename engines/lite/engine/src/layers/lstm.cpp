// lstm.cpp — LstmFp32: single-layer (optionally bidirectional) LSTM
//
// Layout: input {C_in, T, 1} → buffer [T, C_in]; output {H_out, T, 1} → [T, H_out]
// where H_out = hidden_size * (1 + bidirectional).
//
// Implementation:
//   1. Pre-compute input projections [T, 4H] via sgemm_f32
//   2. Sequential LSTM scan (forward + optional backward)
//   3. Concatenate forward/backward hidden states

#include "engine/dev_env.hpp"
#include "lstm.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include "engine/cpu_affinity.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// KT08_LSTM_STATS=1 (diagnostics): wall ms of the two input projections, of the two scan sections, and of each scan alone, printed at exit.
// KT08_LSTM_PIN=0 turns off pinning the scans to the fastest cores.
namespace {
struct LstmStats {
    bool on = false;
    std::atomic<long long> proj{0}, sect{0}, fwd{0}, bwd{0}, calls{0};
    LstmStats() { const char* e = kt_dev_getenv("KT08_LSTM_STATS"); on = e && e[0] == '1'; }
    ~LstmStats() { if (on) std::fprintf(stderr, "LSTM calls=%lld proj=%.1f ms sections=%.1f ms scan_fwd=%.1f ms scan_bwd=%.1f ms\n", (long long)calls.load(), proj.load() * 1e-6, sect.load() * 1e-6, fwd.load() * 1e-6, bwd.load() * 1e-6); }
};
LstmStats g_lstm_stats;
inline long long lstm_now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// Threads per direction for the split scan: KT08_LSTM_PAR=0 turns it off, =1 turns it on whatever the cores are. By default it is on
// only on Android and Arm Linux, where it was measured, and only when the cores the team lands on do not differ a lot in speed: the
// barrier waits for the slowest thread, so when the slowest core is under half as fast as the fastest, the plain scan on the fastest
// cores is kept (a 2.3 / 1.8 GHz pair of A53 clusters is fine). Other platforms (Apple CPUs, x86) keep the plain scan unless asked.
// Only LSTMs with at least 256 hidden units per direction split by default (Kitten TTS 2 mini; TTS 0.8 mini and micro): with
// fewer units per thread the per-step barrier costs more than it saves (GB10, 10 threads: pico (96 units) 7% and nano (128)
// 5% slower with the split, mini (256) 1% faster; mini on 8 x Cortex-A53 phones 13-16% faster).
constexpr int kLstmParMinHidden = 256;
int lstm_par_threads(int H) {
    static const int mode = [] { const char* e = kt_dev_getenv("KT08_LSTM_PAR"); return (e && e[0] == '0') ? 0 : (e && e[0] == '1') ? 2 : 1; }();
    if (mode == 0) return 0;
    if (mode == 1 && H < kLstmParMinHidden) return 0;
#ifdef _OPENMP
    const int n = omp_get_max_threads();
    const int per = n >= 8 ? 4 : n >= 4 ? 2 : 0;
    if (per < 2 || H % (per * 8) != 0) return 0;
#if !(defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__)))
    if (mode == 1) return 0;   // not measured on this platform
#endif
    if (mode == 1 && !kt_fast_core_for_rank(0).empty()) {   // mixed speeds (dynamic tile split on): compare the slowest and fastest core of the team
        const std::vector<KtCpu> by_speed = kt_cpus_by_speed();
        const size_t k = std::min<size_t>((size_t)(2 * per), by_speed.size());
        if (k < 2 || by_speed[k - 1].capacity * 2 < by_speed.front().capacity) return 0;
    }
    return per;
#else
    (void)H; return 0;
#endif
}
bool lstm_pin_on() { static const bool v = [] { const char* e = kt_dev_getenv("KT08_LSTM_PIN"); return !(e && e[0] == '0'); }(); return v; }
}  // namespace

// Use shared read_f32_tensor() from layer.hpp; thin alias kept for call-site compatibility.
static inline bool read_f32_lstm(const ReadTensorFn& r, const std::string& k,
                                  std::vector<float>& out)
{ return read_f32_tensor(r, k, out); }

template <typename T>
static void release_vector(std::vector<T>& v)
{
    kitten_hint_vector(v);  // Apple low-memory: drop the pages from phys_footprint before the free
    std::vector<T>().swap(v);
}

// ─────────────────────────────────────────────────────────────────
// LstmFp32::load_weights
// ─────────────────────────────────────────────────────────────────
void LstmFp32::load_weights(const ReadTensorFn& read)
{
    const int H = hidden_size;
    const std::string& n = name;

    read_f32_lstm(read, n + ".weight_ih_l0", w_ih);
    read_f32_lstm(read, n + ".weight_hh_l0", w_hh);
    read_f32_lstm(read, n + ".bias_ih_l0",   b_ih);
    read_f32_lstm(read, n + ".bias_hh_l0",   b_hh);
    const size_t gates = layer_elems(n, {4, H});
    if (w_ih.empty() || w_ih.size() % gates != 0)
        bad_weights(n + ".weight_ih_l0 has " + std::to_string(w_ih.size()) + " values, not a multiple of 4 * " +
                    std::to_string(H));
    input_size_ = (int)(w_ih.size() / gates);
    need_values(w_hh, layer_elems(n, {4, H, H}), n + ".weight_hh_l0");
    allow_values(b_ih, gates, n + ".bias_ih_l0");
    allow_values(b_hh, gates, n + ".bias_hh_l0");

    // Infer C_in from w_ih [4H, C_in].
    // Always pack using CPU format: this layer always runs on CPU (required_mem_space=Host).
    auto make_tb_f32 = [](float* pk, int C_out, int C_in) -> Buffer {
        size_t nb = (size_t)((C_out+3)/4) * C_in * 4 * sizeof(float);
        return Buffer(pk, nb, MemSpace::Host,
            [](void* p){ kitten_delete_floats(p); });
    };

    if (!w_ih.empty()) {
        int C_in_inferred = (int)w_ih.size() / (4 * H);
        float* pk = pack_weights_f32(w_ih.data(), 4*H, C_in_inferred);
        w_ih_packed = make_tb_f32(pk, 4*H, C_in_inferred);
    }
    if (!w_hh.empty()) {
        float* pk = pack_weights_f32(w_hh.data(), 4*H, H);
        w_hh_packed = make_tb_f32(pk, 4*H, H);
    }

    if (bidirectional) {
        read_f32_lstm(read, n + ".weight_ih_l0_reverse", w_ih_r);
        read_f32_lstm(read, n + ".weight_hh_l0_reverse", w_hh_r);
        read_f32_lstm(read, n + ".bias_ih_l0_reverse",   b_ih_r);
        read_f32_lstm(read, n + ".bias_hh_l0_reverse",   b_hh_r);
        need_values(w_ih_r, w_ih.size(), n + ".weight_ih_l0_reverse");
        need_values(w_hh_r, w_hh.size(), n + ".weight_hh_l0_reverse");
        allow_values(b_ih_r, gates, n + ".bias_ih_l0_reverse");
        allow_values(b_hh_r, gates, n + ".bias_hh_l0_reverse");

        if (!w_ih_r.empty()) {
            int C_in_inferred = (int)w_ih_r.size() / (4 * H);
            float* pk = pack_weights_f32(w_ih_r.data(), 4*H, C_in_inferred);
            w_ih_r_packed = make_tb_f32(pk, 4*H, C_in_inferred);
        }
        if (!w_hh_r.empty()) {
            float* pk = pack_weights_f32(w_hh_r.data(), 4*H, H);
            w_hh_r_packed = make_tb_f32(pk, 4*H, H);
        }
    }

    if (!active_backend()->lstm_fp32_metal && !active_backend()->lstm_fp32_cuda) {
        release_vector(w_ih);
        release_vector(w_hh);
        release_vector(w_ih_r);
        release_vector(w_hh_r);
    }
}

void LstmFp32::upload_weights(Allocator& alloc)
{
    if (alloc.mem_space() == MemSpace::CUDA && active_backend()->lstm_fp32_cuda) {
        // Device copies; W_hh transposed to [H, 4H] so that the recurrence kernel's threads (one per gate row) read
        // neighbouring addresses for each hidden unit.
        const int H = hidden_size;
        auto up = [&](const float* p, size_t n) -> Buffer {
            if (!p || n == 0) return Buffer{};
            Buffer d = alloc.make_buffer(n * sizeof(float));
            alloc.copy_h2d(d.ptr(), p, n * sizeof(float));
            return d;
        };
        auto up_t = [&](const std::vector<float>& w) -> Buffer {
            if (w.empty()) return Buffer{};
            std::vector<float> t((size_t)4 * H * H);
            for (int r = 0; r < 4 * H; ++r)
                for (int k = 0; k < H; ++k) t[(size_t)k * 4 * H + r] = w[(size_t)r * H + k];
            return up(t.data(), t.size());
        };
        w_ih_gpu = up(w_ih.data(), w_ih.size());
        b_ih_gpu = up(b_ih.data(), b_ih.size());
        w_hh_gpu = up_t(w_hh);
        b_hh_gpu = up(b_hh.data(), b_hh.size());
        if (bidirectional) {
            w_ih_r_gpu = up(w_ih_r.data(), w_ih_r.size());
            b_ih_r_gpu = up(b_ih_r.data(), b_ih_r.size());
            w_hh_r_gpu = up_t(w_hh_r);
            b_hh_r_gpu = up(b_hh_r.data(), b_hh_r.size());
        }
        return;
    }
    if (!active_backend()->lstm_fp32_metal) return;  // no GPU path
    // Upload all weights as Metal shared buffers via allocator
    auto upload = [&](const std::vector<float>& v) -> Buffer {
        if (v.empty()) return Buffer{};
        Buffer tb = alloc.make_buffer(v.size() * sizeof(float));
        memcpy(tb.as<float>(), v.data(), v.size() * sizeof(float));
        return tb;
    };
    w_ih_gpu = upload(w_ih);
    b_ih_gpu = upload(b_ih);
    w_hh_gpu = upload(w_hh);
    b_hh_gpu = upload(b_hh);
    if (bidirectional) {
        w_ih_r_gpu = upload(w_ih_r);
        b_ih_r_gpu = upload(b_ih_r);
        w_hh_r_gpu = upload(w_hh_r);
        b_hh_r_gpu = upload(b_hh_r);
    }
}

// ─────────────────────────────────────────────────────────────────
// LstmFp32::forward
// ─────────────────────────────────────────────────────────────────
Shape LstmFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const float* input  = in1.as<float>();
    float*       output = out.as<float>();

    const int T      = sh.d1 * sh.d2;   // time steps
    const int C_in_  = sh.d0;           // input channels
    if (C_in_ != input_size_)
        throw std::runtime_error("model weights do not match the input: " + name + " takes " +
                                 std::to_string(input_size_) + " input channels, got " + std::to_string(C_in_));

    const int H      = hidden_size;
    const int H_out  = H * (bidirectional ? 2 : 1);

    // ── CUDA path ──────────────────────────────────────────────────
    if (in1.where == MemSpace::CUDA && active_backend()->lstm_fp32_cuda && w_ih_gpu.valid()) {
        active_backend()->lstm_fp32_cuda(
            input,
            w_ih_gpu.as<float>(), b_ih_gpu.valid() ? b_ih_gpu.as<float>() : nullptr,
            w_hh_gpu.as<float>(), b_hh_gpu.valid() ? b_hh_gpu.as<float>() : nullptr,
            bidirectional ? w_ih_r_gpu.as<float>() : nullptr,
            bidirectional && b_ih_r_gpu.valid() ? b_ih_r_gpu.as<float>() : nullptr,
            bidirectional ? w_hh_r_gpu.as<float>() : nullptr,
            bidirectional && b_hh_r_gpu.valid() ? b_hh_r_gpu.as<float>() : nullptr,
            output, T, C_in_, H, bidirectional, sc.stream);
        return {H_out, sh.d1, sh.d2};
    }

    // ── Metal GPU path ─────────────────────────────────────────────
    if (in1.where == MemSpace::Metal && active_backend()->lstm_fp32_metal) {
        active_backend()->lstm_fp32_metal(
            input,
            w_ih_gpu.as<float>(), b_ih_gpu.valid() ? b_ih_gpu.as<float>() : nullptr,
            w_hh_gpu.as<float>(), b_hh_gpu.valid() ? b_hh_gpu.as<float>() : nullptr,
            bidirectional ? w_ih_r_gpu.as<float>() : nullptr,
            bidirectional ? (b_ih_r_gpu.valid() ? b_ih_r_gpu.as<float>() : nullptr) : nullptr,
            bidirectional ? w_hh_r_gpu.as<float>() : nullptr,
            bidirectional ? (b_hh_r_gpu.valid() ? b_hh_r_gpu.as<float>() : nullptr) : nullptr,
            output, T, C_in_, H, bidirectional, sc.stream);
        return {H_out, sh.d1, sh.d2};
    }

    std::vector<float> h_fwd, h_bwd;
    KittenHintAtExit hint_h(h_fwd, h_bwd);  // forward-time frees leave the footprint

    if (!bidirectional) {
        // ── Unidirectional ────────────────────────────────────
        std::vector<float> proj(T * 4 * H);
        KittenHintAtExit hint_p(proj);  // forward-time frees leave the footprint
        sgemm_f32(input, w_ih_packed.as<float>(),
                                     b_ih.empty() ? nullptr : b_ih.data(),
                                     proj.data(), /*relu=*/false, T, C_in_, 4*H);
        h_fwd.resize((size_t)T * H);
        lstm_scan_fp32(proj.data(), w_hh.data(),
                  b_hh.empty() ? nullptr : b_hh.data(),
                  w_hh_packed.as<float>(), h_fwd.data(), T, H, /*reverse=*/false);
        memcpy(output, h_fwd.data(), (size_t)T * H * sizeof(float));
    } else {
        // ── Bidirectional: run fwd + bwd in parallel OMP sections ─
        std::vector<float> proj_f(T * 4 * H);
        std::vector<float> proj_r(T * 4 * H);
        KittenHintAtExit hint_p(proj_f, proj_r);  // forward-time frees leave the footprint
        h_fwd.resize((size_t)T * H);
        h_bwd.resize((size_t)T * H);

#ifdef _OPENMP
        // the two input projections are independent GEMMs over all T steps; run each on all threads first,
        // then only the two recurrent scans as sections. sgemm_f32 computes every output tile whole on one thread
        // (work_split.hpp), so the projections are bit-identical to the old single-thread-per-section ones.
        if (!omp_in_parallel()) {
            const bool st = g_lstm_stats.on;
            const long long t0 = st ? lstm_now_ns() : 0;
            sgemm_f32(input, w_ih_packed.as<float>(),
                                         b_ih.empty() ? nullptr : b_ih.data(),
                                         proj_f.data(), false, T, C_in_, 4*H);
            sgemm_f32(input, w_ih_r_packed.as<float>(),
                                         b_ih_r.empty() ? nullptr : b_ih_r.data(),
                                         proj_r.data(), false, T, C_in_, 4*H);
            const long long t1 = st ? lstm_now_ns() : 0;
            // Each scan is a chain of T steps; on a CPU of equal cores every step is split over `per` threads per direction
            // (same values, see lstm_scan_pair_fp32), so the scans use all the cores instead of two.
            const int per = lstm_par_threads(H);
            bool par_done = false;
            if (per >= 2 && lstm_scan_pair_fp32(proj_f.data(), proj_r.data(), b_hh.empty() ? nullptr : b_hh.data(),
                                                b_hh_r.empty() ? nullptr : b_hh_r.data(), w_hh_packed.as<float>(),
                                                w_hh_r_packed.as<float>(), h_fwd.data(), h_bwd.data(), T, H, per)) {
                if (st) { const long long t2 = lstm_now_ns(); g_lstm_stats.proj += t1 - t0; g_lstm_stats.sect += t2 - t1; g_lstm_stats.calls += 1; }
                par_done = true;
            } else
            #pragma omp parallel sections num_threads(std::min(2, omp_get_max_threads()))
            {
            #pragma omp section
            {
                // Each scan is one sequential chain of T mat-vec steps that streams the 4H x H recurrent weights from memory: on a
                // big.LITTLE phone it runs on the fastest cores (rank 0 / 1) instead of wherever the scheduler put the thread.
                KtAffinityScope fast(lstm_pin_on() ? kt_fast_core_for_rank(0) : std::vector<int>());
                const long long a = st ? lstm_now_ns() : 0;
                lstm_scan_fp32(proj_f.data(), w_hh.data(),
                          b_hh.empty() ? nullptr : b_hh.data(),
                          w_hh_packed.as<float>(), h_fwd.data(), T, H, false);
                if (st) g_lstm_stats.fwd += lstm_now_ns() - a;
            }
            #pragma omp section
            {
                KtAffinityScope fast(lstm_pin_on() ? kt_fast_core_for_rank(1) : std::vector<int>());
                const long long a = st ? lstm_now_ns() : 0;
                lstm_scan_fp32(proj_r.data(), w_hh_r.data(),
                          b_hh_r.empty() ? nullptr : b_hh_r.data(),
                          w_hh_r_packed.as<float>(), h_bwd.data(), T, H, true);
                if (st) g_lstm_stats.bwd += lstm_now_ns() - a;
            }
            }
            if (st && !par_done) { const long long t2 = lstm_now_ns(); g_lstm_stats.proj += t1 - t0; g_lstm_stats.sect += t2 - t1; g_lstm_stats.calls += 1; }
        } else
        {
            // Already inside a parallel region: the old path (each section on one thread).
        #pragma omp parallel sections if(!omp_in_parallel())
        {
            #pragma omp section
            {
                // in_parallel=false: sgemm detects omp_in_parallel()==true
                // and runs single-threaded on this thread, computing full GEMM.
                sgemm_f32(input, w_ih_packed.as<float>(),
                                             b_ih.empty() ? nullptr : b_ih.data(),
                                             proj_f.data(), false, T, C_in_, 4*H);
                lstm_scan_fp32(proj_f.data(), w_hh.data(),
                          b_hh.empty() ? nullptr : b_hh.data(),
                          w_hh_packed.as<float>(), h_fwd.data(), T, H, false);
            }
            #pragma omp section
            {
                sgemm_f32(input, w_ih_r_packed.as<float>(),
                                             b_ih_r.empty() ? nullptr : b_ih_r.data(),
                                             proj_r.data(), false, T, C_in_, 4*H);
                lstm_scan_fp32(proj_r.data(), w_hh_r.data(),
                          b_hh_r.empty() ? nullptr : b_hh_r.data(),
                          w_hh_r_packed.as<float>(), h_bwd.data(), T, H, true);
            }
        }
        }
#else
        // Sequential fallback (no OMP)
        sgemm_f32(input, w_ih_packed.as<float>(),
                                     b_ih.empty() ? nullptr : b_ih.data(),
                                     proj_f.data(), false, T, C_in_, 4*H);
        lstm_scan_fp32(proj_f.data(), w_hh.data(),
                  b_hh.empty() ? nullptr : b_hh.data(),
                  w_hh_packed.as<float>(), h_fwd.data(), T, H, false);
        sgemm_f32(input, w_ih_r_packed.as<float>(),
                                     b_ih_r.empty() ? nullptr : b_ih_r.data(),
                                     proj_r.data(), false, T, C_in_, 4*H);
        lstm_scan_fp32(proj_r.data(), w_hh_r.data(),
                  b_hh_r.empty() ? nullptr : b_hh_r.data(),
                  w_hh_r_packed.as<float>(), h_bwd.data(), T, H, true);
#endif

        // Interleave fwd + bwd hidden states into output [T, H_out]
        for (int t = 0; t < T; ++t) {
            memcpy(output + (size_t)t * H_out,     h_fwd.data() + (size_t)t * H, H * sizeof(float));
            memcpy(output + (size_t)t * H_out + H, h_bwd.data() + (size_t)t * H, H * sizeof(float));
        }
    }

    return {H_out, sh.d1, sh.d2};
}
