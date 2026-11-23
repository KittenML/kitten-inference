// C API of the Kitten TTS 0.8 engine (include/kitten_tts08.h). It does what the Python package's
// KittenTTS_1_Cpp.generate_single_chunk does after phonemisation (src/kitten_inference/tts08/cpp_model.py):
// token ids, voice style row, speed, forward, and the 5000-sample tail trim, so both give the same samples.
#include "engine/dev_env.hpp"
#include "kitten_tts08.h"

#include "backend/backend.hpp"
#include "backend/cpu/arm_isa.hpp"
#include "backend/cpu/x86_isa.hpp"
#include "backend/cpu/ops/work_split.hpp"
#include "engine/cpu_affinity.hpp"
#include "engine/load_options.hpp"
#include "engine/memory_mode.hpp"
#include "engine/model.hpp"
#include "capi_internal.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// Phone builds (Android, iOS) get phone defaults: threads 0 picks the fast cores (engine/auto_threads.hpp) and
// kt08_options_init chooses a low-memory mode. Every other build keeps the desktop defaults.
#if defined(__ANDROID__) || (defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE)
#define KT08_PHONE_BUILD 1
#else
#define KT08_PHONE_BUILD 0
#endif

bool kitten_cpu_backend_linked();  // backend/cpu/cpu_backend.cpp

namespace {

constexpr int kStartToken = 0;
constexpr int kEndToken = 10;  // "…" then the pad, as the reference ONNX path
constexpr size_t kTailTrimSamples = 5000;
constexpr int kStyleHalf = KT08_STYLE_DIM / 2;

struct KtSymbol { unsigned cp; int id; };
struct KtRange { unsigned lo, hi; };
#include "symbols.inc"

thread_local std::string t_error;

// KITTEN_TTS08_PHONE_DEFAULTS=1 gives a desktop build the phone defaults, for tests only.
bool phone_defaults() {
    if (KT08_PHONE_BUILD) return true;
    const char* e = std::getenv("KITTEN_TTS08_PHONE_DEFAULTS");
    return e && e[0] == '1' && e[1] == '\0';
}

// Phone default memory mode (see the mobile README for the measurements behind it).
constexpr kt08_memory_mode kPhoneMemoryMode = KT08_MEMORY_EXTREME_LOW;

bool verbose() {
    const char* e = std::getenv("KITTEN_TTS08_VERBOSE");
    return e && e[0] == '1';
}

kt08_status fail(kt08_status s, const std::string& msg) {
    t_error = msg;
    return s;
}

// The GEMM work split (ops/work_split.hpp) is process-wide and every thread of a running synthesis reads it, so it
// may only change while no synthesis runs: syntheses hold this shared, a load that changes it holds it exclusive.
std::shared_mutex& work_split_mutex() {
    static std::shared_mutex m;
    return m;
}

void set_work_split(int chunks_per_thread) {
    std::unique_lock<std::shared_mutex> lock(work_split_mutex());
    if (kt_dynamic_chunks_setting().load() == 0) kt_dynamic_chunks_setting().store(chunks_per_thread);
}

std::mutex& load_mutex() {
    static std::mutex m;
    return m;
}

bool in_ranges(const KtRange* r, size_t n, unsigned cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {  // ranges are sorted and disjoint
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

template <size_t N> size_t count(const KtRange (&)[N]) { return N; }
template <size_t N> size_t count(const KtSymbol (&)[N]) { return N; }

// Python re: 1 = \w, 2 = \s, 0 = other. Outside the generated blocks, code points count as word characters.
int char_class(unsigned cp) {
    if (in_ranges(kWordRanges, count(kWordRanges), cp)) return 1;
    if (in_ranges(kSpaceRanges, count(kSpaceRanges), cp)) return 2;
    if (in_ranges(kCoveredBlocks, count(kCoveredBlocks), cp)) return 0;
    return 1;
}

int symbol_id(unsigned cp) {
    size_t lo = 0, hi = count(kSymbols);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < kSymbols[mid].cp) hi = mid;
        else if (cp > kSymbols[mid].cp) lo = mid + 1;
        else return kSymbols[mid].id;
    }
    return -1;
}

// Strict UTF-8 decoding; false on malformed input.
bool decode_utf8(const char* s, std::vector<unsigned>& out) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s);
    while (*p) {
        unsigned c = *p, cp = 0;
        int extra = 0;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
        else return false;
        ++p;
        for (int i = 0; i < extra; ++i, ++p) {
            if ((*p & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (*p & 0x3F);
        }
        if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) || (extra == 3 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        out.push_back(cp);
    }
    return true;
}

// basic_english_tokenize (re.findall(r"\w+|[^\w\s]")), " ".join, TextCleaner, then 0 ... 10 0.
std::vector<int32_t> phoneme_ids(const std::vector<unsigned>& cps) {
    std::vector<unsigned> joined;
    joined.reserve(cps.size() * 2);
    size_t i = 0;
    bool first = true;
    while (i < cps.size()) {
        const int k = char_class(cps[i]);
        if (k == 2) { ++i; continue; }
        if (!first) joined.push_back(0x20);
        first = false;
        if (k == 1) {
            while (i < cps.size() && char_class(cps[i]) == 1) joined.push_back(cps[i++]);
        } else {
            joined.push_back(cps[i++]);
        }
    }
    std::vector<int32_t> ids;
    ids.reserve(joined.size() + 3);
    ids.push_back(kStartToken);
    for (unsigned cp : joined) {
        const int id = symbol_id(cp);
        if (id >= 0) ids.push_back(id);
    }
    ids.push_back(kEndToken);
    ids.push_back(0);
    return ids;
}

// Rows of the text encoder's position table (bert_embed.pos_weight) in a W1X1 file: the most tokens per call.
// Same reading as _pos_weight_rows in cpp_model.py; 0 when it cannot be told.
size_t pos_weight_rows(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    char magic[4];
    uint32_t hdr[2];
    if (!f.read(magic, 4) || std::memcmp(magic, "W1X1", 4) != 0 || !f.read(reinterpret_cast<char*>(hdr), 8)) return 0;
    const char* want = "bert_embed.pos_weight";
    uint64_t offset = 0;
    bool found = false;
    for (uint32_t i = 0; i < hdr[1]; ++i) {
        char e[80];
        if (!f.read(e, 80)) return 0;
        std::string name(e, strnlen(e, 64));
        const bool is_exact = name == want;
        const bool is_pos = name.size() >= 11 && name.compare(name.size() - 11, 11, ".pos_weight") == 0;
        if (is_exact || (!found && is_pos)) {
            std::memcpy(&offset, e + 64, 8);
            found = true;
            if (is_exact) break;
        }
    }
    if (!found) return 0;
    f.seekg((std::streamoff)offset);
    unsigned char head[4];
    int32_t rows = 0;
    if (!f.read(reinterpret_cast<char*>(head), 4)) return 0;
    const unsigned ndim = head[2] | (head[3] << 8);
    if (ndim < 1 || !f.read(reinterpret_cast<char*>(&rows), 4)) return 0;
    return rows > 0 ? (size_t)rows : 0;
}

[[maybe_unused]] bool arch_has_int8(const std::string& arch_path) {
    std::ifstream f(arch_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return text.find("_int8\"") != std::string::npos;
}

// Shape of a variable-size input, as the Python bindings derive it (python/bindings.cpp).
std::vector<int> shape_for(const InputSpec& spec, size_t n) {
    bool fixed = true;
    for (int d : spec.shape) if (d == 0) { fixed = false; break; }
    if (fixed) return spec.shape;
    const int C = spec.shape.empty() ? 1 : spec.shape.back();
    const int W = std::max((int)spec.shape.size() >= 3 ? spec.shape[1] : 1, 1);
    const int H = (int)(n / ((size_t)C * W));
    if (W > 1) return {H, W, C};
    if (H > 1) return {H, C};
    return {C};
}

class ThreadsScope {
public:
    explicit ThreadsScope(int n) {
#ifdef _OPENMP
        if (n > 0) { saved_ = omp_get_max_threads(); omp_set_num_threads(n); }
#else
        (void)n;
#endif
    }
    ~ThreadsScope() {
#ifdef _OPENMP
        if (saved_ > 0) omp_set_num_threads(saved_);
#endif
    }
    ThreadsScope(const ThreadsScope&) = delete;
    ThreadsScope& operator=(const ThreadsScope&) = delete;

private:
    int saved_ = 0;
};

}  // namespace

struct kt08_model {
    std::unique_ptr<InferenceModel> model;
    kt08_options options{};
    size_t max_tokens = 0;
    int threads = 0;        // threads per call: options.threads, or the phone default when that is 0 (0: OpenMP default)
    std::vector<int> cpus;  // phone default on Android: the cores each call runs on (empty: no restriction)
    std::string profile_path;  // KT08_PROFILE_OUT: where kt08_model_free writes the per-op / per-layer profile (empty: off)
};

extern "C" {

void kt08_options_init(kt08_options* options) {
    if (!options) return;
    std::memset(options, 0, sizeof *options);
    options->struct_size = sizeof *options;
    options->memory_mode = phone_defaults() ? kPhoneMemoryMode : KT08_MEMORY_LATENCY;
}

kt08_status kt08_model_load(const char* arch_json_path, const char* weights_path, const kt08_options* options,
                            kt08_model** out_model) {
    if (!out_model) return fail(KT08_ERR_ARGUMENT, "out_model is NULL");
    *out_model = nullptr;
    if (!arch_json_path || !weights_path || !*arch_json_path || !*weights_path)
        return fail(KT08_ERR_ARGUMENT, "arch_json_path and weights_path are required");
    kt08_options opt;
    kt08_options_init(&opt);
    if (options) {
        // Copy only the fields the caller's struct has (an app built against an older header passes a smaller one).
        const size_t n = std::min<size_t>(options->struct_size ? options->struct_size : sizeof opt, sizeof opt);
        if (n < sizeof(uint32_t)) return fail(KT08_ERR_ARGUMENT, "options->struct_size is too small");
        std::memcpy(&opt, options, n);
        opt.struct_size = sizeof opt;
    }
    if (opt.threads < 0) return fail(KT08_ERR_ARGUMENT, "threads must be 0 (default) or more");
    // Read the enum field as a plain int: a C caller may store any value there, and loading an out-of-range value
    // through the enum type is undefined behaviour in C++ (the range check could be optimised away).
    int32_t memory_mode_raw;
    static_assert(sizeof(opt.memory_mode) == sizeof(memory_mode_raw), "kt08_memory_mode must be 32 bits");
    std::memcpy(&memory_mode_raw, &opt.memory_mode, sizeof memory_mode_raw);
    if (memory_mode_raw < (int32_t)KT08_MEMORY_LATENCY || memory_mode_raw > (int32_t)KT08_MEMORY_EXTREME_LOW)
        return fail(KT08_ERR_ARGUMENT, "memory_mode must be KT08_MEMORY_LATENCY, _LOW or _EXTREME_LOW");
    {
        std::ifstream a(arch_json_path), w(weights_path, std::ios::binary);
        if (!a) return fail(KT08_ERR_IO, std::string("cannot open ") + arch_json_path);
        if (!w) return fail(KT08_ERR_IO, std::string("cannot open ") + weights_path);
    }
#if defined(KITTEN_INT8_NEEDS_DOTPROD)
    // iOS builds compile the int8 kernels with the dot-product instructions (A13 and newer); refuse instead of crashing.
    if (arch_has_int8(arch_json_path) && kt_arm_isa_hw() < KT_ARM_DOTPROD)
        return fail(KT08_ERR_RUNTIME, "int8 models need a CPU with the dot-product instructions (Apple A13 or newer); "
                                      "use an fp32 model");
#endif
    try {
        auto m = std::make_unique<kt08_model>();
        m->options = opt;
        m->max_tokens = pos_weight_rows(weights_path);
        m->threads = opt.threads;
        if (opt.threads == 0 && phone_defaults()) {
            const KtAutoThreads plan = kt_auto_threads_here();
            m->threads = plan.threads;
            m->cpus = plan.cpus;
            // Mixed-speed cores (a prime core and a big one, or same-design cores at two clock speeds): slow cores
            // take fewer tiles. Only which thread computes a tile changes, never a value (cpu_affinity.hpp).
            if (plan.mixed_speeds && kt_dynamic_chunks_setting().load() == 0) set_work_split(4);
            if (verbose()) fprintf(stderr, "kt08: auto %s\n", kt_describe_auto_threads(plan).c_str());
        } else if (opt.threads > 1 && kt_dynamic_chunks_setting().load() == 0) {
            // An explicit thread count on a big.LITTLE phone: with equal static shares every barrier waits for the slowest
            // core, so threads beyond the big cores add little (mini fp32, GB10 2 X925 + 6 A725 as a proxy: 8 threads were
            // 1.15x faster than the 2 big cores, 1.52x with the dynamic split). When the cores the first `threads` threads
            // can land on differ in speed by more than 10%, give each thread several smaller tile ranges and let it take
            // the next as it finishes one (ops/work_split.hpp). Only which thread computes a tile changes, never a value.
            const std::vector<KtCpu> by_speed = kt_cpus_by_speed();
            const size_t n = std::min<size_t>((size_t)opt.threads, by_speed.size());
            if (n > 1 && kt_speeds_differ(by_speed.front().capacity, by_speed[n - 1].capacity))
                set_work_split(kKtMixedCoreChunks);
        }
        const InferenceMemoryMode mode = opt.memory_mode == KT08_MEMORY_LOW ? InferenceMemoryMode::LowMemory
                                         : opt.memory_mode == KT08_MEMORY_EXTREME_LOW ? InferenceMemoryMode::ExtremeLowMemory
                                                                                      : InferenceMemoryMode::Latency;
        KtLoadOverrides o;
        o.fp16_store = opt.fp16_weights ? 1 : 0;
        o.fp16_compute = opt.fp16_compute ? 1 : 0;
        std::lock_guard<std::mutex> lock(load_mutex());  // the engine keeps a few load-time settings in globals
        if (!kitten_cpu_backend_linked() || !set_active_backend("cpu"))
            return fail(KT08_ERR_RUNTIME, "the CPU backend is not available in this build");
        KtLoadOverridesScope scope(o);
        KtAffinityScope cores(m->cpus);
        ThreadsScope threads(m->threads);
        m->model = std::make_unique<InferenceModel>(arch_json_path, weights_path, mode);
        if (const char* pf = kt_dev_getenv("KT08_PROFILE_OUT"); pf && *pf) {  // diagnostics: per-op / per-layer times, written at free
            m->profile_path = pf;
            m->model->set_profile_enabled(true);
        }
        *out_model = m.release();
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return fail(KT08_ERR_MEMORY, "out of memory while loading the model");
    } catch (const std::exception& e) {
        return fail(KT08_ERR_RUNTIME, e.what());
    } catch (...) {
        return fail(KT08_ERR_RUNTIME, "unknown error while loading the model");
    }
}

void kt08_model_free(kt08_model* model) {
    if (model && !model->profile_path.empty() && model->model) {
        if (FILE* f = std::fopen(model->profile_path.c_str(), "w")) {
            for (const auto& kv : model->model->op_profile()) std::fprintf(f, "op\t%s\t%.3f\n", kv.first.c_str(), kv.second);
            for (const auto& kv : model->model->layer_profile()) std::fprintf(f, "layer\t%s\t%.3f\n", kv.first.c_str(), kv.second);
            std::fclose(f);
        }
    }
    delete model;
}

size_t kt08_model_max_tokens(const kt08_model* model) {
    return model ? model->max_tokens : 0;
}

kt08_status kt08_phonemes_to_ids(const char* phonemes_utf8, int32_t* ids, size_t capacity, size_t* out_count) {
    if (!phonemes_utf8 || !out_count) return fail(KT08_ERR_ARGUMENT, "phonemes_utf8 and out_count are required");
    std::vector<unsigned> cps;
    if (!decode_utf8(phonemes_utf8, cps)) return fail(KT08_ERR_ARGUMENT, "phonemes_utf8 is not valid UTF-8");
    const std::vector<int32_t> v = phoneme_ids(cps);
    *out_count = v.size();
    if (v.size() > capacity) {
        if (capacity == 0 && !ids) return KT08_OK;  // size query
        return fail(KT08_ERR_ARGUMENT, "ids capacity is smaller than *out_count");
    }
    if (!ids) return fail(KT08_ERR_ARGUMENT, "ids is NULL");
    std::copy(v.begin(), v.end(), ids);
    return KT08_OK;
}

kt08_status kt08_voice_load(const char* path, float** out_voice, size_t* out_rows) {
    if (!path || !out_voice || !out_rows) return fail(KT08_ERR_ARGUMENT, "path, out_voice and out_rows are required");
    *out_voice = nullptr;
    *out_rows = 0;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return fail(KT08_ERR_IO, std::string("cannot open ") + path);
    const std::streamoff bytes = f.tellg();
    const size_t row_bytes = KT08_STYLE_DIM * sizeof(float);
    if (bytes <= 0 || bytes % (std::streamoff)row_bytes != 0)
        return fail(KT08_ERR_IO, std::string(path) + " is not a voice file (float32 rows of 256 values)");
    float* v = static_cast<float*>(std::malloc((size_t)bytes));
    if (!v) return fail(KT08_ERR_MEMORY, "out of memory while reading the voice");
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(v), bytes)) {
        std::free(v);
        return fail(KT08_ERR_IO, std::string("cannot read ") + path);
    }
    *out_voice = v;
    *out_rows = (size_t)bytes / row_bytes;
    return KT08_OK;
}

size_t kt08_style_row_for_text(const char* chunk_text_utf8, size_t voice_rows) {
    if (!chunk_text_utf8 || voice_rows == 0) return 0;
    size_t n = 0;  // code points: count every byte that does not continue a UTF-8 sequence, as len(str) in Python
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(chunk_text_utf8); *p; ++p)
        if ((*p & 0xC0) != 0x80) ++n;
    return std::min(n, voice_rows - 1);
}

}  // extern "C"

namespace {
// Runs one forward pass (shared by kt08_synthesize and kt08_synthesize_tiny). Arguments are checked by the caller.
std::unordered_map<std::string, ModelIOTensor> run_forward(kt08_model* model, const int32_t* ids, size_t n_ids,
                                                           const float* voice, size_t voice_rows, size_t style_row,
                                                           float speed) {
    std::vector<float> idf(ids, ids + n_ids);  // the engine takes the ids as float32, like the Python package
    const size_t row = std::min(style_row, voice_rows - 1);
    const float* style = voice + row * KT08_STYLE_DIM;
    std::vector<float> style_dec(style, style + kStyleHalf), style_pred(style + kStyleHalf, style + KT08_STYLE_DIM);
    InferenceModel& m = *model->model;
    std::unordered_map<std::string, ModelIOTensor> named;
    for (const InputSpec& spec : m.inputs()) {
        const std::vector<float>* src = spec.name == "phoneme_ids" ? &idf
                                      : spec.name == "style_dec" ? &style_dec
                                      : spec.name == "style_pred" ? &style_pred : nullptr;
        if (!src) throw std::invalid_argument("unexpected model input '" + spec.name + "' (not a Kitten TTS model?)");
        named[spec.name] = ModelIOTensor(src->data(), shape_for(spec, src->size()));
    }
    std::shared_lock<std::shared_mutex> split(work_split_mutex());
    KtAffinityScope cores(model->cpus);
    ThreadsScope threads(model->threads);
    m.set_speed(speed);
    return m.forward(named);
}

kt08_status check_synth_args(kt08_model* model, const int32_t* ids, size_t n_ids, const float* voice,
                             size_t voice_rows, float speed, float** out_audio, size_t* out_samples) {
    if (!out_audio || !out_samples) return fail(KT08_ERR_ARGUMENT, "out_audio and out_samples are required");
    *out_audio = nullptr;
    *out_samples = 0;
    if (!model || !model->model) return fail(KT08_ERR_ARGUMENT, "model is NULL");
    if (!ids || n_ids == 0) return fail(KT08_ERR_ARGUMENT, "ids are required");
    if (!voice || voice_rows == 0) return fail(KT08_ERR_ARGUMENT, "a voice is required");
    if (!std::isfinite(speed) || speed <= 0.0f) return fail(KT08_ERR_ARGUMENT, "speed must be a positive finite value");
    if (model->max_tokens && n_ids > model->max_tokens)
        return fail(KT08_ERR_TOO_LONG, "this chunk has " + std::to_string(n_ids) + " phoneme tokens, more than the model's limit of " +
                                           std::to_string(model->max_tokens) + "; split the text into shorter chunks");
    return KT08_OK;
}

// numpy.linspace(start, stop, n, dtype=float32)[i]: float64 i * step + start, the last one exactly stop, then float32.
float np_linspace(double start, double stop, size_t n, size_t i) {
    if (n <= 1) return (float)start;
    if (i == n - 1) return (float)stop;
    const double step = (stop - start) / (double)(n - 1);
    return (float)((double)i * step + start);
}
}  // namespace

namespace kt08_internal {
// The Python package's chunk finishing for Kitten TTS 2 mini/nano/pico (kt08_synthesize_tiny, and the NPU path): the
// end token's frames are cut, and the start pause too unless first_chunk; samples clipped to [-1, 1]; 5 ms fades.
kt08_status finish_tiny_chunk(const float* wav, size_t n, const float* dur, size_t nd, float speed, int first_chunk,
                              float** out_audio, size_t* out_samples) {
    // Frames per token as the engine counted them (length_regulate: max(1, roundf(d * (1 / speed)))).
    const float scale = 1.0f / speed;
    std::vector<long> fr(nd);
    long total = 0;
    for (size_t i = 0; i < nd; ++i) { fr[i] = std::max(1L, (long)std::lround(dur[i] * scale)); total += fr[i]; }
    const long spf = std::max(1L, (long)std::nearbyint((double)n / (double)std::max(total, 1L)));  // round half even
    const size_t lead = (first_chunk || nd == 0) ? 0 : (size_t)std::max(0L, fr[0] - 1) * spf;
    const size_t trail = nd ? (size_t)std::max(0L, fr[nd - 1] - 1) * spf : 0;
    size_t begin = 0, end = n;
    if (lead + trail + (size_t)spf < n) { begin = lead; end = n - trail; }
    const size_t len = end - begin;
    float* audio = static_cast<float*>(std::malloc(std::max<size_t>(len, 1) * sizeof(float)));
    if (!audio) return fail(KT08_ERR_MEMORY, "out of memory for the audio");
    for (size_t i = 0; i < len; ++i) audio[i] = std::min(1.0f, std::max(-1.0f, wav[begin + i]));
    const size_t f = std::min<size_t>(120, len / 8);
    for (size_t i = 0; i < f; ++i) {
        audio[i] *= np_linspace(0.0, 1.0, f, i);
        audio[len - f + i] *= np_linspace(1.0, 0.0, f, i);
    }
    *out_audio = audio;
    *out_samples = len;
    return KT08_OK;
}

// One forward pass of a loaded model with named float inputs (each tensor's shape derived from its element count as
// for the model's own inputs), on the model's cores and thread count. Throws on engine errors.
std::unordered_map<std::string, ModelIOTensor> forward_named(kt08_model* model,
                                                             const std::unordered_map<std::string, std::vector<float>>& in,
                                                             float speed) {
    InferenceModel& m = *model->model;
    std::unordered_map<std::string, ModelIOTensor> named;
    for (const InputSpec& spec : m.inputs()) {
        auto it = in.find(spec.name);
        if (it == in.end()) throw std::invalid_argument("missing input '" + spec.name + "'");
        named[spec.name] = ModelIOTensor(it->second.data(), shape_for(spec, it->second.size()));
    }
    std::shared_lock<std::shared_mutex> split(work_split_mutex());
    KtAffinityScope cores(model->cpus);
    ThreadsScope threads(model->threads);
    m.set_speed(speed);
    return m.forward(named);
}

kt08_status set_error(kt08_status s, const std::string& msg) { return fail(s, msg); }
}  // namespace kt08_internal

struct kt08_vocab {
    std::unordered_map<unsigned, int32_t> ids;  // code point -> id (every key of vocab.json is one code point)
};

extern "C" {

kt08_status kt08_vocab_load(const char* vocab_json_path, kt08_vocab** out_vocab) {
    if (!out_vocab) return fail(KT08_ERR_ARGUMENT, "out_vocab is NULL");
    *out_vocab = nullptr;
    if (!vocab_json_path) return fail(KT08_ERR_ARGUMENT, "vocab_json_path is NULL");
    std::ifstream f(vocab_json_path, std::ios::binary);
    if (!f) return fail(KT08_ERR_IO, std::string("cannot open ") + vocab_json_path);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
        auto v = std::make_unique<kt08_vocab>();
        size_t p = 0;
        auto skip = [&] { while (p < s.size() && std::isspace((unsigned char)s[p])) ++p; };
        auto hex4 = [&](size_t at) -> unsigned {
            if (at + 4 > s.size() || !std::all_of(s.begin() + at, s.begin() + at + 4,
                                                   [](char c) { return std::isxdigit((unsigned char)c) != 0; }))
                throw std::runtime_error("bad \\u escape");
            return (unsigned)std::stoul(s.substr(at, 4), nullptr, 16);
        };
        auto put_utf8 = [](std::string& o, unsigned cp) {
            if (cp < 0x80) o += (char)cp;
            else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
            else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        };
        auto str = [&]() -> std::string {
            if (s[p] != '"') throw std::runtime_error("expected a string");
            std::string o;
            for (++p; p < s.size() && s[p] != '"'; ++p) {
                if (s[p] != '\\') { o += s[p]; continue; }
                char e = s[++p];
                if (e == 'u') {
                    unsigned cp = hex4(p + 1); p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00) {   // a UTF-16 pair: the low half must follow
                        const unsigned lo = p + 6 < s.size() && s[p + 1] == '\\' && s[p + 2] == 'u' ? hex4(p + 3) : 0;
                        if (lo < 0xDC00 || lo > 0xDFFF) throw std::runtime_error("unpaired \\u surrogate");
                        p += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        throw std::runtime_error("unpaired \\u surrogate");
                    }
                    put_utf8(o, cp);
                } else {
                    o += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
                }
            }
            if (p >= s.size()) throw std::runtime_error("unterminated string");
            ++p;
            return o;
        };
        skip();
        if (p >= s.size() || s[p] != '{') throw std::runtime_error("expected an object");
        ++p;
        for (;;) {
            skip();
            if (p < s.size() && s[p] == '}') break;
            const std::string key = str();
            skip();
            if (p >= s.size() || s[p] != ':') throw std::runtime_error("expected ':'");
            ++p;
            skip();
            size_t used = 0;
            if (p >= s.size() || !std::isdigit((unsigned char)s[p])) throw std::runtime_error("expected a token id (0 or more)");
            long long id = 0;
            try { id = std::stoll(s.substr(p, 24), &used); } catch (const std::out_of_range&) { id = -1; }
            if (id < 0 || id > INT32_MAX) throw std::runtime_error("token id out of range");
            p += used;
            std::vector<unsigned> cps;
            if (!decode_utf8(key.c_str(), cps) || cps.size() != 1)
                throw std::runtime_error("vocab key is not one character: " + key);
            v->ids[cps[0]] = (int32_t)id;
            skip();
            if (p < s.size() && s[p] == ',') { ++p; continue; }
            if (p < s.size() && s[p] == '}') break;
            throw std::runtime_error("expected ',' or '}'");
        }
        if (v->ids.empty()) throw std::runtime_error("empty vocab");
        *out_vocab = v.release();
        return KT08_OK;
    } catch (const std::exception& e) {
        return fail(KT08_ERR_IO, std::string(vocab_json_path) + ": not a vocab.json (" + e.what() + ")");
    }
}

void kt08_vocab_free(kt08_vocab* vocab) {
    delete vocab;
}

kt08_status kt08_vocab_ids(const kt08_vocab* vocab, const char* phonemes_utf8, int32_t* ids, size_t capacity,
                           size_t* out_count) {
    if (!vocab || !phonemes_utf8 || !out_count) return fail(KT08_ERR_ARGUMENT, "vocab, phonemes_utf8 and out_count are required");
    std::vector<unsigned> cps;
    if (!decode_utf8(phonemes_utf8, cps)) return fail(KT08_ERR_ARGUMENT, "phonemes_utf8 is not valid UTF-8");
    std::vector<int32_t> out{kStartToken};
    const auto comma = vocab->ids.find((unsigned)',');
    for (unsigned cp : cps) {
        auto it = vocab->ids.find(cp);
        if (it != vocab->ids.end()) out.push_back(it->second);
        else if ((cp == '(' || cp == ')') && comma != vocab->ids.end()) out.push_back(comma->second);
    }
    out.push_back(kEndToken);
    out.push_back(kStartToken);
    *out_count = out.size();
    if (out.size() > capacity) {
        if (capacity == 0 && !ids) return KT08_OK;  // size query, as kt08_phonemes_to_ids
        return fail(KT08_ERR_ARGUMENT, "ids capacity is smaller than *out_count");
    }
    if (!ids) return fail(KT08_ERR_ARGUMENT, "ids is NULL");
    std::copy(out.begin(), out.end(), ids);
    return KT08_OK;
}

size_t kt08_style_row_for_ids(size_t n_ids, size_t voice_rows) {
    if (voice_rows == 0 || n_ids == 0) return 0;
    return std::min(n_ids - 1, voice_rows - 1);
}

kt08_status kt08_synthesize_tiny(kt08_model* model, const int32_t* ids, size_t n_ids, const float* voice,
                                 size_t voice_rows, size_t style_row, float speed, int first_chunk,
                                 float** out_audio, size_t* out_samples) {
    kt08_status st = check_synth_args(model, ids, n_ids, voice, voice_rows, speed, out_audio, out_samples);
    if (st != KT08_OK) return st;
    if (!(speed >= KT08_TINY_SPEED_MIN && speed <= KT08_TINY_SPEED_MAX))
        return fail(KT08_ERR_ARGUMENT, "speed must be from 0.25 to 4 (1.0 is normal)");
    try {
        auto outputs = run_forward(model, ids, n_ids, voice, voice_rows, style_row, speed);
        auto ia = outputs.find("t_audio");
        auto id = outputs.find("t_dur");
        if (ia == outputs.end() || id == outputs.end())
            return fail(KT08_ERR_RUNTIME, "this model has no t_dur output: not a Kitten TTS 2 mini/nano/pico model");
        return kt08_internal::finish_tiny_chunk(ia->second.begin(), ia->second.numel(), id->second.begin(),
                                                id->second.numel(), speed, first_chunk, out_audio, out_samples);
    } catch (const std::bad_alloc&) {
        return fail(KT08_ERR_MEMORY, "out of memory during synthesis");
    } catch (const std::invalid_argument& e) {
        return fail(KT08_ERR_RUNTIME, e.what());
    } catch (const std::exception& e) {
        return fail(KT08_ERR_RUNTIME, e.what());
    } catch (...) {
        return fail(KT08_ERR_RUNTIME, "unknown error during synthesis");
    }
}

kt08_status kt08_synthesize(kt08_model* model, const int32_t* ids, size_t n_ids, const float* voice, size_t voice_rows,
                            size_t style_row, float speed, float** out_audio, size_t* out_samples) {
    if (!out_audio || !out_samples) return fail(KT08_ERR_ARGUMENT, "out_audio and out_samples are required");
    *out_audio = nullptr;
    *out_samples = 0;
    if (!model || !model->model) return fail(KT08_ERR_ARGUMENT, "model is NULL");
    if (!ids || n_ids == 0) return fail(KT08_ERR_ARGUMENT, "ids are required");
    if (!voice || voice_rows == 0) return fail(KT08_ERR_ARGUMENT, "a voice is required");
    if (!std::isfinite(speed) || speed <= 0.0f) return fail(KT08_ERR_ARGUMENT, "speed must be a positive finite value");
    // Kitten TTS 2 mini/nano/pico loaded here get the same speed range as kt08_synthesize_tiny.
    const std::vector<std::string>& outs = model->model->output_names();
    if (std::find(outs.begin(), outs.end(), "t_dur") != outs.end() &&
        !(speed >= KT08_TINY_SPEED_MIN && speed <= KT08_TINY_SPEED_MAX))
        return fail(KT08_ERR_ARGUMENT, "speed must be from 0.25 to 4 (1.0 is normal)");
    if (model->max_tokens && n_ids > model->max_tokens)
        return fail(KT08_ERR_TOO_LONG, "this chunk has " + std::to_string(n_ids) + " phoneme tokens, more than the model's limit of " +
                                           std::to_string(model->max_tokens) + "; split the text into shorter chunks");
    try {
        std::vector<float> idf(ids, ids + n_ids);  // the engine takes the ids as float32, like the Python package
        const size_t row = std::min(style_row, voice_rows - 1);
        const float* style = voice + row * KT08_STYLE_DIM;
        std::vector<float> style_dec(style, style + kStyleHalf), style_pred(style + kStyleHalf, style + KT08_STYLE_DIM);

        InferenceModel& m = *model->model;
        std::unordered_map<std::string, ModelIOTensor> named;
        for (const InputSpec& spec : m.inputs()) {
            const std::vector<float>* src = spec.name == "phoneme_ids" ? &idf
                                          : spec.name == "style_dec" ? &style_dec
                                          : spec.name == "style_pred" ? &style_pred : nullptr;
            if (!src) return fail(KT08_ERR_RUNTIME, "unexpected model input '" + spec.name + "' (not a Kitten TTS 0.8 model?)");
            named[spec.name] = ModelIOTensor(src->data(), shape_for(spec, src->size()));
        }
        std::unordered_map<std::string, ModelIOTensor> outputs;
        {
            std::shared_lock<std::shared_mutex> split(work_split_mutex());
            KtAffinityScope cores(model->cpus);
            ThreadsScope threads(model->threads);
            m.set_speed(speed);
            outputs = m.forward(named);
        }
        auto it = outputs.find("t_audio");
        if (it == outputs.end()) return fail(KT08_ERR_RUNTIME, "the model has no t_audio output");
        const ModelIOTensor& t = it->second;
        const size_t n = t.numel();
        const size_t keep = n > kTailTrimSamples ? n - kTailTrimSamples : 0;
        float* audio = static_cast<float*>(std::malloc(std::max<size_t>(keep, 1) * sizeof(float)));
        if (!audio) return fail(KT08_ERR_MEMORY, "out of memory for the audio");
        if (keep) std::memcpy(audio, t.begin(), keep * sizeof(float));
        *out_audio = audio;
        *out_samples = keep;
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return fail(KT08_ERR_MEMORY, "out of memory during synthesis");
    } catch (const std::exception& e) {
        return fail(KT08_ERR_RUNTIME, e.what());
    } catch (...) {
        return fail(KT08_ERR_RUNTIME, "unknown error during synthesis");
    }
}

void kt08_free(void* p) {
    std::free(p);
}

int kt08_pin_fastest_cores(int n) {
#if defined(__linux__) || defined(__ANDROID__)
    if (n < 1) return 0;
    try {
        std::unique_lock<std::shared_mutex> split(work_split_mutex());  // may turn on the dynamic split
        const KtPinResult r = kt_pin_fastest_cores(n);
        return r.pinned ? (int)r.cpus.size() : 0;
    } catch (...) {
        return 0;
    }
#else
    (void)n;
    return 0;
#endif
}

const char* kt08_last_error(void) {
    return t_error.c_str();
}

int kt08_api_version(void) {
    return KT08_API_VERSION;
}

const char* kt08_build_info(void) {
    static const std::string info = [] {
        std::string s = "kitten_tts08 api " + std::to_string(KT08_API_VERSION);
#if defined(__aarch64__) || defined(_M_ARM64)
        s += std::string(" arm64 int8-level=") + kt_arm_isa_name(kt_arm_isa());
#if defined(KT_ARM_ISA_DISPATCH)
        s += " (run-time dispatch)";
#endif
#elif defined(KT_X86_ISA_DISPATCH_BUILD) && defined(__x86_64__)
        s += " x86_64 avx2";
        if (kt_x86_avx512_vnni_hw()) s += " int8-level=avx512-vnni (tiny models, run-time dispatch)";
#elif defined(__AVX2__)
        s += " x86_64 avx2";
#elif defined(__x86_64__) || defined(_M_X64)
        s += " x86_64";
#endif
#if defined(KITTEN_OMP_SHIM)
        s += " openmp=shim";
#elif defined(_OPENMP)
        s += " openmp";
#else
        s += " single-thread";
#endif
#if defined(__APPLE__) && defined(KITTEN_HAVE_ACCELERATE)
        s += " accelerate";
#endif
        return s;
    }();
    return info.c_str();
}

}  // extern "C"
