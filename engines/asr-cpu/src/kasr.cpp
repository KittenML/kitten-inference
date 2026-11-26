// Kitten ASR CPU engine: the C API of include/kasr.h on llama.cpp's libllama + libmtmd.
//
// One transcription = log-mel features (mtmd), the audio encoder and projector (mtmd), the decoder over the prompt
// (text + audio embeddings), an optional batched check of a draft, then greedy one-token steps until a stop token.
#include "kasr.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

thread_local std::string g_error;
std::mutex g_init_mutex;
std::mutex g_log_mutex;        // separate from g_init_mutex: backend loading logs while that one is held
bool g_backends_loaded = false;
std::string g_backend_log;     // "loaded ... backend from ..." lines, for kasr_info
bool g_verbose = false;

void set_error(const std::string & msg) { g_error = msg; }

void log_cb(ggml_log_level level, const char * text, void *) {
    if (text == nullptr) {
        return;
    }
    if (std::strstr(text, "loaded ") && std::strstr(text, " backend from ")) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_backend_log += text;
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_error += text;             // the library's own reason, appended to ours
    }
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
}

// The folder of this library, where the ggml CPU backend variants sit next to it. Never the current folder: a library
// of the same name placed there would be loaded instead (an empty string makes kasr_backend_init fail with a clear
// error).
std::string own_dir() {
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR) (void *) &kasr_api_version, &mod)) {
        std::wstring w(32768, L'\0');
        DWORD n = GetModuleFileNameW(mod, w.data(), (DWORD) w.size());
        if (n > 0 && n < w.size()) {
            w.resize(n);
            size_t k = w.find_last_of(L"\\/");
            if (k != std::wstring::npos) {
                w.resize(k);
                int m = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), nullptr, 0, nullptr, nullptr);
                std::string out((size_t) (m > 0 ? m : 0), '\0');
                if (m > 0 && WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), &out[0], m, nullptr, nullptr) == m) {
                    return out;
                }
            }
        }
    }
#else
    Dl_info info;
    if (dladdr((void *) &kasr_api_version, &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        size_t k = p.find_last_of('/');
        if (k != std::string::npos) {
            return p.substr(0, k);
        }
    }
#endif
    return "";
}

int32_t argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    float bv = logits[0];
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

// Op profile (KITTEN_ASR_PROFILE=1): the scheduler computes the graph one node at a time and this callback times
// each node, summed by phase / op / weight type / tensor name without its layer number. Per-node computation adds
// a thread wake-up per node, so the shares are what matters, not the totals. For measurement only.
struct op_profile {
    const char * phase = "";
    std::chrono::steady_clock::time_point t_last;
    bool armed = false;
    std::map<std::string, std::pair<double, long>> * acc = nullptr;
};

bool profile_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * p = (op_profile *) ud;
    if (ask) {
        if (!p->armed) {
            p->t_last = std::chrono::steady_clock::now();
            p->armed = true;
        }
        return true;
    }
    double ms = ms_since(p->t_last);
    std::string name = t->name;
    size_t dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size() && std::isdigit((unsigned char) name[dash + 1])) {
        name = name.substr(0, dash);
    }
    std::string key = std::string(p->phase) + "|" + ggml_op_desc(t) + "|" +
                      (t->src[0] ? ggml_type_name(t->src[0]->type) : "-") + "|" + name;
    auto & e = (*p->acc)[key];
    e.first += ms;
    e.second += 1;
    p->t_last = std::chrono::steady_clock::now();
    return true;
}

}  // namespace

struct kasr_ctx {
    ggml_threadpool_t tp = nullptr;        // decode steps
    ggml_threadpool_t tp_batch = nullptr;  // prompt and draft batches
    decltype(ggml_threadpool_free) * tp_free = nullptr;
    decltype(ggml_threadpool_pause) * tp_pause = nullptr;
    bool pause_ok = true;                  // KITTEN_ASR_POOL_PAUSE=0 keeps the pools polling (for A/B only)
    std::map<std::string, std::pair<double, long>> prof_acc;
    op_profile prof_dec, prof_enc;
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;
    kasr_params params{};
    int32_t n_vocab = 0;
    int32_t n_ctx = 0;
    std::vector<int32_t> tokens;
    std::string text;
    std::string info;
    std::string detok;
    // kasr_transcribe_batch: a second decoding context with one KV stream per slot, made on first use
    llama_context * lctx_b = nullptr;
    int32_t b_slots = 0;
    int32_t b_ctx_seq = 0;
    std::vector<std::vector<int32_t>> b_tokens;
    std::vector<std::string> b_text;
    // Live audio: the decoder state of one growing window is kept between calls with the same stream id. What is
    // kept is the prompt's first text chunk and the complete audio blocks after it (8 s encoder windows), with the
    // encoder input of each block; a later call reuses the longest prefix whose tokens and mel are bit-identical, so
    // its key/value cache rows are exactly what a fresh pass would compute (the decoder is causal).
    struct stream_cache {
        int64_t id = 0;                             // 0: nothing kept
        std::vector<llama_token> head;              // tokens of the first text chunk
        llama_pos head_pos = 0;
        std::vector<std::vector<float>> mel;        // encoder input of each kept block
        std::vector<llama_pos> n_pos;               // decoder positions of each kept block
    } sc;
    bool stream_cache_on = true;                    // KITTEN_ASR_STREAM_CACHE=0 turns it off (A/B only)
    bool resync_on = true;                          // KITTEN_ASR_RESYNC=0 turns draft resync off (A/B only)
    // Callers waiting for this context (kasr_batch_yield): while above 0, kasr_transcribe_batch starts no new window
    // after its first, finishes the windows in flight and returns, marking the rest not started (finish -1).
    std::atomic<int32_t> yield_req{0};
};

extern "C" {

int32_t kasr_api_version(void) { return KASR_API_VERSION; }

int32_t kasr_has_packed_weights(void) { return 1; }

const char * kasr_last_error(void) { return g_error.c_str(); }

int32_t kasr_backend_init(const char * backend_dir, const char * variant) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_backends_loaded) {
        return 1;
    }
    const char * v = std::getenv("KITTEN_ASR_LOG");
    g_verbose = v && *v && std::strcmp(v, "0") != 0;
    llama_log_set(log_cb, nullptr);
    mtmd_helper_log_set(log_cb, nullptr);
#ifdef GGML_BACKEND_DL
    std::string dir = backend_dir && *backend_dir ? std::string(backend_dir) : own_dir();
    if (dir.empty()) {
        set_error("could not find the folder of the kitten-asr library (where its CPU backends are)");
        return 0;
    }
    if (variant && *variant) {
#ifdef _WIN32
        std::string path = dir + "\\ggml-cpu-" + variant + ".dll";
#else
        std::string path = dir + "/libggml-cpu-" + variant + ".so";
#endif
        if (!ggml_backend_load(path.c_str())) {
            set_error("could not load the CPU backend " + path + " (the CPU may lack its instructions)");
            return 0;
        }
    } else {
        ggml_backend_load_all_from_path(dir.c_str());
    }
#else
    (void) backend_dir;
    if (variant && *variant) {
        set_error("this build has one CPU backend; a CPU variant cannot be chosen");
        return 0;
    }
#endif
    llama_backend_init();
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == nullptr) {
        set_error("no CPU backend could be loaded for this CPU (x86-64 CPUs need AVX2: Intel 2013+, AMD 2015+)");
        return 0;
    }
    g_backends_loaded = true;
    return 1;
}

kasr_params kasr_default_params(void) {
    kasr_params p;
    unsigned hc = std::thread::hardware_concurrency();
    p.n_threads = (int32_t) std::max(1u, hc / 2);
    p.n_threads_batch = 0;
    p.n_ctx = 2048;
    p.n_batch = 512;
    p.flash_attn = -1;
    p.audio_window = -1;
    p.use_gpu = 0;
    p.kv_type = 0;
    p.warmup = 1;
    p.repack = 1;
    return p;
}

kasr_ctx * kasr_load(const char * decoder_gguf, const char * encoder_gguf, const kasr_params * params) {
    g_error.clear();
    if (!kasr_backend_init(nullptr, nullptr)) {
        return nullptr;
    }
    kasr_params p = params ? *params : kasr_default_params();
    if (p.n_threads <= 0) {
        p.n_threads = kasr_default_params().n_threads;
    }
    if (p.n_threads_batch <= 0) {
        p.n_threads_batch = p.n_threads;
    }
    if (p.n_ctx <= 0) {
        p.n_ctx = 2048;
    }
    if (p.n_batch <= 0) {
        p.n_batch = 512;
    }
    if (p.use_gpu && ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        p.use_gpu = 0;   // no GPU backend in this build or no GPU on this machine: plain CPU, with the thread pools
    }
    if (p.use_gpu) {
        // kitten: no CPU weight repacking on the GPU. With extra buffer types on, llama.cpp 9048bad put every q8_0
        // layer weight in the CPU_REPACK buffer (302 MiB for tiny) although all 29 layers were "offloaded", so each
        // layer's matrix products ran on the CPU between Metal kernels (macOS virtual machine, stack samples).
        p.repack = 0;
    }
    auto * c = new kasr_ctx();
    c->params = p;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.use_gpu ? -1 : 0;
    mp.use_extra_bufts = p.repack != 0;
    c->model = llama_model_load_from_file(decoder_gguf, mp);
    if (!c->model) {
        set_error(std::string("could not load the decoder ") + decoder_gguf + ": " + g_error);
        kasr_free(c);
        return nullptr;
    }
    c->vocab = llama_model_get_vocab(c->model);
    c->n_vocab = llama_vocab_n_tokens(c->vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) p.n_ctx;
    cp.n_batch = (uint32_t) p.n_batch;
    cp.n_ubatch = (uint32_t) p.n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = p.n_threads;
    cp.n_threads_batch = p.n_threads_batch;
    cp.flash_attn_type = (llama_flash_attn_type) p.flash_attn;
    cp.no_perf = true;
    cp.op_offload = p.use_gpu != 0;
    if (p.kv_type > 0) {
        cp.type_k = (ggml_type) p.kv_type;
        cp.type_v = (ggml_type) p.kv_type;
    }
    const char * pe = std::getenv("KITTEN_ASR_PROFILE");
    const bool profile = pe && *pe && std::strcmp(pe, "0") != 0;
    if (profile) {
        c->prof_dec.phase = "decoder";
        c->prof_dec.acc = &c->prof_acc;
        c->prof_enc.phase = "encoder";
        c->prof_enc.acc = &c->prof_acc;
        cp.cb_eval = profile_cb;
        cp.cb_eval_user_data = &c->prof_dec;
    }
    c->lctx = llama_init_from_model(c->model, cp);
    if (!c->lctx) {
        set_error("could not create the decoder context: " + g_error);
        kasr_free(c);
        return nullptr;
    }
    c->n_ctx = (int32_t) llama_n_ctx(c->lctx);

    // Persistent CPU thread pools. Without one, ggml creates and joins n_threads OS threads for every graph, which
    // is once per decoded token (ggml's own thread pool is used because this build has no OpenMP).
    const char * sce = std::getenv("KITTEN_ASR_STREAM_CACHE");
    c->stream_cache_on = !(sce && std::strcmp(sce, "0") == 0);
    const char * rse = std::getenv("KITTEN_ASR_RESYNC");
    c->resync_on = !(rse && std::strcmp(rse, "0") == 0);
    const char * ppe = std::getenv("KITTEN_ASR_POOL_PAUSE");
    c->pause_ok = !(ppe && std::strcmp(ppe, "0") == 0);
    const char * tpe = std::getenv("KITTEN_ASR_THREADPOOL");
    if (!(tpe && std::strcmp(tpe, "0") == 0) && !p.use_gpu) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t reg = cpu ? ggml_backend_dev_backend_reg(cpu) : nullptr;
        auto tp_new = reg ? (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new") : nullptr;
        c->tp_free = reg ? (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free") : nullptr;
        c->tp_pause = reg ? (decltype(ggml_threadpool_pause) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_pause") : nullptr;
        if (tp_new && c->tp_free) {   // without pause (older ggml), the pools still work; they just keep polling
            ggml_threadpool_params tpp = ggml_threadpool_params_default(p.n_threads);
            c->tp = tp_new(&tpp);
            if (p.n_threads_batch != p.n_threads) {
                ggml_threadpool_params tpb = ggml_threadpool_params_default(p.n_threads_batch);
                c->tp_batch = tp_new(&tpb);
            }
            if (c->tp) {
                llama_attach_threadpool(c->lctx, c->tp, c->tp_batch ? c->tp_batch : c->tp);
            }
        }
    }

    mtmd_context_params mp2 = mtmd_context_params_default();
    mp2.use_gpu = p.use_gpu != 0;
    mp2.print_timings = false;
    mp2.n_threads = p.n_threads_batch;
    mp2.flash_attn_type = (llama_flash_attn_type) p.flash_attn;
    mp2.warmup = p.warmup != 0;
    if (profile) {
        mp2.cb_eval = profile_cb;
        mp2.cb_eval_user_data = &c->prof_enc;
    }
    c->mctx = mtmd_init_from_file(encoder_gguf, c->model, mp2);
    if (!c->mctx) {
        set_error(std::string("could not load the audio encoder ") + encoder_gguf + ": " + g_error);
        kasr_free(c);
        return nullptr;
    }
    if (!mtmd_support_audio(c->mctx)) {
        set_error(std::string(encoder_gguf) + " has no audio encoder");
        kasr_free(c);
        return nullptr;
    }
    if (p.audio_window >= 0) {
        mtmd_kitten_set_audio_window(c->mctx, p.audio_window);
    }
    if (p.warmup) {
        std::vector<float> silence(16000, 0.0f);
        silence[100] = 1e-3f;
        kasr_result r;
        const char * head = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
        const char * tail = "<|im_end|>\n<|im_start|>assistant\n";
        if (!kasr_transcribe(c, silence.data(), (int32_t) silence.size(), head, tail, nullptr, 0, 4, &r)) {
            std::string why = g_error;
            kasr_free(c);
            set_error("warm-up transcription failed: " + why);
            return nullptr;
        }
    }
    return c;
}

void kasr_free(kasr_ctx * c) {
    if (!c) {
        return;
    }
    if (c->mctx) {
        mtmd_free(c->mctx);
    }
    if (c->lctx) {
        llama_free(c->lctx);
    }
    if (c->lctx_b) {
        llama_free(c->lctx_b);
    }
    if (c->tp_free) {
        if (c->tp) {
            c->tp_free(c->tp);
        }
        if (c->tp_batch) {
            c->tp_free(c->tp_batch);
        }
    }
    if (c->model) {
        llama_model_free(c->model);
    }
    delete c;
}

int32_t kasr_set_threads(kasr_ctx * c, int32_t n_threads, int32_t n_threads_batch) {
    if (!c || n_threads <= 0) {
        set_error("n_threads must be 1 or more");
        return 0;
    }
    if (n_threads_batch <= 0) {
        n_threads_batch = n_threads;
    }
    llama_set_n_threads(c->lctx, n_threads, n_threads_batch);
    if (c->lctx_b) {
        llama_set_n_threads(c->lctx_b, n_threads, n_threads_batch);
    }
    c->params.n_threads = n_threads;
    c->params.n_threads_batch = n_threads_batch;
    // mtmd's encoder threads are fixed at load (mtmd_context_params.n_threads)
    return 1;
}

// The decoder's pool threads poll for a while after each graph. Paused, they sleep instead, so they do not compete
// with the audio encoder's threads (mtmd runs its own) or with the caller between calls. The next llama_process()
// resumes them (ggml_graph_compute_kickoff).
// Draft resync (same rule as the GPU engine): at least this many draft tokens must remain after the meeting point,
// the meeting point is looked for this far into the rejected draft, and at most this many resyncs per call.
constexpr int32_t kResyncMin = 3;
constexpr int32_t kResyncWindow = 48;
constexpr int32_t kResyncMax = 6;

static void pause_pools(kasr_ctx * c) {
    if (c->tp_pause && c->pause_ok) {
        if (c->tp) {
            c->tp_pause(c->tp);
        }
        if (c->tp_batch) {
            c->tp_pause(c->tp_batch);
        }
    }
}

// With separate pools for one-token steps and for batches, the pool that is not about to run is paused, so its
// polling threads do not take cores from the one that runs. The running pool is never paused between its own
// graphs: its threads stay hot across decode steps.
static void pause_other_pool(kasr_ctx * c, bool batch) {
    if (c->tp_pause && c->pause_ok && c->tp && c->tp_batch) {
        c->tp_pause(batch ? c->tp : c->tp_batch);
    }
}

static bool decode_tokens(kasr_ctx * c, llama_batch_ext * batch, const int32_t * toks, int32_t n, int32_t pos0,
                          bool all_logits) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n; i++) {
        int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
        if (idx < 0) {
            set_error("decoder batch is full");
            return false;
        }
        llama_pos pos = pos0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
        if (all_logits || i == n - 1) {
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
    }
    c->prof_dec.armed = false;
    pause_other_pool(c, n > 1);
    int32_t ret = llama_process(c->lctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (ret != 0) {
        set_error("decoder step failed (" + std::to_string(ret) + ")");
        return false;
    }
    return true;
}

static int32_t transcribe_window(kasr_ctx * c, const float * pcm, int32_t n_samples, const char * head,
                                 const char * tail, const int32_t * draft, int32_t n_draft, int32_t max_new_tokens,
                                 int64_t stream_id, kasr_stream_result * sout, kasr_result * out);

// A failed call leaves no stream state behind: the kept prompt head and audio blocks may name decoder positions that
// were never written, and a retry with the same stream id would reuse them and give wrong text without an error.
static int32_t transcribe_impl(kasr_ctx * c, const float * pcm, int32_t n_samples, const char * head,
                               const char * tail, const int32_t * draft, int32_t n_draft, int32_t max_new_tokens,
                               int64_t stream_id, kasr_stream_result * sout, kasr_result * out) {
    int32_t ok = transcribe_window(c, pcm, n_samples, head, tail, draft, n_draft, max_new_tokens, stream_id, sout, out);
    if (!ok && c) {
        c->sc = kasr_ctx::stream_cache{};
        if (c->lctx) {
            llama_memory_clear(llama_get_memory(c->lctx), true);
        }
    }
    return ok;
}

static int32_t transcribe_window(kasr_ctx * c, const float * pcm, int32_t n_samples, const char * head,
                                 const char * tail, const int32_t * draft, int32_t n_draft, int32_t max_new_tokens,
                                 int64_t stream_id, kasr_stream_result * sout, kasr_result * out) {
    g_error.clear();
    if (!c || !out || !pcm || n_samples <= 0) {
        set_error("kasr_transcribe: empty audio or no context");
        return 0;
    }
    std::memset(out, 0, sizeof *out);
    if (sout) {
        std::memset(sout, 0, sizeof *sout);
    }
    c->tokens.clear();
    c->text.clear();
    if (!c->stream_cache_on) {
        stream_id = 0;
    }
    auto & sc = c->sc;
    if (stream_id == 0 || sc.id != stream_id) {
        sc = kasr_ctx::stream_cache{};          // another window (or none): nothing in the decoder can be reused
        sc.id = stream_id;
    }
    c->prof_dec.armed = c->prof_enc.armed = false;

    auto t0 = std::chrono::steady_clock::now();
    std::string prompt = std::string(head ? head : "") + mtmd_default_marker() + (tail ? tail : "");
    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio((size_t) n_samples, pcm);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    const mtmd_bitmap * bmps[1] = {bmp};
    int32_t rc = mtmd_tokenize(c->mctx, chunks, &txt, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        set_error("audio preprocessing failed (" + std::to_string(rc) + ")");
        return 0;
    }
    const size_t n_chunks = mtmd_input_chunks_size(chunks);

    // Which leading chunks this stream's kept decoder state already holds: the first text chunk if its tokens are
    // the same, then each complete audio block whose encoder input is bit-for-bit the same (the window only grew).
    size_t n_keep = 0;                          // leading chunks reused
    size_t n_keep_blocks = 0;
    llama_pos n_past = 0;
    const int win = mtmd_kitten_get_audio_window(c->mctx);
    if (stream_id != 0 && sc.head_pos > 0 && n_chunks > 0 &&
        mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, 0)) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t nt = 0;
        const llama_token * tk = mtmd_input_chunk_get_tokens_text(mtmd_input_chunks_get(chunks, 0), &nt);
        if (nt == sc.head.size() && std::equal(sc.head.begin(), sc.head.end(), tk)) {
            n_keep = 1;
            n_past = sc.head_pos;
            for (size_t j = 0; j < sc.mel.size() && 1 + j < n_chunks; j++) {
                const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, 1 + j);
                size_t nf = 0;
                int fr = 0;
                const float * m = mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO
                                ? mtmd_kitten_audio_chunk_mel(c->mctx, ch, &nf, &fr) : nullptr;
                if (!m || nf != sc.mel[j].size() || mtmd_input_chunk_get_n_pos(ch) != sc.n_pos[j] ||
                    std::memcmp(m, sc.mel[j].data(), nf * sizeof(float)) != 0) {
                    break;
                }
                n_keep++;
                n_keep_blocks++;
                n_past += sc.n_pos[j];
            }
        }
    }
    if (n_keep == 0) {
        llama_memory_clear(llama_get_memory(c->lctx), true);
        sc.head.clear();
        sc.head_pos = 0;
        sc.mel.clear();
        sc.n_pos.clear();
    } else {
        llama_memory_seq_rm(llama_get_memory(c->lctx), 0, n_past, -1);
        sc.mel.resize(n_keep_blocks);
        sc.n_pos.resize(n_keep_blocks);
    }
    if (sout) {
        sout->cached_positions = n_past;
        sout->cached_blocks = (int32_t) n_keep_blocks;
    }
    out->mel_ms = ms_since(t0);

    bool keep_more = stream_id != 0;            // blocks are kept only while they follow each other from the start
    for (size_t i = n_keep; i < n_chunks; i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        const bool last = i + 1 == n_chunks;
        llama_pos new_past = n_past;
        if (keep_more) {
            const auto type = mtmd_input_chunk_get_type(ch);
            if (i == 0 && type == MTMD_INPUT_CHUNK_TYPE_TEXT && !last) {
                size_t nt = 0;
                const llama_token * tk = mtmd_input_chunk_get_tokens_text(ch, &nt);
                sc.head.assign(tk, tk + nt);
                sc.head_pos = mtmd_input_chunk_get_n_pos(ch);
            } else if (i > 0 && type == MTMD_INPUT_CHUNK_TYPE_AUDIO && win > 0 && sc.head_pos > 0) {
                size_t nf = 0;
                int fr = 0;
                const float * m = mtmd_kitten_audio_chunk_mel(c->mctx, ch, &nf, &fr);
                if (m && fr == win) {           // a complete block: a longer window starts with the same one
                    sc.mel.emplace_back(m, m + nf);
                    sc.n_pos.push_back(mtmd_input_chunk_get_n_pos(ch));
                } else {
                    keep_more = false;
                }
            } else {
                keep_more = false;
            }
        }
        auto t = std::chrono::steady_clock::now();
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            c->prof_enc.armed = false;
            pause_pools(c);
            if (mtmd_encode_chunk(c->mctx, ch) != 0) {
                mtmd_input_chunks_free(chunks);
                set_error("audio encoder failed: " + g_error);
                return 0;
            }
            out->encode_ms += ms_since(t);
            out->n_audio_tokens += (int32_t) mtmd_input_chunk_get_n_tokens(ch);
            t = std::chrono::steady_clock::now();
            c->prof_dec.armed = false;
            pause_other_pool(c, true);
            rc = mtmd_helper_decode_image_chunk(c->mctx, c->lctx, ch, mtmd_get_output_embd(c->mctx), n_past, 0,
                                                c->params.n_batch, &new_past, nullptr, nullptr);
        } else {
            c->prof_dec.armed = false;
            pause_other_pool(c, true);
            rc = mtmd_helper_eval_chunk_single(c->mctx, c->lctx, ch, n_past, 0, c->params.n_batch, last, &new_past);
        }
        out->prefill_ms += ms_since(t);
        if (rc != 0) {
            mtmd_input_chunks_free(chunks);
            set_error("decoder failed on the prompt (" + std::to_string(rc) + ")");
            return 0;
        }
        n_past = new_past;
    }
    mtmd_input_chunks_free(chunks);
    out->n_prompt = n_past;

    int32_t cur = argmax(llama_get_logits_ith(c->lctx, -1), c->n_vocab);
    llama_batch_ext * batch = llama_batch_ext_init(c->lctx);
    int32_t finish = 1;
    const int32_t room = c->n_ctx - 1;

    // Rejected rest of the draft: decoding can meet it again further on (a changed first word or tag, a word that
    // came out differently), then the rest is checked in one batch instead of one step per token ("resync").
    std::vector<int32_t> rem;
    int32_t n_resync = 0;
    if (draft && n_draft > 0 && c->resync_on) {
        rem.assign(draft, draft + n_draft);     // replaced below when the draft's first token is accepted
    }
    // draft check: decode the draft as one batch, keep the prefix greedy decoding agrees with
    if (draft && n_draft > 0 && max_new_tokens > 0 && draft[0] == cur && !llama_vocab_is_eog(c->vocab, cur)) {
        auto t = std::chrono::steady_clock::now();
        int32_t k = std::min({n_draft, max_new_tokens, room - n_past});
        out->n_draft = n_draft;
        if (k > 0 && decode_tokens(c, batch, draft, k, n_past, true)) {
            int32_t acc = 0;
            for (int32_t j = 0; j < k; j++) {
                c->tokens.push_back(draft[j]);
                acc++;
                int32_t pred = argmax(llama_get_logits_ith(c->lctx, j), c->n_vocab);
                if (j + 1 < k && draft[j + 1] == pred && !llama_vocab_is_eog(c->vocab, pred)) {
                    continue;
                }
                cur = pred;
                break;
            }
            out->n_accepted = acc;
            n_past += acc;
            if (c->resync_on) {
                rem.assign(draft + acc, draft + n_draft);
            }
            llama_memory_seq_rm(llama_get_memory(c->lctx), 0, n_past, -1);
        } else if (k > 0) {
            llama_batch_ext_free(batch);
            return 0;
        }
        out->draft_ms = ms_since(t);
    }

    auto td = std::chrono::steady_clock::now();
    while (true) {
        if (llama_vocab_is_eog(c->vocab, cur)) {
            finish = 0;
            break;
        }
        if ((int32_t) c->tokens.size() >= max_new_tokens) {
            finish = 1;
            break;
        }
        if (n_past >= room) {
            finish = 2;
            break;
        }
        c->tokens.push_back(cur);
        if ((int32_t) c->tokens.size() >= max_new_tokens) {
            finish = 1;     // the last token needs no step
            break;
        }
        // resync: has the output met the rejected draft again? Either the token it was rejected at came out, or
        // the last two tokens appear a little further in it. Then feed this token and the draft's rest after the
        // meeting point as one batch and keep the part greedy decoding agrees with (every kept token is checked).
        int32_t j = -1;
        if ((int32_t) rem.size() > kResyncMin && n_resync < kResyncMax) {
            if (cur == rem[0]) {
                j = 0;
            } else if (c->tokens.size() >= 2) {
                const int32_t p = c->tokens[c->tokens.size() - 2];
                for (int32_t q = 1; q < std::min((int32_t) rem.size(), kResyncWindow); q++) {
                    if (rem[q] == cur && rem[q - 1] == p) {
                        j = q;
                        break;
                    }
                }
            }
            if (j >= 0 && (int32_t) rem.size() - j - 1 < kResyncMin) {
                j = -1;
            }
        }
        if (j >= 0) {
            const int32_t left = std::min(max_new_tokens - (int32_t) c->tokens.size(), room - n_past) - 1;
            std::vector<int32_t> feed(1, cur);
            feed.insert(feed.end(), rem.begin() + j + 1, rem.begin() + j + 1 + std::max(0, std::min(left, (int32_t) rem.size() - j - 1)));
            n_resync++;
            if (!decode_tokens(c, batch, feed.data(), (int32_t) feed.size(), n_past, true)) {
                llama_batch_ext_free(batch);
                return 0;
            }
            int32_t n = 0;           // candidates (feed[1..]) greedy decoding agrees with
            int32_t pred = argmax(llama_get_logits_ith(c->lctx, 0), c->n_vocab);
            while (n + 1 < (int32_t) feed.size() && pred == feed[n + 1] && !llama_vocab_is_eog(c->vocab, pred)) {
                c->tokens.push_back(feed[n + 1]);
                n++;
                pred = argmax(llama_get_logits_ith(c->lctx, n), c->n_vocab);
            }
            if (sout) {
                sout->resyncs++;
                sout->resync_tokens += (int32_t) feed.size() - 1;
                sout->resync_accepted += n;
            }
            n_past += 1 + n;
            llama_memory_seq_rm(llama_get_memory(c->lctx), 0, n_past, -1);
            rem.assign(feed.begin() + 1 + n, feed.end());
            cur = pred;
            continue;
        }
        if (!decode_tokens(c, batch, &cur, 1, n_past, false)) {
            llama_batch_ext_free(batch);
            return 0;
        }
        n_past++;
        out->n_steps++;
        cur = argmax(llama_get_logits_ith(c->lctx, -1), c->n_vocab);
    }
    out->decode_ms = ms_since(td);
    llama_batch_ext_free(batch);
    pause_pools(c);

    char buf[512];
    for (int32_t t : c->tokens) {
        int32_t n = llama_token_to_piece(c->vocab, t, buf, sizeof buf, 0, true);
        if (n > 0) {
            c->text.append(buf, (size_t) n);
        }
    }
    out->tokens = c->tokens.data();
    out->n_tokens = (int32_t) c->tokens.size();
    out->text = c->text.c_str();
    out->finish = finish;
    return 1;
}

int32_t kasr_transcribe(kasr_ctx * c, const float * pcm, int32_t n_samples, const char * head, const char * tail,
                        const int32_t * draft, int32_t n_draft, int32_t max_new_tokens, kasr_result * out) {
    return transcribe_impl(c, pcm, n_samples, head, tail, draft, n_draft, max_new_tokens, 0, nullptr, out);
}

int32_t kasr_transcribe_stream(kasr_ctx * c, const float * pcm, int32_t n_samples, const char * head, const char * tail,
                               const int32_t * draft, int32_t n_draft, int32_t max_new_tokens, int64_t stream_id,
                               kasr_stream_result * sout, kasr_result * out) {
    return transcribe_impl(c, pcm, n_samples, head, tail, draft, n_draft, max_new_tokens, stream_id, sout, out);
}

int32_t kasr_batch_yield(kasr_ctx * c, int32_t delta) {
    return c ? c->yield_req.fetch_add(delta) + delta : 0;
}

void kasr_stream_release(kasr_ctx * c, int64_t stream_id) {
    if (c && (stream_id == 0 || c->sc.id == stream_id)) {
        c->sc = kasr_ctx::stream_cache{};
    }
}

// ---------------------------------------------------------------- many windows at once (files and batches)
// One decode step of a single window reads every decoder weight to produce one token; the step is bound by memory
// bandwidth (and by the barrier that ends every op), not by arithmetic. kasr_transcribe_batch keeps up to n_slots
// windows in flight in a second decoding context with one KV stream per slot, and advances all of them by one token
// in the same step, so each weight read serves every window in flight. A finished window's slot is refilled at once
// (its mel, audio encoder and prompt run between two steps).

// Audio tokens per second of 16 kHz audio after the encoder (12.5 for these models), with room to spare: only used
// to size the per-slot context; a window whose prompt does not fit after all runs through kasr_transcribe instead.
static const double kBatchAudioTokPerS = 14.0;
static const int32_t kBatchTextTokens = 128;

static llama_context * make_batch_ctx(kasr_ctx * c, int32_t n_slots, int32_t ctx_seq) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((n_slots + 1) * ctx_seq);
    cp.n_batch = (uint32_t) c->params.n_batch;
    cp.n_ubatch = (uint32_t) c->params.n_batch;
    cp.n_seq_max = (uint32_t) (n_slots + 1);   // + the padding row's sequence (see kasr_transcribe_batch)
    cp.kv_unified = false;              // one KV stream per slot: attention only reads that window's cells
    cp.n_threads = c->params.n_threads;
    cp.n_threads_batch = c->params.n_threads_batch;
    cp.flash_attn_type = (llama_flash_attn_type) c->params.flash_attn;
    cp.no_perf = true;
    cp.op_offload = c->params.use_gpu != 0;
    if (c->params.kv_type > 0) {
        cp.type_k = (ggml_type) c->params.kv_type;
        cp.type_v = (ggml_type) c->params.kv_type;
    }
    llama_context * l = llama_init_from_model(c->model, cp);
    if (l && c->tp) {
        llama_attach_threadpool(l, c->tp, c->tp_batch ? c->tp_batch : c->tp);
    }
    return l;
}

struct batch_slot {
    int32_t win = -1;       // window in this slot, -1 when free
    llama_pos n_past = 0;
    int32_t cur = 0;        // next token: the argmax of the last logits
    int32_t idx = -1;       // its row in the current step's batch
};

// The stop rules of kasr_transcribe's loop, before one decode step: true when `cur` needs a step (it was appended),
// false when the window is finished (out->finish set).
static bool batch_needs_step(kasr_ctx * c, batch_slot & sl, int32_t max_new, int32_t room, kasr_result & r) {
    std::vector<int32_t> & toks = c->b_tokens[sl.win];
    if (llama_vocab_is_eog(c->vocab, sl.cur)) {
        r.finish = 0;
        return false;
    }
    if ((int32_t) toks.size() >= max_new) {
        r.finish = 1;
        return false;
    }
    if (sl.n_past >= room) {
        r.finish = 2;
        return false;
    }
    toks.push_back(sl.cur);
    if ((int32_t) toks.size() >= max_new) {
        r.finish = 1;
        return false;
    }
    return true;
}

// Mel, audio encoder and prompt of window w into slot s. Returns 1 when the window is in the slot (sl.cur is its first
// token), 0 when its prompt does not fit the slot (left for kasr_transcribe), -1 on error.
static int batch_start(kasr_ctx * c, batch_slot & sl, int32_t s, int32_t w, const float * pcm, int32_t n_samples,
                       const std::string & prompt, int32_t max_new, int32_t room, kasr_result & r) {
    auto t0 = std::chrono::steady_clock::now();
    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio((size_t) n_samples, pcm);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    const mtmd_bitmap * bmps[1] = {bmp};
    int32_t rc = mtmd_tokenize(c->mctx, chunks, &txt, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        set_error("audio preprocessing failed (" + std::to_string(rc) + ")");
        return -1;
    }
    r.mel_ms = ms_since(t0);
    if ((int32_t) mtmd_helper_get_n_pos(chunks) + max_new > room) {
        mtmd_input_chunks_free(chunks);
        return 0;
    }
    llama_memory_seq_rm(llama_get_memory(c->lctx_b), s, -1, -1);
    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        llama_pos new_past = n_past;
        auto t = std::chrono::steady_clock::now();
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            pause_pools(c);
            if (mtmd_encode_chunk(c->mctx, ch) != 0) {
                mtmd_input_chunks_free(chunks);
                set_error("audio encoder failed: " + g_error);
                return -1;
            }
            r.encode_ms += ms_since(t);
            r.n_audio_tokens += (int32_t) mtmd_input_chunk_get_n_tokens(ch);
            t = std::chrono::steady_clock::now();
            pause_other_pool(c, true);
            rc = mtmd_helper_decode_image_chunk(c->mctx, c->lctx_b, ch, mtmd_get_output_embd(c->mctx), n_past, s,
                                                c->params.n_batch, &new_past, nullptr, nullptr);
        } else {
            pause_other_pool(c, true);
            rc = mtmd_helper_eval_chunk_single(c->mctx, c->lctx_b, ch, n_past, s, c->params.n_batch,
                                               i + 1 == n_chunks, &new_past);
        }
        r.prefill_ms += ms_since(t);
        if (rc != 0) {
            mtmd_input_chunks_free(chunks);
            set_error("decoder failed on the prompt (" + std::to_string(rc) + ")");
            return -1;
        }
        n_past = new_past;
    }
    mtmd_input_chunks_free(chunks);
    r.n_prompt = n_past;
    sl.win = w;
    sl.n_past = n_past;
    sl.cur = argmax(llama_get_logits_ith(c->lctx_b, -1), c->n_vocab);
    return 1;
}

int32_t kasr_transcribe_batch(kasr_ctx * c, int32_t n, const float * const * pcm, const int32_t * n_samples,
                              const char * head, const char * tail, const int32_t * max_new_tokens, int32_t n_slots,
                              kasr_result * out) {
    g_error.clear();
    if (!c || n <= 0 || !pcm || !n_samples || !max_new_tokens || !out) {
        set_error("kasr_transcribe_batch: no windows or no context");
        return 0;
    }
    for (int32_t i = 0; i < n; i++) {
        if (!pcm[i] || n_samples[i] <= 0) {
            set_error("kasr_transcribe_batch: window " + std::to_string(i) + " has no audio");
            return 0;
        }
    }
    n_slots = std::max(1, std::min({n_slots, n, (int32_t) 255}));
    // the per-slot context: room for the longest window's prompt and token budget, rounded up to 256 cells
    int32_t need = 0;
    for (int32_t i = 0; i < n; i++) {
        need = std::max(need, (int32_t) (n_samples[i] / 16000.0 * kBatchAudioTokPerS) + kBatchTextTokens +
                                  std::max(0, max_new_tokens[i]) + 1);
    }
    // at least a 30 s window's (the longest KittenASR makes), so the context is made once, not per call
    need = std::max(need, (int32_t) (30 * kBatchAudioTokPerS) + kBatchTextTokens + 64 + 16 * 30 + 1);
    int32_t ctx_seq = std::min((need + 255) / 256 * 256, std::max(256, c->n_ctx));
    n_slots = std::max(n_slots, c->b_slots);    // a context made for more slots is kept (a slot costs only memory)
    if (!c->lctx_b || c->b_slots != n_slots || c->b_ctx_seq < ctx_seq) {
        if (c->lctx_b) {
            llama_free(c->lctx_b);
            c->lctx_b = nullptr;
        }
        c->lctx_b = make_batch_ctx(c, n_slots, ctx_seq);
        if (!c->lctx_b) {
            set_error("could not create the batch decoding context: " + g_error);
            return 0;
        }
        c->b_slots = n_slots;
        c->b_ctx_seq = ctx_seq;
    }
    ctx_seq = c->b_ctx_seq;
    const int32_t room = ctx_seq - 1;
    const llama_seq_id pad_seq = n_slots;       // the padding row's sequence
    std::memset(out, 0, sizeof(kasr_result) * (size_t) n);
    c->b_tokens.assign((size_t) n, std::vector<int32_t>());
    c->b_text.assign((size_t) n, std::string());
    llama_memory_clear(llama_get_memory(c->lctx_b), true);
    const std::string prompt = std::string(head ? head : "") + mtmd_default_marker() + (tail ? tail : "");

    std::vector<batch_slot> slots((size_t) n_slots);
    std::vector<int32_t> rest;          // windows whose prompt did not fit a slot
    llama_batch_ext * batch = llama_batch_ext_init(c->lctx_b);
    int32_t next = 0;
    bool ok = true;
    while (ok) {
        for (int32_t s = 0; s < n_slots && ok; s++) {     // fill free slots
            batch_slot & sl = slots[(size_t) s];
            // a waiting caller stops new windows, but each call starts at least one, so a file is never starved
            while (sl.win < 0 && next < n && (next == 0 || c->yield_req.load(std::memory_order_relaxed) <= 0)) {
                int32_t w = next++;
                int rc = batch_start(c, sl, s, w, pcm[w], n_samples[w], prompt, max_new_tokens[w], room, out[w]);
                if (rc < 0) {
                    ok = false;
                    break;
                }
                if (rc == 0) {
                    rest.push_back(w);
                } else if (!batch_needs_step(c, sl, max_new_tokens[w], room, out[w])) {
                    sl.win = -1;
                }
            }
        }
        if (!ok) {
            break;
        }
        llama_batch_ext_clear(batch);
        int32_t n_active = 0;
        for (int32_t s = 0; s < n_slots; s++) {
            batch_slot & sl = slots[(size_t) s];
            if (sl.win < 0) {
                continue;
            }
            sl.idx = llama_batch_ext_add_token(batch, s, sl.cur);
            if (sl.idx < 0) {
                set_error("decoder batch is full");
                ok = false;
                break;
            }
            llama_pos pos = sl.n_past;
            llama_batch_ext_set_pos(batch, sl.idx, &pos);
            llama_batch_ext_set_output_logits(batch, sl.idx, true);
            n_active++;
        }
        if (!ok || n_active == 0) {
            break;
        }
        // A step is never one row: with one row ggml takes its single-token kernels (matrix-vector products and,
        // from 512 cached positions, attention split across threads), whose summation order differs from the
        // batched ones. A throwaway row of a padding sequence keeps every step on the batched kernels, so a window's
        // text does not depend on which other windows share its steps (or on being alone at the end of a list).
        if (n_active == 1) {
            llama_memory_seq_rm(llama_get_memory(c->lctx_b), pad_seq, -1, -1);
            int32_t pi = llama_batch_ext_add_token(batch, pad_seq, 0);
            llama_pos p0 = 0;
            if (pi < 0 || !llama_batch_ext_set_pos(batch, pi, &p0)) {
                set_error("decoder batch is full");
                ok = false;
                break;
            }
        }
        auto t = std::chrono::steady_clock::now();
        pause_other_pool(c, true);   // every step here runs on the batch pool (a lone window has its padding row)
        int32_t ret = llama_process(c->lctx_b, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (ret != 0) {
            set_error("decoder step failed (" + std::to_string(ret) + ")");
            ok = false;
            break;
        }
        const double share = ms_since(t) / n_active;
        for (int32_t s = 0; s < n_slots; s++) {
            batch_slot & sl = slots[(size_t) s];
            if (sl.win < 0) {
                continue;
            }
            kasr_result & r = out[sl.win];
            r.decode_ms += share;
            r.n_steps++;
            sl.n_past++;
            sl.cur = argmax(llama_get_logits_ith(c->lctx_b, sl.idx), c->n_vocab);
            if (!batch_needs_step(c, sl, max_new_tokens[sl.win], room, r)) {
                sl.win = -1;
            }
        }
    }
    llama_batch_ext_free(batch);
    pause_pools(c);
    if (!ok) {
        return 0;
    }
    for (int32_t w = next; w < n; w++) {  // another caller asked for a turn: these windows were not started
        out[w].finish = -1;
    }
    for (int32_t w : rest) {            // one at a time, exactly as kasr_transcribe
        if (!kasr_transcribe(c, pcm[w], n_samples[w], head, tail, nullptr, 0, max_new_tokens[w], &out[w])) {
            return 0;
        }
        c->b_tokens[(size_t) w] = c->tokens;
    }
    char buf[512];
    for (int32_t w = 0; w < n; w++) {
        std::string & txt = c->b_text[(size_t) w];
        txt.clear();
        for (int32_t t : c->b_tokens[(size_t) w]) {
            int32_t k = llama_token_to_piece(c->vocab, t, buf, sizeof buf, 0, true);
            if (k > 0) {
                txt.append(buf, (size_t) k);
            }
        }
        out[w].tokens = c->b_tokens[(size_t) w].data();
        out[w].n_tokens = (int32_t) c->b_tokens[(size_t) w].size();
        out[w].text = txt.c_str();
    }
    return 1;
}

const char * kasr_detokenize(kasr_ctx * c, const int32_t * tokens, int32_t n_tokens) {
    if (!c) {
        return "";
    }
    c->detok.clear();
    char buf[512];
    for (int32_t i = 0; i < n_tokens; i++) {
        int32_t n = llama_token_to_piece(c->vocab, tokens[i], buf, sizeof buf, 0, true);
        if (n > 0) {
            c->detok.append(buf, (size_t) n);
        }
    }
    return c->detok.c_str();
}

const char * kasr_info(kasr_ctx * c) {
    static thread_local std::string s;
    std::string o = "{\"api\": " + std::to_string(KASR_API_VERSION);
    o += ", \"system\": \"" + json_escape(llama_print_system_info()) + "\"";
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        o += ", \"backend_log\": \"" + json_escape(g_backend_log) + "\"";
    }
    o += ", \"devices\": [";
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        o += std::string(i ? ", " : "") + "{\"name\": \"" + json_escape(ggml_backend_dev_name(d)) +
             "\", \"description\": \"" + json_escape(ggml_backend_dev_description(d)) + "\"}";
    }
    o += "], \"cpu_features\": {";
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
        auto get = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        bool first = true;
        for (ggml_backend_feature * f = get ? get(reg) : nullptr; f && f->name; f++) {
            o += std::string(first ? "" : ", ") + "\"" + json_escape(f->name) + "\": \"" + json_escape(f->value) + "\"";
            first = false;
        }
    }
    o += "}";
    if (c) {
        // llama_model_desc() is "<arch> <size class> <type>"; the size class is llama.cpp's guess from the layer
        // count (not this model's size), so only the architecture and the weight type are reported
        char desc[256] = {0};
        llama_model_desc(c->model, desc, sizeof desc);
        std::string d = desc;
        size_t a = d.find(' '), b = d.rfind(' ');
        o += ", \"decoder_arch\": \"" + json_escape(d.substr(0, a)) + "\"";
        o += ", \"decoder_type\": \"" + json_escape(b == std::string::npos ? d : d.substr(b + 1)) + "\"";
        o += ", \"decoder_params\": " + std::to_string(llama_model_n_params(c->model));
        o += ", \"decoder_bytes\": " + std::to_string(llama_model_size(c->model));
        o += ", \"n_threads\": " + std::to_string(llama_n_threads(c->lctx));
        o += ", \"n_threads_batch\": " + std::to_string(llama_n_threads_batch(c->lctx));
        o += ", \"encoder_threads\": " + std::to_string(c->params.n_threads_batch);
        o += ", \"n_ctx\": " + std::to_string(c->n_ctx);
        o += ", \"n_batch\": " + std::to_string(c->params.n_batch);
        o += ", \"flash_attn\": " + std::to_string(c->params.flash_attn);
        o += ", \"kv_type\": " + std::to_string(c->params.kv_type);
        o += ", \"repack\": " + std::to_string(c->params.repack);
        o += ", \"threadpool\": " + std::to_string(c->tp ? 1 : 0);
        o += ", \"use_gpu\": " + std::to_string(c->params.use_gpu);
        o += ", \"audio_window\": " + std::to_string(mtmd_kitten_get_audio_window(c->mctx));
        if (!c->prof_acc.empty()) {
            o += ", \"op_profile\": {";
            bool first = true;
            for (auto & kv : c->prof_acc) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "[%.3f, %ld]", kv.second.first, kv.second.second);
                o += std::string(first ? "" : ", ") + "\"" + json_escape(kv.first) + "\": " + buf;
                first = false;
            }
            o += "}";
        }
    }
    o += "}";
    s = o;
    return s.c_str();
}

int32_t kasr_quantize(const char * in_gguf, const char * out_gguf, const char * type, int32_t n_threads) {
    g_error.clear();
    if (!kasr_backend_init(nullptr, nullptr)) {
        return 0;
    }
    struct { const char * name; llama_ftype ftype; } kinds[] = {
        // only the precisions that kept the accuracy in our tests (lower ones are not offered)
        {"q8_0", LLAMA_FTYPE_MOSTLY_Q8_0}, {"f16", LLAMA_FTYPE_MOSTLY_F16}, {"bf16", LLAMA_FTYPE_MOSTLY_BF16},
    };
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    bool found = false;
    for (auto & k : kinds) {
        if (type && std::strcmp(type, k.name) == 0) {
            qp.ftype = k.ftype;
            found = true;
        }
    }
    if (!found) {
        set_error(std::string("unknown quantisation type ") + (type ? type : "(null)"));
        return 0;
    }
    qp.nthread = n_threads;
    if (llama_model_quantize(in_gguf, out_gguf, &qp) != 0) {
        set_error(std::string("quantising ") + in_gguf + " failed: " + g_error);
        return 0;
    }
    return 1;
}

}  // extern "C"
