/* Kitten ASR CPU engine: a small C API over llama.cpp's libllama (text decoder) and libmtmd (mel front end,
 * audio encoder and projector), loaded by kitten_inference.asr._cpu with ctypes.
 *
 * One kasr_ctx holds one model (two GGUF files: the decoder and the audio encoder) and one decoding context.
 * Calls on one kasr_ctx must not overlap; different contexts may run on different threads.
 * Every function that can fail returns 0 / NULL on failure and kasr_last_error() says why (per thread).
 */
#ifndef KASR_H
#define KASR_H

#include <stdint.h>

#ifdef _WIN32
#  define KASR_API __declspec(dllexport)
#else
#  define KASR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KASR_API_VERSION 1

typedef struct kasr_ctx kasr_ctx;

typedef struct kasr_params {
    int32_t n_threads;        /* threads for one-token decode steps (<= 0: all physical cores) */
    int32_t n_threads_batch;  /* threads for the prompt, the audio encoder and draft checks (<= 0: n_threads) */
    int32_t n_ctx;            /* decoder context in tokens (<= 0: 2048) */
    int32_t n_batch;          /* tokens per prompt batch (<= 0: 512) */
    int32_t flash_attn;       /* -1 auto, 0 off, 1 on */
    int32_t audio_window;     /* mel frames per encoder pass: -1 the model file's value, 0 whole clip, 800 = 8 s */
    int32_t use_gpu;          /* 1: offload to a GPU backend when one is loaded (Metal), 0: CPU only */
    int32_t kv_type;          /* ggml type id of the key/value cache (<= 0: f16) */
    int32_t warmup;           /* 1: run one short decode at load */
    int32_t repack;           /* 1: let ggml repack weights for this CPU (default), 0: keep the file layout */
} kasr_params;

typedef struct kasr_result {
    const int32_t * tokens;   /* generated token ids (owned by the context until the next call) */
    int32_t n_tokens;
    const char * text;        /* the tokens as UTF-8 text, special tokens written out */
    int32_t n_prompt;         /* decoder positions before generation (text + audio tokens) */
    int32_t n_audio_tokens;
    int32_t n_draft;          /* draft tokens offered */
    int32_t n_accepted;       /* draft tokens greedy decoding agreed with */
    int32_t finish;           /* 0 stop token, 1 max_new_tokens reached, 2 context full, -1 not started (batch) */
    double mel_ms;            /* log-mel features and prompt tokenisation */
    double encode_ms;         /* audio encoder + projector */
    double prefill_ms;        /* decoder over the prompt (text + audio embeddings) */
    double draft_ms;          /* decoder over the draft (one batch) */
    double decode_ms;         /* one-token decode steps */
    int32_t n_steps;          /* one-token decode steps */
} kasr_result;

KASR_API int32_t      kasr_api_version(void);
KASR_API int32_t      kasr_has_packed_weights(void);
KASR_API const char * kasr_last_error(void);

/* Loads the ggml backends once per process. backend_dir: folder of the libggml-cpu-*.so variants (NULL: next to
 * this library). variant: load only that CPU variant (for example "haswell"), NULL: the best for this CPU.
 * Returns 1 on success. Called by kasr_load when not called before. */
KASR_API int32_t      kasr_backend_init(const char * backend_dir, const char * variant);

/* A JSON object describing the loaded backends, CPU features and (with a context) the model and settings. */
KASR_API const char * kasr_info(kasr_ctx * ctx);

KASR_API kasr_params  kasr_default_params(void);
KASR_API kasr_ctx *   kasr_load(const char * decoder_gguf, const char * encoder_gguf, const kasr_params * params);
KASR_API void         kasr_free(kasr_ctx * ctx);
KASR_API int32_t      kasr_set_threads(kasr_ctx * ctx, int32_t n_threads, int32_t n_threads_batch);

/* Transcribes one window of 16 kHz mono float audio with greedy decoding.
 * The prompt is head + <audio> + tail, where <audio> is the encoded audio between the model's audio start and end
 * tokens; head and tail are chat text with special tokens written out.
 * draft: token ids of an earlier result for the same window (may be NULL). The longest prefix greedy decoding
 * agrees with is kept after one batched check, and decoding goes on from the first disagreement.
 * Returns 1 on success. */
KASR_API int32_t      kasr_transcribe(kasr_ctx * ctx, const float * pcm, int32_t n_samples,
                                      const char * head, const char * tail,
                                      const int32_t * draft, int32_t n_draft, int32_t max_new_tokens,
                                      kasr_result * out);

/* Transcribes n windows of 16 kHz mono float audio with greedy decoding, decoding up to n_slots of them together:
 * one decode step advances every window in flight by one token, so the decoder's weights are read once per step
 * for all of them. A window enters a slot as soon as one is free. Same prompt (head + <audio> + tail) for every
 * window; max_new_tokens per window. out[i] is window i's result (its tokens and text are owned by the context
 * until the next call; draft fields are 0; decode_ms is the window's share of the steps it was in).
 * The text is kasr_transcribe's up to floating-point summation order (batched matrix products and attention).
 * Returns 1 on success. Available when the library exports it (the API version is unchanged). */
KASR_API int32_t      kasr_transcribe_batch(kasr_ctx * ctx, int32_t n, const float * const * pcm,
                                            const int32_t * n_samples, const char * head, const char * tail,
                                            const int32_t * max_new_tokens, int32_t n_slots, kasr_result * out);
/* Turns between callers of one context (the caller serialises its calls). A caller that wants the context while a
 * kasr_transcribe_batch call runs adds 1 here (from any thread) and subtracts 1 once it has its turn. While the count
 * is above 0, kasr_transcribe_batch starts no new window after its first: it finishes the windows in flight and returns 1, and every
 * window it did not start has finish == -1 (no tokens) and is to be sent again. Returns the new count (delta 0: read
 * it). Available when the library exports it. */
KASR_API int32_t      kasr_batch_yield(kasr_ctx * ctx, int32_t delta);
/* Live audio: the decoder state kept for one stream between calls. */
typedef struct kasr_stream_result {
    int32_t cached_positions; /* decoder positions reused from the previous call of this stream (0: none) */
    int32_t cached_blocks;    /* complete audio encoder blocks among them */
    int32_t resyncs;          /* batched checks of a rejected draft's rest after decoding met it again */
    int32_t resync_tokens;    /* draft tokens offered in them */
    int32_t resync_accepted;  /* of which greedy decoding agreed with */
} kasr_stream_result;

/* kasr_transcribe for a window that grows between calls (live audio). stream_id != 0 names the window: the decoder
 * state of the prompt head and of every complete audio encoder block is kept, and the next call with the same id
 * reuses the leading part whose tokens and encoder input are bit-identical, so it computes exactly what
 * kasr_transcribe would (the reused rows are the rows a fresh pass computes). A call with another id, or
 * kasr_transcribe, drops what was kept. Models whose encoder sees the whole clip at once reuse only the prompt head.
 * sout (may be NULL) says how much was reused. Returns 1 on success. */
KASR_API int32_t      kasr_transcribe_stream(kasr_ctx * ctx, const float * pcm, int32_t n_samples,
                                             const char * head, const char * tail,
                                             const int32_t * draft, int32_t n_draft, int32_t max_new_tokens,
                                             int64_t stream_id, kasr_stream_result * sout, kasr_result * out);

/* Drops what is kept for stream_id (0: for any stream). */
KASR_API void         kasr_stream_release(kasr_ctx * ctx, int64_t stream_id);

/* Token ids -> text (special tokens written out). Returns the text, owned by the context until the next call. */
KASR_API const char * kasr_detokenize(kasr_ctx * ctx, const int32_t * tokens, int32_t n_tokens);

/* Converts a decoder GGUF (bf16/f16/f32) to `type`: "q8_0", "f16" or "bf16".
 * Returns 1 on success. */
KASR_API int32_t      kasr_quantize(const char * in_gguf, const char * out_gguf, const char * type, int32_t n_threads);

#ifdef __cplusplus
}
#endif
#endif
