// Part of kitten-inference (Apache-2.0).
// The English fallback G2P of the Kitten TTS 2 models: a character-level BART (g2p/g2p_bart_en_us in each model
// repository), greedy decoding, the same computation as frontend/bart_numpy.py. Weights are read once from
// model.safetensors (F32 or F16/BF16 widened) and kept in float. The arithmetic is float32 by default (set_fp32;
// measured to pick the same phonemes as the float64 numpy reference) or double. Decoding keeps a key/value cache per
// decoder layer (the reference recomputes the whole prefix each step; every row is computed the same way).
// Scratch buffers are allocated once per model and reused: no allocation inside the decode loop.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kt_tinyen {

class Bart {
public:
    // model_dir holds config.json and model.safetensors. Returns false with a message in err.
    bool load(const std::string& model_dir, std::string& err);
    // Greedy decoding; returns the decoder ids including the start token (as transformers' generate).
    void generate(const std::vector<int>& input_ids, int max_length, std::vector<int>& out);
    int max_len() const { return max_len_; }
    // true (default): float32 arithmetic (NEON fused multiply-add on arm64), about 2x faster than double; on 3000
    // sampled CMUdict words and every BART word of the parity sets it picks exactly the numpy reference's phonemes.
    // false: double arithmetic, as the numpy reference (a near-tie could in principle resolve differently in float32).
    void set_fp32(bool on) { fp32_ = on; }
    size_t weight_bytes() const { return store_.size() * sizeof(float); }

private:
    struct Lin { const float* w = nullptr; const float* b = nullptr; int in = 0, out = 0; };
    struct LN { const float* w = nullptr; const float* b = nullptr; };
    struct Attn { Lin q, k, v, o; };
    struct EncL { Attn sa; LN sa_ln; Lin fc1, fc2; LN fin_ln; };
    struct DecL { Attn sa; LN sa_ln; Attn ca; LN ca_ln; Lin fc1, fc2; LN fin_ln; };

    const float* tensor(const std::string& name, size_t numel, std::string& err);
    bool lin(const std::string& p, int in, int out, Lin& l, std::string& err);
    bool ln(const std::string& p, LN& l, std::string& err);

    template <class T> struct Buf {
        std::vector<T> enc_x, enc_k, enc_v, q, att, ffn_, scores, x, tmp, ck, cv, kc, vc;
        void size(int n_dec, int max_len, int d, int ffn, int vocab);
    };
    template <class T> void linear(const Lin& l, const T* x, T* y) const;
    template <class T> void layernorm(const LN& l, T* x) const;
    template <class T> void embed(const float* pos_table, const LN& ln_emb, int id, int pos, T* x) const;
    template <class T> void attention(int heads, const T* q, const T* K, const T* V, int S, T* out, T* scores) const;
    template <class T> void run(const std::vector<int>& ids, int max_length, std::vector<int>& out, Buf<T>& B);

    std::vector<float> store_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> index_;  // name -> (offset, numel) in store_
    int d_ = 0, heads_e_ = 0, heads_d_ = 0, n_enc_ = 0, n_dec_ = 0, ffn_e_ = 0, ffn_d_ = 0, vocab_ = 0;
    int max_len_ = 0, start_ = 2, eos_ = 2, forced_eos_ = -1;
    double scale_emb_ = 1.0;
    const float* emb_ = nullptr;
    const float* final_bias_ = nullptr;
    const float* pos_enc_ = nullptr;
    const float* pos_dec_ = nullptr;
    LN ln_emb_enc_, ln_emb_dec_;
    std::vector<EncL> enc_;
    std::vector<DecL> dec_;
    bool fp32_ = true;
    Buf<double> bd_;
    Buf<float> bf_;
};

}  // namespace kt_tinyen
