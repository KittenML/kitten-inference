// Part of kitten-inference (Apache-2.0). See bart.hpp.
#include "bart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "json_lite.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace kt_tinyen {
namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int cfg_int(const json::Value& c, const char* k, int def) {
    const json::Value* v = c.get(k);
    return (v && v->kind == json::Value::Num) ? (int)v->num : def;
}

}  // namespace

bool Bart::load(const std::string& dir, std::string& err) {
    std::string cfg_text, st;
    if (!read_file(dir + "/config.json", cfg_text)) { err = "cannot read " + dir + "/config.json"; return false; }
    json::Value cfg;
    if (!json::parse(cfg_text, cfg) || cfg.kind != json::Value::Obj) { err = "bad BART config.json"; return false; }
    const json::Value* mt = cfg.get("model_type");
    if (!mt || mt->str != "bart" || !cfg.get("encoder_layers")) { err = "not a BART config"; return false; }
    const json::Value* act = cfg.get("activation_function");
    if (act && act->kind == json::Value::Str && act->str != "gelu") { err = "only the erf GELU BART is supported"; return false; }
    d_ = cfg_int(cfg, "d_model", 0);
    heads_e_ = cfg_int(cfg, "encoder_attention_heads", 0);
    heads_d_ = cfg_int(cfg, "decoder_attention_heads", 0);
    n_enc_ = cfg_int(cfg, "encoder_layers", 0);
    n_dec_ = cfg_int(cfg, "decoder_layers", 0);
    ffn_e_ = cfg_int(cfg, "encoder_ffn_dim", 0);
    ffn_d_ = cfg_int(cfg, "decoder_ffn_dim", 0);
    vocab_ = cfg_int(cfg, "vocab_size", 0);
    max_len_ = cfg_int(cfg, "max_position_embeddings", 0);
    start_ = cfg_int(cfg, "decoder_start_token_id", 2);
    eos_ = cfg_int(cfg, "eos_token_id", 2);
    const json::Value* fe = cfg.get("forced_eos_token_id");
    forced_eos_ = (fe && fe->kind == json::Value::Num) ? (int)fe->num : -1;
    const json::Value* se = cfg.get("scale_embedding");
    scale_emb_ = (se && se->kind == json::Value::Bool && se->b) ? std::sqrt((double)d_) : 1.0;
    if (d_ <= 0 || d_ > 4096 || heads_e_ <= 0 || heads_d_ <= 0 || d_ % heads_e_ || d_ % heads_d_ || n_enc_ <= 0 ||
        n_dec_ <= 0 || n_enc_ > 64 || n_dec_ > 64 || ffn_e_ <= 0 || ffn_d_ <= 0 || ffn_e_ > 65536 || ffn_d_ > 65536 ||
        vocab_ <= 0 || vocab_ > 1000000 || max_len_ < 3 || max_len_ > 4096) {
        err = "BART config values out of range";
        return false;
    }

    if (!read_file(dir + "/model.safetensors", st)) { err = "cannot read " + dir + "/model.safetensors"; return false; }
    if (st.size() < 8) { err = "model.safetensors too short"; return false; }
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; --i) hlen = (hlen << 8) | (unsigned char)st[i];
    if (hlen > st.size() - 8 || hlen > (1u << 26)) { err = "bad safetensors header"; return false; }
    json::Value hdr;
    if (!json::parse(st.substr(8, hlen), hdr) || hdr.kind != json::Value::Obj) { err = "bad safetensors header"; return false; }
    const size_t data0 = 8 + hlen, data_size = st.size() - data0;
    size_t total = 0;
    struct Item { std::string name, dtype; size_t a, b, numel; };
    std::vector<Item> items;
    for (auto& kv : hdr.obj) {
        if (kv.first == "__metadata__") continue;
        const json::Value* dt = kv.second.get("dtype");
        const json::Value* sh = kv.second.get("shape");
        const json::Value* off = kv.second.get("data_offsets");
        if (!dt || !sh || !off || off->arr.size() != 2 || sh->kind != json::Value::Arr) { err = "bad tensor entry"; return false; }
        size_t numel = 1;
        for (auto& s : sh->arr) {
            if (s.num < 0 || s.num > 1e8) { err = "bad tensor shape"; return false; }
            numel *= (size_t)s.num;
            if (numel > (1u << 28)) { err = "tensor too large"; return false; }
        }
        double a = off->arr[0].num, b = off->arr[1].num;
        if (a < 0 || b < a || b > (double)data_size) { err = "tensor offsets out of range"; return false; }
        size_t es = dt->str == "F32" ? 4 : (dt->str == "F16" || dt->str == "BF16") ? 2 : 0;
        if (es == 0) continue;  // other dtypes are not used by the model
        if ((size_t)(b - a) != numel * es) { err = "tensor size mismatch: " + kv.first; return false; }
        items.push_back({kv.first, dt->str, (size_t)a, (size_t)b, numel});
        total += numel;
    }
    store_.assign(total, 0.f);
    size_t off = 0;
    for (auto& it : items) {
        const char* src = st.data() + data0 + it.a;
        float* dst = store_.data() + off;
        if (it.dtype == "F32") std::memcpy(dst, src, it.numel * 4);
        else if (it.dtype == "F16") for (size_t i = 0; i < it.numel; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); dst[i] = half_to_float(h); }
        else for (size_t i = 0; i < it.numel; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); uint32_t u = (uint32_t)h << 16; std::memcpy(&dst[i], &u, 4); }
        index_[it.name] = {off, it.numel};
        off += it.numel;
    }

    emb_ = tensor("model.shared.weight", (size_t)vocab_ * d_, err);
    if (!emb_) return false;
    if (index_.count("final_logits_bias")) {
        final_bias_ = tensor("final_logits_bias", (size_t)vocab_, err);
        if (!final_bias_) return false;
    }
    pos_enc_ = tensor("model.encoder.embed_positions.weight", (size_t)(max_len_ + 2) * d_, err);
    pos_dec_ = tensor("model.decoder.embed_positions.weight", (size_t)(max_len_ + 2) * d_, err);
    if (!pos_enc_ || !pos_dec_) return false;
    if (!ln("model.encoder.layernorm_embedding", ln_emb_enc_, err) || !ln("model.decoder.layernorm_embedding", ln_emb_dec_, err)) return false;
    auto attn = [&](const std::string& p, Attn& a) {
        return lin(p + ".q_proj", d_, d_, a.q, err) && lin(p + ".k_proj", d_, d_, a.k, err) &&
               lin(p + ".v_proj", d_, d_, a.v, err) && lin(p + ".out_proj", d_, d_, a.o, err);
    };
    enc_.resize(n_enc_);
    for (int i = 0; i < n_enc_; ++i) {
        std::string p = "model.encoder.layers." + std::to_string(i);
        EncL& L = enc_[i];
        if (!attn(p + ".self_attn", L.sa) || !ln(p + ".self_attn_layer_norm", L.sa_ln, err) ||
            !lin(p + ".fc1", d_, ffn_e_, L.fc1, err) || !lin(p + ".fc2", ffn_e_, d_, L.fc2, err) ||
            !ln(p + ".final_layer_norm", L.fin_ln, err)) return false;
    }
    dec_.resize(n_dec_);
    for (int i = 0; i < n_dec_; ++i) {
        std::string p = "model.decoder.layers." + std::to_string(i);
        DecL& L = dec_[i];
        if (!attn(p + ".self_attn", L.sa) || !ln(p + ".self_attn_layer_norm", L.sa_ln, err) ||
            !attn(p + ".encoder_attn", L.ca) || !ln(p + ".encoder_attn_layer_norm", L.ca_ln, err) ||
            !lin(p + ".fc1", d_, ffn_d_, L.fc1, err) || !lin(p + ".fc2", ffn_d_, d_, L.fc2, err) ||
            !ln(p + ".final_layer_norm", L.fin_ln, err)) return false;
    }
    const int F = std::max(ffn_e_, ffn_d_);
    bd_.size(n_dec_, max_len_, d_, F, vocab_);
    bf_.size(n_dec_, max_len_, d_, F, vocab_);
    return true;
}

const float* Bart::tensor(const std::string& name, size_t numel, std::string& err) {
    auto it = index_.find(name);
    if (it == index_.end()) { err = "BART weights lack " + name; return nullptr; }
    if (it->second.second != numel) { err = "BART tensor " + name + " has the wrong size"; return nullptr; }
    return store_.data() + it->second.first;
}

bool Bart::lin(const std::string& p, int in, int out, Lin& l, std::string& err) {
    l.w = tensor(p + ".weight", (size_t)in * out, err);
    l.b = l.w ? tensor(p + ".bias", (size_t)out, err) : nullptr;
    l.in = in; l.out = out;
    return l.w && l.b;
}

bool Bart::ln(const std::string& p, LN& l, std::string& err) {
    l.w = tensor(p + ".weight", (size_t)d_, err);
    l.b = l.w ? tensor(p + ".bias", (size_t)d_, err) : nullptr;
    return l.w && l.b;
}

// Dot product of a float row with a double vector, in double, with four independent partial sums: no loop-carried
// dependency on one add, so in-order cores (Cortex-A53/A55) keep their FP pipeline busy. The summation order differs
// from numpy's by rounding only (double precision); the argmax picks are unchanged on the parity sets.
static inline double dot(const float* w, const double* x, int n) {
    int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    // The same four partial sums as the portable loop below, two per 128-bit register (lanes {0,1} and {2,3}), with
    // a separate multiply and add (no fused multiply-add): bit-identical to the portable loop, which GCC vectorises
    // this way by itself and Clang (the Android NDK and Xcode compilers) does not.
    float64x2_t a01 = vdupq_n_f64(0.0), a23 = vdupq_n_f64(0.0);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t wf = vld1q_f32(w + i);
        a01 = vaddq_f64(a01, vmulq_f64(vld1q_f64(x + i), vcvt_f64_f32(vget_low_f32(wf))));
        a23 = vaddq_f64(a23, vmulq_f64(vld1q_f64(x + i + 2), vcvt_high_f64_f32(wf)));
    }
    double s0 = vgetq_lane_f64(a01, 0), s1 = vgetq_lane_f64(a01, 1), s2 = vgetq_lane_f64(a23, 0),
           s3 = vgetq_lane_f64(a23, 1);
#else
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#endif
    for (; i + 4 <= n; i += 4) {
        s0 += x[i] * (double)w[i];
        s1 += x[i + 1] * (double)w[i + 1];
        s2 += x[i + 2] * (double)w[i + 2];
        s3 += x[i + 3] * (double)w[i + 3];
    }
    for (; i < n; ++i) s0 += x[i] * (double)w[i];
    return (s0 + s1) + (s2 + s3);
}

// float32 variant (Bart::set_fp32): 16 products in flight, fused multiply-add on NEON.
static inline float dot(const float* w, const float* x, int n) {
    int i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(x + i), vld1q_f32(w + i));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vld1q_f32(w + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vld1q_f32(w + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vld1q_f32(w + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i + 4 <= n; i += 4) { s0 += x[i] * w[i]; s1 += x[i + 1] * w[i + 1]; s2 += x[i + 2] * w[i + 2]; s3 += x[i + 3] * w[i + 3]; }
    float s = (s0 + s1) + (s2 + s3);
#endif
    for (; i < n; ++i) s += x[i] * w[i];
    return s;
}

template <class T> static inline T erf_t(T x) { return std::erf(x); }
template <class T> static inline T exp_t(T x) { return std::exp(x); }
template <class T> static inline T gelu(T x) { return T(0.5) * x * (T(1) + erf_t<T>(x / std::sqrt(T(2)))); }

template <class T>
void Bart::linear(const Lin& l, const T* x, T* y) const {
    for (int o = 0; o < l.out; ++o) y[o] = dot(l.w + (size_t)o * l.in, x, l.in) + (T)l.b[o];
}

template <class T>
void Bart::layernorm(const LN& l, T* x) const {
    T m = 0;
    for (int i = 0; i < d_; ++i) m += x[i];
    m /= d_;
    T v = 0;
    for (int i = 0; i < d_; ++i) { T t = x[i] - m; v += t * t; }
    v /= d_;
    T inv = T(1) / std::sqrt(v + T(1e-5));
    for (int i = 0; i < d_; ++i) x[i] = (x[i] - m) * inv * (T)l.w[i] + (T)l.b[i];
}

template <class T>
void Bart::embed(const float* pos, const LN& lne, int id, int p, T* x) const {
    const float* e = emb_ + (size_t)id * d_;
    const float* q = pos + (size_t)(p + 2) * d_;
    for (int i = 0; i < d_; ++i) x[i] = (T)e[i] * (T)scale_emb_ + (T)q[i];
    layernorm(lne, x);
}

// out = softmax(q . k[0..S) per head) v for one query row; q already scaled.
template <class T>
void Bart::attention(int heads, const T* q, const T* K, const T* V, int S, T* out, T* scores) const {
    const int hd = d_ / heads;
    for (int h = 0; h < heads; ++h) {
        const T* qh = q + h * hd;
        T mx = -std::numeric_limits<T>::infinity();
        for (int s = 0; s < S; ++s) {
            const T* kh = K + (size_t)s * d_ + h * hd;
            T acc = 0;
            for (int i = 0; i < hd; ++i) acc += qh[i] * kh[i];
            scores[s] = acc;
            if (acc > mx) mx = acc;
        }
        T sum = 0;
        for (int s = 0; s < S; ++s) { scores[s] = exp_t<T>(scores[s] - mx); sum += scores[s]; }
        T* oh = out + h * hd;
        for (int i = 0; i < hd; ++i) oh[i] = 0;
        for (int s = 0; s < S; ++s) {
            const T a = scores[s] / sum;
            const T* vh = V + (size_t)s * d_ + h * hd;
            for (int i = 0; i < hd; ++i) oh[i] += a * vh[i];
        }
    }
}

template <class T>
void Bart::Buf<T>::size(int n_dec, int max_len, int d, int ffn, int vocab) {
    const size_t TT = (size_t)max_len, D = (size_t)d;
    enc_x.assign(TT * D, 0); enc_k.assign(TT * D, 0); enc_v.assign(TT * D, 0); q.assign(TT * D, 0);
    att.assign(TT * D, 0); ffn_.assign((size_t)ffn, 0); scores.assign(TT, 0); x.assign(D, 0); tmp.assign(D, 0);
    ck.assign((size_t)n_dec * TT * D, 0); cv.assign((size_t)n_dec * TT * D, 0);
    kc.assign((size_t)n_dec * TT * D, 0); vc.assign((size_t)n_dec * TT * D, 0);
    (void)vocab;
}

template <class T>
void Bart::run(const std::vector<int>& ids_in, int max_length, std::vector<int>& out, Buf<T>& B) {
    out.clear();
    if (max_length <= 0) max_length = max_len_;
    if (max_length > max_len_) max_length = max_len_;
    int S = (int)ids_in.size();
    if (S > max_len_) S = max_len_;
    const size_t D = (size_t)d_;
    // ---- encoder ----
    for (int t = 0; t < S; ++t) {
        int id = ids_in[t];
        if (id < 0 || id >= vocab_) id = 3;  // <unk>
        embed(pos_enc_, ln_emb_enc_, id, t, &B.enc_x[t * D]);
    }
    const T qs_e = T(1) / std::sqrt((T)(d_ / heads_e_));
    for (auto& L : enc_) {
        for (int t = 0; t < S; ++t) {
            linear(L.sa.q, &B.enc_x[t * D], &B.q[t * D]);
            for (size_t i = 0; i < D; ++i) B.q[t * D + i] *= qs_e;
            linear(L.sa.k, &B.enc_x[t * D], &B.enc_k[t * D]);
            linear(L.sa.v, &B.enc_x[t * D], &B.enc_v[t * D]);
        }
        for (int t = 0; t < S; ++t) attention(heads_e_, &B.q[t * D], B.enc_k.data(), B.enc_v.data(), S, &B.att[t * D], B.scores.data());
        for (int t = 0; t < S; ++t) {
            T* x = &B.enc_x[t * D];
            linear(L.sa.o, &B.att[t * D], B.tmp.data());
            for (size_t i = 0; i < D; ++i) x[i] += B.tmp[i];
            layernorm(L.sa_ln, x);
            linear(L.fc1, x, B.ffn_.data());
            for (int i = 0; i < L.fc1.out; ++i) B.ffn_[i] = gelu(B.ffn_[i]);
            linear(L.fc2, B.ffn_.data(), B.tmp.data());
            for (size_t i = 0; i < D; ++i) x[i] += B.tmp[i];
            layernorm(L.fin_ln, x);
        }
    }
    for (int l = 0; l < n_dec_; ++l) {
        for (int t = 0; t < S; ++t) {
            linear(dec_[l].ca.k, &B.enc_x[t * D], &B.ck[((size_t)l * max_len_ + t) * D]);
            linear(dec_[l].ca.v, &B.enc_x[t * D], &B.cv[((size_t)l * max_len_ + t) * D]);
        }
    }
    // ---- greedy decoding with a key/value cache ----
    const T qs_d = T(1) / std::sqrt((T)(d_ / heads_d_));
    out.push_back(start_);
    int pos = 0;
    while ((int)out.size() < max_length) {
        T* x = B.x.data();
        embed(pos_dec_, ln_emb_dec_, out.back() >= 0 && out.back() < vocab_ ? out.back() : 3, pos, x);
        for (int l = 0; l < n_dec_; ++l) {
            DecL& L = dec_[l];
            T* kc = &B.kc[(size_t)l * max_len_ * D];
            T* vc = &B.vc[(size_t)l * max_len_ * D];
            linear(L.sa.q, x, B.q.data());
            for (size_t i = 0; i < D; ++i) B.q[i] *= qs_d;
            linear(L.sa.k, x, kc + (size_t)pos * D);
            linear(L.sa.v, x, vc + (size_t)pos * D);
            attention(heads_d_, B.q.data(), kc, vc, pos + 1, B.att.data(), B.scores.data());
            linear(L.sa.o, B.att.data(), B.tmp.data());
            for (size_t i = 0; i < D; ++i) x[i] += B.tmp[i];
            layernorm(L.sa_ln, x);
            linear(L.ca.q, x, B.q.data());
            for (size_t i = 0; i < D; ++i) B.q[i] *= qs_d;
            attention(heads_d_, B.q.data(), &B.ck[(size_t)l * max_len_ * D], &B.cv[(size_t)l * max_len_ * D], S,
                      B.att.data(), B.scores.data());
            linear(L.ca.o, B.att.data(), B.tmp.data());
            for (size_t i = 0; i < D; ++i) x[i] += B.tmp[i];
            layernorm(L.ca_ln, x);
            linear(L.fc1, x, B.ffn_.data());
            for (int i = 0; i < L.fc1.out; ++i) B.ffn_[i] = gelu(B.ffn_[i]);
            linear(L.fc2, B.ffn_.data(), B.tmp.data());
            for (size_t i = 0; i < D; ++i) x[i] += B.tmp[i];
            layernorm(L.fin_ln, x);
        }
        int nxt;
        if (forced_eos_ >= 0 && (int)out.size() == max_length - 1) nxt = forced_eos_;
        else {
            T best = -std::numeric_limits<T>::infinity();
            nxt = 0;
            for (int v = 0; v < vocab_; ++v) {
                T s = dot(emb_ + (size_t)v * D, x, d_);
                if (final_bias_) s += (T)final_bias_[v];
                if (s > best) { best = s; nxt = v; }
            }
        }
        out.push_back(nxt);
        ++pos;
        if (nxt == eos_) break;
    }
}

void Bart::generate(const std::vector<int>& input_ids, int max_length, std::vector<int>& out) {
    if (fp32_) run(input_ids, max_length, out, bf_);
    else run(input_ids, max_length, out, bd_);
}

}  // namespace kt_tinyen
