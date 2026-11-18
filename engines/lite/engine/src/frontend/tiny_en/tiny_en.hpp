// Part of kitten-inference (Apache-2.0).
// Kitten TTS 2 mini / nano / pico, English text -> phoneme chunks, on the device: a C++ port of
// src/kitten_inference/tts2_tiny/frontend (english.py: tts_norm, misaki's lexicon G2P, CMUdict, the BART fallback;
// __init__.py: the chunking). spaCy is not ported: its tokenizer is re-implemented from its rules (with the special
// cases exported by export_phone_data.py) and its part-of-speech tagger is replaced by a small rule tagger (only a
// few hundred homographs and function words depend on the tag; see engines/lite/mobile/README.md for the measured
// effect).
#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "uni.hpp"

namespace kt_tinyen {

struct SpTok {           // one spaCy-style token
    u32 text;
    std::string tag;     // Penn Treebank tag
    bool ws = false;     // followed by a space
};

struct Timing {          // for the benchmarks: time spent in the BART fallback during the last call
    double bart_ms = 0;
    int bart_words = 0;
};

class EnglishFrontend {
public:
    // g2p_dir: the model repository's g2p/ folder (cmu_misaki_us.json, g2p_bart_en_us/).
    // data_dir: us_gold.json, us_silver.json, spacy_en_exceptions.json (export_phone_data.py); may equal g2p_dir.
    // vocab_json: the model repository's vocab.json. phonemizer: "v1" (nano, pico) or "v2" (mini).
    static std::unique_ptr<EnglishFrontend> load(const std::string& g2p_dir, const std::string& data_dir,
                                                 const std::string& vocab_json, const std::string& phonemizer,
                                                 std::string& err);
    ~EnglishFrontend();

    // The package's Frontend(text, "en")[0]: phoneme chunks of at most 350 symbols each.
    bool text_to_chunks(std::string_view text_utf8, std::vector<std::string>& chunks, std::string& err);
    // The same for text that is already normalised (kt08_stream, as the package's stream(): front_end(normalise=False)).
    bool normalized_to_chunks(std::string_view normalized_utf8, std::vector<std::string>& chunks, std::string& err);

    // ---- diagnostics (tests and the parity tools) ----
    std::string normalize(std::string_view text_utf8);
    std::vector<SpTok> tokenize(std::string_view normalized_utf8);   // spaCy tokens with the rule tags
    // misaki G2P + fold + vocab filter of already-normalised text with the given tokens (e.g. spaCy's own);
    // returns the phoneme string as EnglishG2P(text, normalise=False).phonemes.
    std::string g2p_tokens(std::vector<SpTok> tokens);
    const Timing& timing() const;
    // The BART fallback alone (english.py Fallback._bart, no memo), and its arithmetic: float32 (default) or double.
    std::string bart_phonemes(std::string_view word_utf8);
    void set_bart_fp32(bool on);
    size_t memory_bytes() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace kt_tinyen
