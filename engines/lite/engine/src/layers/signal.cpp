// signal.cpp — Signal processing layer classes: STFT, ISTFT, SineGen

#include "signal.hpp"
#include "engine/dev_env.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────
// IstftFp32
//   in1 = magnitude  {n_fft/2+1, T, 1}
//   in2 = phase      {n_fft/2+1, T, 1}
//   out = waveform   {1, (T-1)*hop, 1}  (n_fft/2 trimmed each end)
// ─────────────────────────────────────────────────────────────────

Shape IstftFp32::forward(const TensorView* inputs, int n_inputs,
                             BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    BufferView  in2      = (n_inputs > 1) ? inputs[1].view  : BufferView{};
    Shape in2_sh   = (n_inputs > 1) ? inputs[1].shape : Shape{};
    (void)in2_sh;
    const int T_frames  = sh.d1 * sh.d2;
    const int out_len   = (T_frames - 1) * hop_size;   // trimmed length

    if (in1.where == MemSpace::CUDA && active_backend()->istft_fp32_cuda) {
        active_backend()->istft_fp32_cuda(in1.as<float>(), in2.as<float>(), out.as<float>(),
                                          T_frames, n_fft, hop_size, normalized, sc.stream);
        return {1, out_len, 1};
    }

    if (in1.where == MemSpace::Metal && active_backend()->istft_fp32_metal) {
        active_backend()->istft_fp32_metal(
            in1.as<float>(), in2.as<float>(),
            out.as<float>(),
            T_frames, n_fft, hop_size, normalized, sc.stream);
        return {1, out_len, 1};
    }

    istft_fp32(in1.as<float>(),
               in2.as<float>(),
               out.as<float>(),
               T_frames, n_fft, hop_size, normalized);
    return {1, out_len, 1};
}

// ─────────────────────────────────────────────────────────────────
// StftFp32
//   Input:  {1, T_audio, 1}
//   Output: {2*(n_fft/2+1), T_stft, 1}
// ─────────────────────────────────────────────────────────────────

void StftFp32::load_weights(const ReadTensorFn& read)
{
    // Load precomputed Hann-windowed cosine/sine filters from bin file.
    // These are extracted from the ONNX model's weight_forward_real/imag
    // Conv weights to ensure bit-exact match with ONNX float32 accumulation.
    // Stored as [K, n_fft] (row-major), K = n_fft/2+1.
    read_f32_tensor(read, name + ".w_real", w_real);
    read_f32_tensor(read, name + ".w_imag", w_imag);
    const size_t taps = layer_elems(name, {n_fft / 2 + 1, n_fft});
    allow_values(w_real, taps, name + ".w_real");
    allow_values(w_imag, taps, name + ".w_imag");
    // If weights not found in bin, stft_fp32() falls back to computing them.
}

Shape StftFp32::dynamic_output_size(const void* const* /*in_ptrs*/,
                                        const Shape* in_shapes,
                                        int /*n_inputs*/) const
{
    int T_audio = in_shapes[0].d1 * in_shapes[0].d2;
    int T_stft  = T_audio / hop_size + 1;
    return {n_fft + 2, T_stft, 1};
}

Shape StftFp32::forward(const TensorView* inputs, int n_inputs,
                            BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int T_audio = sh.d1 * sh.d2;
    const int T_stft  = T_audio / hop_size + 1;
    const int K       = n_fft / 2 + 1;
    const float* wr   = w_real.empty() ? nullptr : w_real.data();
    const float* wi   = w_imag.empty() ? nullptr : w_imag.data();


    if (in1.where == MemSpace::Metal && !w_real.empty() && !w_imag.empty() && !torch_stft
        && active_backend()->stft_fp32_metal) {
        active_backend()->stft_fp32_metal(
            in1.as<float>(), wr, wi,
            out.as<float>(),
            T_audio, n_fft, hop_size, sc.stream);
        return {2 * K, T_stft, 1};
    }

    if (torch_stft && in1.where == MemSpace::CUDA && active_backend()->stft_torch_fp32_cuda) {
        if (!wr || !wi) throw std::runtime_error(name + ": torch_stft needs the w_real/w_imag filters in the weights");
        active_backend()->stft_torch_fp32_cuda(in1.as<float>(), out.as<float>(), T_audio, n_fft, hop_size, wr, wi,
                                               sc.stream);
        return {2 * K, T_stft, 1};
    }

    // torch_stft runs on the CPU: required_mem_space() is Host, so a Metal input arrives here staged to host memory.
    if (torch_stft) {
        if (in1.where != MemSpace::Host) throw std::runtime_error(name + ": torch_stft needs its input in host memory");
        if (!wr || !wi) throw std::runtime_error(name + ": torch_stft needs the w_real/w_imag filters in the weights");
        stft_torch_fp32(in1.as<float>(), out.as<float>(), T_audio, n_fft, hop_size, wr, wi);
        return {2 * K, T_stft, 1};
    }
    stft_fp32(in1.as<float>(),
              out.as<float>(),
              T_audio, n_fft, hop_size, wr, wi);
    return {2 * K, T_stft, 1};
}

// ─────────────────────────────────────────────────────────────────
// SineGenFp32
//   Input:  F0 {1, T, 1}  (Hz, audio-rate)
//   Output: harmonics {harmonic_num+1, T, 1}
// ─────────────────────────────────────────────────────────────────

Shape SineGenFp32::forward(const TensorView* inputs, int n_inputs,
                               BufferView out, ScratchPads& sc)
{
    BufferView  in1      = inputs[0].view;
    Shape sh       = inputs[0].shape;
    (void)n_inputs;
    const int H = harmonic_num + 1;

    // The Metal kernel is the TTS 0.8 generator (no torch step order, no excitation noise), so Kitten TTS 2
    // mini/nano/pico never take it. required_mem_space() is Host, so the engine stages this layer to the CPU anyway.
    if (in1.where == MemSpace::Metal && upsample_scale > 0 && !torch_order
        && active_backend()->sine_gen_fp32_metal) {
        active_backend()->sine_gen_fp32_metal(
            in1.as<float>(), out.as<float>(),
            sh.d1 * sh.d2, H, sample_rate, upsample_scale,
            sine_amp, voiced_threshold, sc.stream);
        return {H, sh.d1, sh.d2};
    }

    if (torch_order && upsample_scale > 0 && in1.where == MemSpace::CUDA && active_backend()->sine_gen_torch_fp32_cuda) {
        // the same noise as the CPU op: its seed is mixed with a hash of F0 unless KT08_NOISE_SEED_FIXED=1
        static const bool seed_fixed = [] { const char* e = kt_dev_getenv("KT08_NOISE_SEED_FIXED"); return e && e[0] == '1'; }();
        active_backend()->sine_gen_torch_fp32_cuda(in1.as<float>(), out.as<float>(), sh.d1 * sh.d2, harmonic_num,
                                                   sample_rate, sine_amp, voiced_threshold, upsample_scale,
                                                   noise ? noise_std : -1.f,
                                                   (unsigned long long)(uint32_t)noise_seed, !seed_fixed, sc.stream);
        return {H, sh.d1, sh.d2};
    }

    if (torch_order && upsample_scale > 0) {
        sine_gen_torch_fp32(in1.as<float>(), out.as<float>(), sh.d1 * sh.d2, harmonic_num, sample_rate,
                            sine_amp, voiced_threshold, upsample_scale, noise ? noise_std : -1.f,
                            (uint64_t)(uint32_t)noise_seed);
        return {H, sh.d1, sh.d2};
    }
    sine_gen_fp32(in1.as<float>(),
                  out.as<float>(),
                  sh.d1 * sh.d2, harmonic_num, sample_rate,
                  sine_amp, voiced_threshold, noise_std, upsample_scale);
    return {H, sh.d1, sh.d2};
}
