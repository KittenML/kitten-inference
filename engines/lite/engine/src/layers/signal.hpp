#pragma once
#include "engine/layer.hpp"

class StftFp32 : public ILayer {
public:
    int n_fft    = 20;
    int hop_size = 5;
    // true: torch.stft semantics (reflect padding; the DC and Nyquist bins are exactly real, so their phase is 0 or pi)
    // as Kitten TTS 2 mini/nano/pico were trained with; false: edge padding (TTS 0.8, bit-exact with its ONNX).
    bool torch_stft = false;

    std::vector<float> w_real;
    std::vector<float> w_imag;

    void load_weights(const ReadTensorFn& read) override;
    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape dynamic_output_size(const void* const*, const Shape*, int) const override;
    Shape output_shape(Shape in) const override {
        int T_stft = in.d1 / hop_size + 1;
        return {n_fft + 2, T_stft, 1};
    }
    MemSpace required_mem_space() const override {
        // CUDA: torch_stft runs on the GPU (ops/signal.cu); everything else is staged through the CPU
        if (torch_stft && active_backend()->mem_space == MemSpace::CUDA && active_backend()->stft_torch_fp32_cuda)
            return MemSpace::CUDA;
        return MemSpace::Host;
    }
};

class IstftFp32 : public ILayer {
public:
    int  n_fft      = 20;
    int  hop_size   = 5;
    bool normalized = false;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {1, (in.d1 - 1) * hop_size + n_fft, 1};
    }
    MemSpace required_mem_space() const override {
        MemSpace ms = active_backend()->mem_space;
        if (ms == MemSpace::CUDA && active_backend()->istft_fp32_cuda) return ms;
        return (ms == MemSpace::CUDA || ms == MemSpace::HIP) ? MemSpace::Host : ms;
    }
};

class SineGenFp32 : public ILayer {
public:
    int   sample_rate       = 24000;
    int   harmonic_num      = 8;
    float sine_amp          = 0.1f;
    float voiced_threshold  = 10.0f;
    float noise_std         = 0.0f;
    int   upsample_scale    = 0;
    bool  torch_order       = false;  // Kitten TTS 2 mini/nano/pico: float32 steps in torch's order (see sine_gen_fp32)
    // torch_order only: add the reference's Gaussian excitation noise (noise_std where voiced, sine_amp/3 elsewhere)
    // from a generator seeded with noise_seed on every call, so equal inputs give equal audio.
    bool  noise             = false;
    int   noise_seed        = 0;

    Shape forward(const TensorView*, int, BufferView, ScratchPads&) override;
    DType out_dtype() const override { return DType::fp32(); }
    Shape output_shape(Shape in) const override {
        return {harmonic_num + 1, in.d1, in.d2};
    }
    MemSpace required_mem_space() const override {
        // CUDA: the torch-order generator (Kitten TTS 2 mini/nano/pico) runs on the GPU (ops/signal.cu)
        if (torch_order && upsample_scale > 0 && active_backend()->mem_space == MemSpace::CUDA &&
            active_backend()->sine_gen_torch_fp32_cuda)
            return MemSpace::CUDA;
        return MemSpace::Host;
    }
};
