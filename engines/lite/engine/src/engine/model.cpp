#if defined(__APPLE__)
#include "engine/dev_env.hpp"
#include <mach/mach.h>
#endif
#include "model.hpp"
#include "arch_loader.hpp"
#include "weights_loader.hpp"
#include "tensor_pool.hpp"
#include "fp16_store.hpp"
#include "layers/factory.hpp"
#include "layers/activation.hpp"
#include "layers/add.hpp"
#include "layers/attention.hpp"
#include "layers/conv1d.hpp"
#include "layers/conv_fp32.hpp"
#include "layers/conv_int8.hpp"
#include "layers/embedding.hpp"
#include "layers/gemm.hpp"
#include "layers/lstm.hpp"
#include "layers/norm.hpp"
#include "layers/signal.hpp"
#include "layers/tensor.hpp"
#include "backend/cpu/ops_neon.hpp"
#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include <cstdio>
#include <mutex>
#include <cstdlib>
#include <memory>
#include <unordered_set>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#elif defined(__APPLE__)
#include <malloc/malloc.h>   // malloc_zone_pressure_relief
#endif

static inline double _now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// ── Shape helpers ──────────────────────────────────────────────
// Internal layout: Shape{d0=C, d1=H_or_T, d2=W} (NHWC-derived).
// External format: compact vector — last element is C, leading elements are spatial dims.
// spec_to_shape  : compact vector[int]  →  Shape  (with optional runtime override)
// shape_to_compact: Shape            →  compact vector[int]  (inverse)

static Shape spec_to_shape(const std::vector<int>& spec,
                             const std::vector<int>& rt = {})
{
    int spec_size = (int)spec.size();
    int features  = spec_size >= 1 ? spec.back() : 1;  // features  → d0
    int primary   = spec_size >= 2 ? spec[0]     : 1;  // primary   → d1
    int secondary = spec_size >= 3 ? spec[1]     : 1;  // secondary → d2
    if (!rt.empty()) {
        int rt_size = (int)rt.size();
        if (features  == 0) features  = rt_size >= 1 ? rt.back() : 1;
        if (primary   == 0) primary   = rt_size >= 2 ? rt[0]     : 1;
        if (secondary == 0) secondary = rt_size >= 3 ? rt[1]     : 1;
    }
    return {features, std::max(primary, 1), std::max(secondary, 1)};
}

// Inverse of spec_to_shape: drops trailing d2=1 then d1=1.
static std::vector<int> shape_to_compact(const Shape& b) {
    if (b.d2 > 1) return {b.d1, b.d2, b.d0};
    if (b.d1 > 1) return {b.d1, b.d0};
    return {b.d0};
}

// Shared CPU allocator for host-side staging buffers.
static CpuAllocator s_cpu_alloc;

struct LayerMemoryUsage {
    size_t persistent_weights = 0;
    size_t local_scratch = 0;
    size_t persistent_state = 0;
};

template<typename T>
static size_t vec_bytes(const std::vector<T>& v) {
    return v.capacity() * sizeof(T);
}

static size_t buf_bytes(const Buffer& b) {
    return b.valid() ? b.bytes() : 0;
}

static size_t shape_elems(const Shape& sh) {
    return (size_t)std::max(sh.d0, 0) *
           (size_t)std::max(sh.d1, 0) *
           (size_t)std::max(sh.d2, 0);
}

static size_t shape_bytes(const Shape& sh, const DType& dtype) {
    return shape_elems(sh) * dtype.elem_bytes();
}

static void grow_buffer(Buffer& b, size_t need, Allocator& alloc) {
    if (need > b.bytes())
        b = alloc.make_buffer(need);
}

template<size_t N>
static size_t buf_array_bytes(const Buffer (&buffers)[N]) {
    size_t total = 0;
    for (const auto& b : buffers) total += buf_bytes(b);
    return total;
}

static size_t buf_vector_bytes(const std::vector<Buffer>& buffers) {
    size_t total = 0;
    for (const auto& b : buffers) total += buf_bytes(b);
    return total;
}

static LayerMemoryUsage layer_memory_usage(const ILayer* layer) {
    LayerMemoryUsage m;
    if (!layer) return m;

    if (auto L = dynamic_cast<const ConvInt8*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight) + vec_bytes(L->req_scale)
                              + vec_bytes(L->req_mult) + vec_bytes(L->req_exp)
                              + vec_bytes(L->eff_bias) + vec_bytes(L->wino_eff_bias)
                              + vec_bytes(L->w_row_sums) + vec_bytes(L->w_dw_hwc)
                              + vec_bytes(L->dw_eff_bias32) + buf_bytes(L->w_packed);
        if (L->w_packed_nhwc.ptr() != L->w_packed.ptr())
            m.persistent_weights += buf_bytes(L->w_packed_nhwc);
        for (const auto& rows : L->w_wino_row_sums) m.persistent_weights += vec_bytes(rows);
        m.persistent_weights += buf_array_bytes(L->w_wino_packed);
    } else if (auto L = dynamic_cast<const ConvFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_fp32) + vec_bytes(L->bias_fp32)
                              + buf_bytes(L->w_packed_f32)
                              + buf_array_bytes(L->w_wino_f32_packed);
    } else if (auto L = dynamic_cast<const GemmInt8*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight) + vec_bytes(L->req_scale)
                              + vec_bytes(L->req_mult) + vec_bytes(L->req_exp)
                              + vec_bytes(L->eff_bias) + vec_bytes(L->w_row_sums)
                              + buf_bytes(L->w_packed);
    } else if (auto L = dynamic_cast<const GemmFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_fp32) + vec_bytes(L->bias_fp32)
                              + buf_bytes(L->w_packed_f32);
    } else if (auto L = dynamic_cast<const Conv1dFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_fp32) + vec_bytes(L->bias_fp32)
                              + buf_bytes(L->w_packed_f32)
                              + buf_vector_bytes(L->w_per_k_packed_)
                              + buf_vector_bytes(L->w_wino_packed_)
                              + buf_bytes(L->w_kmerged_packed_) + buf_bytes(L->w_kmerged_h_);
        m.local_scratch += vec_bytes(L->pad_buf_) + vec_bytes(L->tmp_buf_);
    } else if (auto L = dynamic_cast<const Conv1dInt8*>(layer)) {
        m.persistent_weights += vec_bytes(L->w_scales) + vec_bytes(L->bias_fp32)
                              + vec_bytes(L->eff_zeros)
                              + buf_vector_bytes(L->w_per_k_packed_)
                              + buf_bytes(L->w_packed_full_)
                              + buf_bytes(L->w_gpu_packed_);
        m.local_scratch += vec_bytes(L->im2col_buf_) + vec_bytes(L->pad_buf_)
                         + vec_bytes(L->tmp_buf_) + vec_bytes(L->inp_scratch_)
                         + vec_bytes(L->req_scratch_) + vec_bytes(L->row_scale_buf_);
    } else if (auto L = dynamic_cast<const ConvTranspose1dFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_fp32) + vec_bytes(L->bias_fp32)
                              + buf_vector_bytes(L->w_packed_k)
                              + vec_bytes(L->w_gpu_data);
        m.local_scratch += vec_bytes(L->pad_buf_);
    } else if (auto L = dynamic_cast<const PatchPrepFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->cls_token) + vec_bytes(L->pos_embed);
    } else if (auto L = dynamic_cast<const AttentionFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->w_qkv) + vec_bytes(L->b_qkv)
                              + vec_bytes(L->w_proj) + vec_bytes(L->b_proj)
                              + buf_bytes(L->w_qkv_packed) + buf_bytes(L->w_proj_packed);
    } else if (auto L = dynamic_cast<const SeqGemmFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_fp32) + vec_bytes(L->bias_fp32)
                              + buf_bytes(L->w_packed_f32)
                              + vec_bytes(L->w_i8_matvec_)
                              + vec_bytes(L->w_scales_matvec_)
                              + buf_bytes(L->w_i8_matvec_buf_)
                              + buf_bytes(L->w_scales_matvec_buf_);
    } else if (auto L = dynamic_cast<const PosEmbAddFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight);
    } else if (auto L = dynamic_cast<const SeqGemmInt8*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight_i8) + vec_bytes(L->w_scales)
                              + vec_bytes(L->bias_fp32) + vec_bytes(L->eff_zeros)
                              + buf_bytes(L->w_packed)
                              + buf_bytes(L->weight_i8_buf_)
                              + buf_bytes(L->w_scales_buf_);
    } else if (auto L = dynamic_cast<const AttentionInt8*>(layer)) {
        m.persistent_weights += vec_bytes(L->w_qkv_i8) + vec_bytes(L->w_qkv_scales)
                              + vec_bytes(L->b_qkv) + vec_bytes(L->w_proj_i8)
                              + vec_bytes(L->w_proj_scales) + vec_bytes(L->b_proj)
                              + vec_bytes(L->eff_zeros)
                              + buf_bytes(L->w_qkv_packed)
                              + buf_bytes(L->w_proj_packed);
    } else if (auto L = dynamic_cast<const CausalAttnCoreFp32*>(layer)) {
        m.persistent_state += buf_bytes(L->kv_cache_buf_);
        m.local_scratch += vec_bytes(L->fwd_scratch_);
    } else if (auto L = dynamic_cast<const RmsNormFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->gamma);
    } else if (auto L = dynamic_cast<const LayerNormFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->gamma) + vec_bytes(L->beta);
    } else if (auto L = dynamic_cast<const AdaIn1dFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->fc_weight) + vec_bytes(L->fc_bias)
                              + vec_bytes(L->norm_weight) + vec_bytes(L->norm_bias)
                              + buf_bytes(L->fc_packed);
        m.local_scratch += vec_bytes(L->gamma_beta_buf_) + buf_bytes(L->gamma_beta_d_);
    } else if (auto L = dynamic_cast<const AdaLayerNormFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->fc_weight) + vec_bytes(L->fc_bias)
                              + buf_bytes(L->fc_packed);
        m.local_scratch += vec_bytes(L->gamma_beta_buf_) + buf_bytes(L->gamma_beta_d_);
    } else if (auto L = dynamic_cast<const EmbeddingFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->weight);
    } else if (auto L = dynamic_cast<const BertEmbeddingsFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->word_weight) + vec_bytes(L->pos_weight)
                              + vec_bytes(L->type_weight) + vec_bytes(L->ln_weight)
                              + vec_bytes(L->ln_bias);
    } else if (auto L = dynamic_cast<const StftFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->w_real) + vec_bytes(L->w_imag);
    } else if (auto L = dynamic_cast<const SnakeFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->alpha) + vec_bytes(L->inv_alpha)
                              + buf_bytes(L->alpha_d) + buf_bytes(L->inv_alpha_d);
    } else if (auto L = dynamic_cast<const LstmFp32*>(layer)) {
        m.persistent_weights += vec_bytes(L->w_ih) + vec_bytes(L->w_hh)
                              + vec_bytes(L->b_ih) + vec_bytes(L->b_hh)
                              + buf_bytes(L->w_ih_packed) + buf_bytes(L->w_hh_packed)
                              + vec_bytes(L->w_ih_r) + vec_bytes(L->w_hh_r)
                              + vec_bytes(L->b_ih_r) + vec_bytes(L->b_hh_r)
                              + buf_bytes(L->w_ih_r_packed) + buf_bytes(L->w_hh_r_packed)
                              + buf_bytes(L->w_ih_gpu) + buf_bytes(L->b_ih_gpu)
                              + buf_bytes(L->w_hh_gpu) + buf_bytes(L->b_hh_gpu)
                              + buf_bytes(L->w_ih_r_gpu) + buf_bytes(L->b_ih_r_gpu)
                              + buf_bytes(L->w_hh_r_gpu) + buf_bytes(L->b_hh_r_gpu);
    }
    return m;
}

static double mib(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

void InferenceModel::reset_forward_memory_accounting() {
    active_tensor_bytes_.clear();
    active_tensor_total_bytes_ = 0;
    manual_min_transient_peak_bytes_ = 0;
    manual_min_activation_peak_bytes_ = 0;
    manual_min_scratch_peak_bytes_ = 0;
    manual_min_staging_peak_bytes_ = 0;
    engine_transient_peak_bytes_ = 0;
}

void InferenceModel::remember_active_tensor_bytes(const std::string& name, size_t bytes) {
    if (name.empty()) return;
    auto it = active_tensor_bytes_.find(name);
    if (it != active_tensor_bytes_.end()) {
        active_tensor_total_bytes_ -= it->second;
        it->second = bytes;
    } else {
        active_tensor_bytes_[name] = bytes;
    }
    active_tensor_total_bytes_ += bytes;
}

void InferenceModel::forget_active_tensor_bytes(const std::string& name) {
    auto it = active_tensor_bytes_.find(name);
    if (it == active_tensor_bytes_.end()) return;
    active_tensor_total_bytes_ -= it->second;
    active_tensor_bytes_.erase(it);
}

bool InferenceModel::is_graph_output(const std::string& name) const {
    return std::find(loaded_arch_.arch.output_names.begin(),
                     loaded_arch_.arch.output_names.end(),
                     name) != loaded_arch_.arch.output_names.end();
}

void InferenceModel::record_manual_transient_peak(size_t activation_bytes,
                                                  size_t scratch_bytes,
                                                  size_t staging_bytes) {
    const size_t total = activation_bytes + scratch_bytes + staging_bytes;
    if (total <= manual_min_transient_peak_bytes_) return;
    manual_min_transient_peak_bytes_ = total;
    manual_min_activation_peak_bytes_ = activation_bytes;
    manual_min_scratch_peak_bytes_ = scratch_bytes;
    manual_min_staging_peak_bytes_ = staging_bytes;
}

size_t InferenceModel::current_engine_transient_bytes() const {
    size_t total = 0;
    for (const auto& b : pool_.buffers) total += buf_bytes(b);
    for (const auto& b : scratch_.buffers) total += buf_bytes(b);
    for (const auto& b : h_stage_ins_) total += buf_bytes(b);
    total += buf_bytes(h_stage_out_);
    for (const auto& b : h_stage_scratch_) total += buf_bytes(b);
    return total;
}

void InferenceModel::record_engine_transient_peak() {
    engine_transient_peak_bytes_ = std::max(engine_transient_peak_bytes_,
                                            current_engine_transient_bytes());
}


// ──────────────────────────────────────────────────────────────
// update_buffers — propagate shapes + grow-only buffer allocation.
//
// For each input in loaded_arch_.arch.all_inputs:
//   - Fixed dims (> 0 in InputSpec) are taken from the spec.
//   - Variable dims (== 0 in InputSpec) are filled from the matching
//     runtime ModelIOTensor's shape (if present in `inputs`); default to 1.
// Shapes are then propagated through the graph and buffers / scratch
// are grown as needed.  Called from the constructor (with an empty map,
// which uses spec sizes for all fixed-size models) and at the start of
// every forward() call.
// ──────────────────────────────────────────────────────────────
void InferenceModel::update_buffers(
    const std::unordered_map<std::string, ModelIOTensor>& inputs)
{
    // ── Resolve concrete Shape for every input ─────────────
    std::unordered_map<std::string, Shape> input_shapes;
    for (const auto& inp : loaded_arch_.arch.all_inputs) {
        auto it = inputs.find(inp.name);
        input_shapes[inp.name] = spec_to_shape(
            inp.shape,
            it != inputs.end() ? it->second.shape : std::vector<int>{});
    }

    auto result = pool_.propagate_shapes(loaded_arch_.arch, input_shapes);
    if (inference_memory_mode_releases_transients(memory_mode_))
        return;

    // ── Grow-only buffer allocation ───────────────────────────
    for (int slot = 0; slot < pool_.slots.num_slots; ++slot)
        pool_.grow(slot, result.slot_sizes[slot], *alloc_);

    // ── Grow-only scratch allocation ──────────────────────────
    for (int node_idx = 0; node_idx < (int)loaded_arch_.arch.nodes.size(); ++node_idx) {
        if (!pool_.node_layers[node_idx]) continue;
        ILayer* layer = pool_.node_layers[node_idx];
        const std::string& in0 = loaded_arch_.arch.nodes[node_idx].inputs.empty()
            ? std::string("") : loaded_arch_.arch.nodes[node_idx].inputs[0];
        Shape input_shape = result.shapes.count(in0) ? result.shapes.at(in0) : Shape{};
        size_t sizes[Scratch::N] = {};
        layer->scratch_needed(input_shape, sizes);
        const bool needs_staging = (layer->required_mem_space() != active_backend()->mem_space);
        for (int i = 0; i < Scratch::N; ++i) {
            if (needs_staging)
                grow_buffer(h_stage_scratch_[i], sizes[i], s_cpu_alloc);
            else
                scratch_.grow(i, sizes[i], *alloc_);
        }
    }
}


// ──────────────────────────────────────────────────────────────
// dispatch_layer — run one layer, staging inputs/output to host if needed.
// `inputs` is a mutable stack array — views are replaced with host copies
// in the staging path without affecting the caller's slot buffers.
// When inputs_staged=true the caller has already copied inputs into
// h_stage_ins and updated inputs[i].view; only output staging is performed.
// ──────────────────────────────────────────────────────────────
static Shape dispatch_layer(
    ILayer* layer,
    TensorView* inputs, int n_inputs,
    BufferView out_view,
    ScratchPads& scratch,
    Allocator* alloc,
    Buffer* h_stage_ins,   // [MAX_LAYER_INPUTS], grow-only host buffers
    Buffer& h_stage_out,
    bool inputs_staged = false)
{
    const bool needs_staging = (layer->required_mem_space() != active_backend()->mem_space);
    const size_t out_device_bytes = out_view.bytes;

    if (needs_staging) {
        void* dev_out = out_view.ptr;
        if (!inputs_staged) {
            for (int i = 0; i < n_inputs; ++i) {
                grow_buffer(h_stage_ins[i], inputs[i].view.bytes, s_cpu_alloc);
                alloc->copy_d2h(h_stage_ins[i].ptr(), inputs[i].view.ptr, inputs[i].view.bytes);
                inputs[i].view = h_stage_ins[i].view;
            }
        }
        grow_buffer(h_stage_out, out_view.bytes, s_cpu_alloc);
        out_view = h_stage_out.view;
        Shape output_shape = layer->forward(inputs, n_inputs, out_view, scratch);
        // Copy back the output tensor, not the whole device slot: slots keep the capacity of the largest chunk seen,
        // so a full-slot copy moved tens of MB per staged layer whatever the chunk length. Never more than
        // out_device_bytes (the device buffer), which out_view.bytes may exceed after a larger earlier layer.
        size_t copy_bytes = out_device_bytes;
        const size_t live_bytes = shape_bytes(output_shape, layer->out_dtype());
        if (live_bytes > 0 && live_bytes < copy_bytes) copy_bytes = live_bytes;
        alloc->copy_h2d(dev_out, h_stage_out.ptr(), copy_bytes);
        return output_shape;
    }
    return layer->forward(inputs, n_inputs, out_view, scratch);
}

// ──────────────────────────────────────────────────────────────
// execute_graph — run all graph nodes using pre-built ScratchPads
// ──────────────────────────────────────────────────────────────
void InferenceModel::execute_graph(ScratchPads& scratch) {
    const bool release_transients = inference_memory_mode_releases_transients(memory_mode_);
    const bool release_layer_scratch = inference_memory_mode_releases_layer_scratch(memory_mode_);
    for (int node_idx = 0; node_idx < (int)loaded_arch_.arch.nodes.size(); ++node_idx) {
        const GraphNode& node = loaded_arch_.arch.nodes[node_idx];
        ILayer* layer = pool_.node_layers[node_idx];
        if (!layer) throw std::runtime_error("No layer for: " + node.name);
        const std::string out_name = node.outputs.empty() ? std::string("") : node.outputs[0];
        int out_slot = pool_.slots.slot_map.at(out_name);

        // Resolve slot index for each input
        int n_inputs = (int)node.inputs.size();
        if (n_inputs > MAX_LAYER_INPUTS)
            throw std::runtime_error("node '" + node.name + "' has " +
                                     std::to_string(n_inputs) + " inputs (max " +
                                     std::to_string(MAX_LAYER_INPUTS) + ")");
        int in_slots[MAX_LAYER_INPUTS] = {};
        for (int i = 0; i < n_inputs; ++i)
            in_slots[i] = pool_.slots.slot_map.at(node.inputs[i]);

        // If the layer requires host memory (CPU-only under a device backend), stage all
        // inputs into h_stage_ins_ once.  The same staged copies are reused for both
        // dynamic_output_size (which may read tensor contents) and forward().
        const bool needs_staging = (layer->required_mem_space() != active_backend()->mem_space);
        const void* dyn_ptrs[MAX_LAYER_INPUTS] = {};
        Shape    dyn_shapes[MAX_LAYER_INPUTS] = {};
        for (int i = 0; i < n_inputs; ++i) {
            dyn_ptrs[i]   = pool_.raw_ptr(in_slots[i]);
            dyn_shapes[i] = pool_.slot_shapes[in_slots[i]];
        }
        if (needs_staging) {
            for (int i = 0; i < n_inputs; ++i) {
                size_t size_bytes = pool_.buffers[in_slots[i]].bytes();
                // the input tensor only, not the slot's capacity (see dispatch_layer)
                const size_t live_bytes = shape_bytes(pool_.slot_shapes[in_slots[i]], pool_.slot_dtype[in_slots[i]]);
                if (live_bytes > 0 && live_bytes < size_bytes) size_bytes = live_bytes;
                if (size_bytes > 0) {
                    if (!h_stage_ins_[i].valid() || h_stage_ins_[i].bytes() < size_bytes)
                        h_stage_ins_[i] = s_cpu_alloc.make_buffer(size_bytes);
                    alloc_->copy_d2h(h_stage_ins_[i].ptr(), dyn_ptrs[i], size_bytes);
                    dyn_ptrs[i] = h_stage_ins_[i].ptr();
                }
            }
        }

        // Grow output buffer to fit dynamic output shape
        Shape output_shape = layer->dynamic_output_size(dyn_ptrs, dyn_shapes, n_inputs);
        const size_t output_bytes = shape_bytes(output_shape, pool_.slot_dtype[out_slot]);
        {
            if (output_bytes > pool_.buffers[out_slot].bytes())
                pool_.buffers[out_slot] = alloc_->make_buffer(output_bytes);
        }

        // Grow scratch buffers for this layer.
        // This intentionally runs per-node rather than just once in update_buffers:
        // variable-length outputs (e.g. LengthRegulate) can change the input shape
        // seen by later layers at runtime, growing scratch requirements beyond the
        // static estimate computed during update_buffers.
        size_t sizes[Scratch::N] = {};
        size_t scratch_bytes = 0;
        {
            layer->scratch_needed(n_inputs > 0 ? pool_.slot_shapes[in_slots[0]] : Shape{}, sizes);
            for (int i = 0; i < Scratch::N; ++i)
                scratch_bytes += sizes[i];
            if (needs_staging) {
                for (int i = 0; i < Scratch::N; ++i)
                    grow_buffer(h_stage_scratch_[i], sizes[i], s_cpu_alloc);
            } else {
                for (int i = 0; i < Scratch::N; ++i) {
                    scratch_.grow(i, sizes[i], *alloc_);
                    scratch.buffers[i] = scratch_.buffers[i].valid()
                                       ? scratch_.buffers[i].view : BufferView{};
                }
            }
        }

        size_t staged_input_bytes = 0;
        if (needs_staging) {
            for (int i = 0; i < n_inputs; ++i) {
                auto it = active_tensor_bytes_.find(node.inputs[i]);
                if (it != active_tensor_bytes_.end())
                    staged_input_bytes += it->second;
            }
        }
        record_manual_transient_peak(
            active_tensor_total_bytes_ + output_bytes,
            needs_staging ? 0 : scratch_bytes,
            needs_staging ? staged_input_bytes + output_bytes + scratch_bytes : 0);

        ScratchPads layer_scratch = scratch;
        if (needs_staging) {
            layer_scratch.stream = nullptr;
            for (int i = 0; i < Scratch::N; ++i)
                layer_scratch.buffers[i] = h_stage_scratch_[i].valid()
                                         ? h_stage_scratch_[i].view : BufferView{};
        }

        // Build TensorView array from slot buffers.
        // If inputs were staged above, use the staged host views so dispatch_layer
        // can skip the redundant d2h copy.
        TensorView inp_buf[MAX_LAYER_INPUTS];
        for (int i = 0; i < n_inputs; ++i) {
            inp_buf[i].view  = (needs_staging && h_stage_ins_[i].valid())
                               ? h_stage_ins_[i].view : pool_.buffers[in_slots[i]].view;
            inp_buf[i].shape = pool_.slot_shapes[in_slots[i]];
        }
        BufferView out_view = pool_.buffers[out_slot].view;

        // conv with a fused residual add (the next node's other operand), if its shape matches the output.
        Conv1dFp32* res_conv = nullptr;
        const float* res_ptr = nullptr;
        if (!fused_res_idx_.empty() && fused_res_idx_[node_idx] >= 0) {
            const GraphNode& nb = loaded_arch_.arch.nodes[node_idx + 1];
            const int rslot = pool_.slots.slot_map.at(nb.inputs[fused_res_idx_[node_idx]]);
            const Shape& rs = pool_.slot_shapes[rslot];
            const bool ok = rs.d0 == output_shape.d0 && rs.d1 == output_shape.d1 && rs.d2 == output_shape.d2 &&
                            pool_.buffers[rslot].bytes() >= output_bytes;
            fused_live_[node_idx + 1] = ok ? 1 : 0;
            if (ok) {
                res_conv = static_cast<Conv1dFp32*>(layer);
                res_ptr = pool_.buffers[rslot].as<float>();
                res_conv->fused_residual = res_ptr;
                res_conv->fused_residual_done = false;
            }
        }

        if (!fused_skip_.empty() && fused_skip_[node_idx] &&
            (fused_res_idx_.empty() || fused_res_idx_[node_idx - 1] < 0 || fused_live_[node_idx])) {
            // the previous node already wrote this node's output into this (shared) slot.
            pool_.slot_shapes[out_slot] = inp_buf[fused_skip_[node_idx] - 1].shape;
        } else if (profiling_enabled_) {
            double start_ms = _now_ms();
            if (fp16_store_) fp16_store_->widen(layer);
            pool_.slot_shapes[out_slot] = dispatch_layer(layer, inp_buf, n_inputs, out_view, layer_scratch,
                                                         alloc_, h_stage_ins_, h_stage_out_,
                                                         needs_staging);
            if (fp16_store_) fp16_store_->unwiden(layer);
            double elapsed_ms = _now_ms() - start_ms;
            op_ms_[op_type_name(node.op)] += elapsed_ms;
            layer_ms_[node.name] += elapsed_ms;
        } else {
            if (fp16_store_) fp16_store_->widen(layer);
            pool_.slot_shapes[out_slot] = dispatch_layer(layer, inp_buf, n_inputs, out_view, layer_scratch,
                                                         alloc_, h_stage_ins_, h_stage_out_,
                                                         needs_staging);
            if (fp16_store_) fp16_store_->unwiden(layer);
        }

        if (res_conv) {
            if (!res_conv->fused_residual_done) {  // conv path without the in-kernel residual: add it here
                const Shape& os = pool_.slot_shapes[out_slot];
                float* o = pool_.buffers[out_slot].as<float>();
                active_backend()->add_fp32(o, res_ptr, o, os.d0 * os.d1 * os.d2, false, nullptr);
            }
            res_conv->fused_residual = nullptr;
            res_conv->fused_residual_done = false;
        }

        remember_active_tensor_bytes(out_name, output_bytes);
        record_engine_transient_peak();
        if (trace_rss_) {  // diagnostic: VmRSS right after the node, before its transients are released
            long pages = 0, res = 0;
#if defined(__APPLE__)
            {   // macOS has no /proc: phys_footprint in 4 KiB units, so the print below stays in MiB
                task_vm_info_data_t vi; mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
                if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) == KERN_SUCCESS) res = (long)(vi.phys_footprint / 4096);
            }
#else
            if (FILE* f = fopen("/proc/self/statm", "r")) { if (fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0; fclose(f); }
#endif
            std::string big;
            double pool_mb = 0, scr_mb = 0;
            for (int si = 0; si < pool_.slots.num_slots; ++si) {
                const double mb = pool_.buffers[si].bytes() / 1048576.0;
                pool_mb += mb;
                if (mb >= 4.0) big += " s" + std::to_string(si) + ":" + std::to_string((int)mb);
            }
            for (int i = 0; i < Scratch::N; ++i) scr_mb += scratch_.buffers[i].bytes() / 1048576.0;
            fprintf(stderr, "TRACE node=%d name=%s rss_mb=%.1f out=%dx%dx%d out_slot=%d pool_mb=%.1f scratch_mb=%.1f big=[%s]\n",
                    node_idx, node.name.c_str(), res * 4096.0 / 1048576.0, pool_.slot_shapes[out_slot].d0,
                    pool_.slot_shapes[out_slot].d1, pool_.slot_shapes[out_slot].d2, out_slot, pool_mb, scr_mb, big.c_str());
        }

        if (release_transients) {
            alloc_->sync();
            if (release_layer_scratch)
                layer->release_runtime_scratch();
            for (int i = 0; i < Scratch::N; ++i) {
                scratch_.buffers[i].release();
                scratch.buffers[i] = {};
            }
            for (int i = 0; i < MAX_LAYER_INPUTS; ++i)
                h_stage_ins_[i].release();
            h_stage_out_.release();
            for (int i = 0; i < Scratch::N; ++i)
                h_stage_scratch_[i].release();

            for (int i = 0; i < n_inputs; ++i) {
                const std::string& input_name = node.inputs[i];
                auto last_it = pool_.slots.last_use.find(input_name);
                if (last_it == pool_.slots.last_use.end() || last_it->second != node_idx)
                    continue;
                if (is_graph_output(input_name) || input_name == out_name || in_slots[i] == out_slot)
                    continue;
                pool_.buffers[in_slots[i]].release();
                pool_.slot_shapes[in_slots[i]] = {};
                forget_active_tensor_bytes(input_name);
            }
            // (KITTEN_MALLOC_TRIM=2) also return freed pages after every node, not only per request.
            // macOS libmalloc keeps freed large blocks in its large-entry cache (counted in phys_footprint);
            // glibc keeps freed sub-mmap-threshold blocks resident. Values untouched (lossless).
            static const bool trim_node = [] { const char* e = kt_dev_getenv("KITTEN_MALLOC_TRIM"); return e && e[0] == '2'; }();
            if (trim_node) {
#if defined(__GLIBC__)
                malloc_trim(0);
#elif defined(__APPLE__)
                malloc_zone_pressure_relief(nullptr, 0);
#endif
            }
        } else {
            for (int i = 0; i < n_inputs; ++i) {
                const std::string& input_name = node.inputs[i];
                auto last_it = pool_.slots.last_use.find(input_name);
                if (last_it != pool_.slots.last_use.end() &&
                    last_it->second == node_idx &&
                    !is_graph_output(input_name) &&
                    input_name != out_name)
                    forget_active_tensor_bytes(input_name);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────
// plan_fused_epilogues — AdaIN1d followed by Snake1D or LeakyReLU that consumes only the AdaIN output
// (and that output has no other use and is not a graph output): the AdaIN layer applies the activation in its final
// pass (bit-identical, one fewer [T, C] write + read) into its own output slot, and the activation node runs in place
// on that slot as a no-op. CPU (x86) backend only; KITTEN_FP32_FUSE=0 disables.
// ──────────────────────────────────────────────────────────────
void InferenceModel::plan_fused_epilogues() {
    const auto& nodes = loaded_arch_.arch.nodes;
    fused_skip_.assign(nodes.size(), 0);
    if (active_backend() && active_backend()->mem_space == MemSpace::CUDA && active_backend()->conv1d_fp32_cuda) {
        // CUDA: (1) AdaIn1dFp32 -> SnakeFp32 that consumes only the AdaIN output: the AdaIN kernel applies Snake in its
        // output pass and the Snake node runs in place on its slot as a no-op (KITTEN_CUDA_FUSE_SNAKE=0 disables);
        // (2) the same Conv1dFp32 -> AddFp32 residual fusion as the CPU's below: the conv's implicit GEMM adds the
        // other operand in its epilogue, the add runs in place on the conv's slot as a no-op (KITTEN_CUDA_FUSE_ADD=0).
        fused_res_idx_.assign(nodes.size(), -1);
        std::unordered_map<std::string, ILayer*> by_name;
        for (auto& l : loaded_arch_.layers) if (l) by_name[l->name] = l.get();
        std::unordered_map<std::string, int> uses;
        for (const auto& n : nodes) for (const auto& t : n.inputs) ++uses[t];
        const char* envs = kt_dev_getenv("KITTEN_CUDA_FUSE_SNAKE");
        if (!(envs && envs[0] == '0') && active_backend()->ada_in1d_snake_fp32_cuda) {
            for (size_t i = 0; i + 1 < nodes.size(); ++i) {
                const GraphNode& a = nodes[i];
                const GraphNode& b = nodes[i + 1];
                if (a.outputs.size() != 1 || b.outputs.size() != 1 || b.inputs.size() != 1) continue;
                if (b.inputs[0] != a.outputs[0] || uses[a.outputs[0]] != 1 || is_graph_output(a.outputs[0])) continue;
                auto ia = by_name.find(a.name), ib = by_name.find(b.name);
                if (ia == by_name.end() || ib == by_name.end()) continue;
                auto* ada = dynamic_cast<AdaIn1dFp32*>(ia->second);
                auto* sn = dynamic_cast<SnakeFp32*>(ib->second);
                if (!ada || !sn || ada->relu || sn->relu) continue;
                ada->cuda_snake_epi = sn;
                fused_skip_[i + 1] = 1;
                ++i;
            }
        }
        const char* envc = kt_dev_getenv("KITTEN_CUDA_FUSE_ADD");
        if (envc && envc[0] == '0') return;
        for (size_t i = 0; i + 1 < nodes.size(); ++i) {
            if (fused_skip_[i] || fused_skip_[i + 1]) continue;
            const GraphNode& a = nodes[i];
            const GraphNode& b = nodes[i + 1];
            if (a.outputs.size() != 1 || b.outputs.size() != 1 || b.inputs.size() != 2) continue;
            if (b.inputs[0] == b.inputs[1]) continue;
            int idx = b.inputs[0] == a.outputs[0] ? 0 : (b.inputs[1] == a.outputs[0] ? 1 : -1);
            if (idx < 0 || uses[a.outputs[0]] != 1 || is_graph_output(a.outputs[0])) continue;
            auto ia = by_name.find(a.name), ib = by_name.find(b.name);
            if (ia == by_name.end() || ib == by_name.end()) continue;
            auto* conv = dynamic_cast<Conv1dFp32*>(ia->second);
            auto* add = dynamic_cast<AddFp32*>(ib->second);
            if (!conv || !add || add->relu || conv->relu) continue;
            fused_res_idx_[i] = (signed char)(1 - idx);
            fused_skip_[i + 1] = (char)(idx + 1);
            ++i;
        }
        return;
    }
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
    const char* env = kt_dev_getenv("KITTEN_FP32_FUSE");
    if (env && env[0] == '0') return;
    if (!active_backend() || active_backend()->mem_space != MemSpace::Host) return;
    std::unordered_map<std::string, ILayer*> by_name;
    for (auto& l : loaded_arch_.layers) if (l) by_name[l->name] = l.get();
    std::unordered_map<std::string, int> uses;
    for (const auto& n : nodes) for (const auto& t : n.inputs) ++uses[t];
#if defined(__AVX2__) || defined(__AVX512F__)   // the AdaIN1d final pass applies the activation on x86 only (no NEON epilogue)
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        const GraphNode& a = nodes[i];
        const GraphNode& b = nodes[i + 1];
        if (a.outputs.size() != 1 || b.outputs.size() != 1 || b.inputs.size() != 1) continue;
        if (b.inputs[0] != a.outputs[0] || uses[a.outputs[0]] != 1 || is_graph_output(a.outputs[0])) continue;
        auto ia = by_name.find(a.name), ib = by_name.find(b.name);
        if (ia == by_name.end() || ib == by_name.end()) continue;
        auto* ada = dynamic_cast<AdaIn1dFp32*>(ia->second);
        if (!ada) continue;
        AdaEpilogue epi;
        if (auto* sn = dynamic_cast<SnakeFp32*>(ib->second)) {
            if (sn->alpha_d.valid() || sn->inv_alpha_d.valid()) continue;
            epi.kind = 1;
            epi.alpha = sn->alpha.data();
            epi.inv_alpha = sn->inv_alpha.data();
            ada->epi_min_C = (int)sn->alpha.size();
        } else if (auto* act = dynamic_cast<ActivationFp32*>(ib->second)) {
            if (act->act_type != ActType::LEAKY_RELU) continue;
            epi.kind = 2;
            epi.slope = act->alpha;
            ada->epi_min_C = 0;
        } else {
            continue;
        }
        ada->epi = epi;
        fused_skip_[i + 1] = 1;
        ++i;
    }
#endif
    // Conv1dFp32 whose output is used only by the next node, an AddFp32 (no relu) of two distinct tensors:
    // the conv adds the other operand to its output (inside the K-split GEMM per 32-row block while it is in cache,
    // else right after the conv), and the add runs in place on the conv's slot as a no-op. a + b == b + a exactly.
    fused_res_idx_.assign(nodes.size(), -1);
    const char* env2 = kt_dev_getenv("KITTEN_FP32_FUSE_ADD");
    if (env2 && env2[0] == '0') return;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        if (fused_skip_[i] || fused_skip_[i + 1]) continue;
        const GraphNode& a = nodes[i];
        const GraphNode& b = nodes[i + 1];
        if (a.outputs.size() != 1 || b.outputs.size() != 1 || b.inputs.size() != 2) continue;
        if (b.inputs[0] == b.inputs[1]) continue;
        int idx = b.inputs[0] == a.outputs[0] ? 0 : (b.inputs[1] == a.outputs[0] ? 1 : -1);
        if (idx < 0 || uses[a.outputs[0]] != 1 || is_graph_output(a.outputs[0])) continue;
        auto ia = by_name.find(a.name), ib = by_name.find(b.name);
        if (ia == by_name.end() || ib == by_name.end()) continue;
        auto* conv = dynamic_cast<Conv1dFp32*>(ia->second);
        auto* add = dynamic_cast<AddFp32*>(ib->second);
        if (!conv || !add || add->relu) continue;
#if !defined(__AVX2__) && !defined(__AVX512F__)
        if (conv->wino2_r_ == 0) continue;   // ARM: only the blocked Winograd conv adds the residual in its output transform
#endif
        fused_res_idx_[i] = (signed char)(1 - idx);
        fused_skip_[i + 1] = (char)(idx + 1);
        ++i;
    }
#endif
}

// ──────────────────────────────────────────────────────────────
// plan_inplace_convs (low-memory modes) — a Conv1dFp32 on the K-fused path with padding > 0 copies its
// whole input into the padded scratch before it computes anything, so its output can overwrite the input's slot when
// this conv is the input's last use (stride 1, groups 1, T_out == T, C_out <= C_in: the output fits). One fewer live
// [T, C] tensor at the vocoder's resblock convs; the kernel and its inputs are unchanged, so values are bit-identical.
// KITTEN_FP32_CONV_INPLACE=0 disables.
// ──────────────────────────────────────────────────────────────
void InferenceModel::plan_inplace_convs() {
    const auto& nodes = loaded_arch_.arch.nodes;
    slot_inplace_ = fused_skip_;
    slot_inplace_.resize(nodes.size(), 0);
    const char* env = kt_dev_getenv("KITTEN_FP32_CONV_INPLACE");
    if (env && env[0] == '0') return;
    if (!active_backend() || active_backend()->mem_space != MemSpace::Host) return;
    std::unordered_map<std::string, ILayer*> by_name;
    for (auto& l : loaded_arch_.layers) if (l) by_name[l->name] = l.get();
    std::unordered_map<std::string, int> last_use;
    for (int i = 0; i < (int)nodes.size(); ++i) for (const auto& t : nodes[i].inputs) last_use[t] = i;
    std::unordered_set<std::string> graph_inputs;
    for (const auto& inp : loaded_arch_.arch.all_inputs) graph_inputs.insert(inp.name);
    int n = 0, na = 0;
    const char* env_ada = kt_dev_getenv("KITTEN_FP32_ADA_INPLACE");
    const bool ada_inplace = !(env_ada && env_ada[0] == '0');
    // default on where the x86 epilogue fusions  are not compiled in (ARM). On x86 the remaining
    // unfused Adds in place made the slot plan worse (mini, fp16 weights: 239.9 -> 247.0 MiB), so it is off there.
    // KITTEN_FP32_ACT_INPLACE=1 forces on, =0 off.
    const char* env_act = kt_dev_getenv("KITTEN_FP32_ACT_INPLACE");
#if defined(__AVX2__) || defined(__AVX512F__)
    const bool act_inplace = env_act && env_act[0] == '1';
#else
    const bool act_inplace = !(env_act && env_act[0] == '0');
#endif
    int ns = 0, nadd = 0;
    for (int i = 0; i < (int)nodes.size(); ++i) {
        if (slot_inplace_[i]) continue;
        const GraphNode& a = nodes[i];
        auto it = by_name.find(a.name);
        if (it == by_name.end() || a.outputs.size() != 1 || a.inputs.empty()) continue;
        if (dynamic_cast<AdaIn1dFp32*>(it->second) && ada_inplace) {
            // AdaIN1d's passes 1-2 only read the input; its final pass reads x[t][c] then writes y[t][c] at the
            // same index (an elementwise map), so the output can overwrite the input when this is the input's last use.
            const std::string& in = a.inputs[0];
            if (last_use[in] == i && !is_graph_output(in) && !graph_inputs.count(in) &&
                (a.inputs.size() < 2 || a.inputs[1] != in)) { slot_inplace_[i] = 1; ++na; }
            continue;
        }
        if (auto* ad = dynamic_cast<AddFp32*>(it->second); ad && act_inplace && a.inputs.size() == 2) {
            // AddFp32 adds two same-size tensors elementwise (no broadcasting: N = inputs[0] numel for
            // both), so its output can overwrite whichever input has its last use here. On x86 the residual fusion fuses most of these
            // residual adds into the conv; ARM does not, so each one held an extra [T, C] tensor.
            const std::string& i0 = a.inputs[0];
            const std::string& i1 = a.inputs[1];
            if (i0 == i1) continue;
            auto ok = [&](const std::string& t) { return last_use[t] == i && !is_graph_output(t) && !graph_inputs.count(t); };
            if (ok(i0)) { slot_inplace_[i] = 1; ++nadd; }
            else if (ok(i1)) { slot_inplace_[i] = 2; ++nadd; }
            continue;
        }
        if (a.inputs.size() != 1) continue;
        if (auto* sn = dynamic_cast<SnakeFp32*>(it->second); sn && act_inplace) {
            // Snake1D is elementwise (each y[t][c] is computed from x[t][c] only and stored at the same
            // index), so it can overwrite its input when this is the input's last use. On x86 the AdaIN epilogue fusion
            // (the epilogue fusion) already removes these nodes; ARM has no fusion, so each unfused Snake held one extra [T, C] tensor.
            const std::string& in = a.inputs[0];
            if (last_use[in] == i && !is_graph_output(in) && !graph_inputs.count(in)) { slot_inplace_[i] = 1; ++ns; }
            continue;
        }
        auto* c = dynamic_cast<Conv1dFp32*>(it->second);
        if (!c || !c->kfused_path || c->padding <= 0 || c->stride != 1 || c->groups != 1) continue;
        if (2 * c->padding != c->dilation * (c->kernel_size - 1) || c->C_out > c->C_in) continue;
        const std::string& in = a.inputs[0];
        if (last_use[in] != i || is_graph_output(in) || graph_inputs.count(in)) continue;
        slot_inplace_[i] = 1;
        ++n;
    }
    if (kt_dev_getenv("KITTEN_FP16_STORE_VERBOSE")) fprintf(stderr, "in-place nodes: %d convs + %d AdaIN + %d Snake + %d Add\n", n, na, ns, nadd);
}

// ──────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────
InferenceModel::InferenceModel(const std::string& arch_path,
                           const std::string& weights_path,
                           InferenceMemoryMode memory_mode)
    : memory_mode_(memory_mode)
{
    // Cache the backend's allocator singleton once.  Both CPU and CUDA backends
    // return a pointer to a static object, so this pointer remains valid for the
    // lifetime of the process.  All Buffer deleters capture this pointer,
    // which is safe because the allocator outlives every buffer we create.
    alloc_ = active_backend()->make_allocator();

    // The conv layers pick their kernel from process-wide hints while their weights are packed, so two models that
    // load at the same time (a TTS 0.8 model and a Kitten TTS 2 tiny one) must not see each other's hints: one load
    // at a time sets the hints and packs the weights.
    static std::mutex weight_pack_mutex;
    std::unique_lock<std::mutex> pack_lock(weight_pack_mutex);
    // On ARM the low-memory modes pack kernel-3 conv1d weights for direct conv, not Winograd.
    kitten_conv1d_set_low_memory_hint(inference_memory_mode_releases_transients(memory_mode));
    kitten_conv1d_set_extreme_memory_hint(inference_memory_mode_releases_layer_scratch(memory_mode));
    loaded_arch_ = ArchLoader::load(arch_path);
    kitten_conv1d_set_tiny_hint(loaded_arch_.arch.model.rfind("kitten_tts_2_", 0) == 0);  // Kitten TTS 2 tiny models
    layer_ops_.clear();
    for (const auto& node : loaded_arch_.arch.nodes)
        layer_ops_[node.name] = op_type_name(node.op);
    WeightLoadMemoryStats load_memory_stats;
    if (Fp16WeightStore::enabled_from_env() && active_backend()->mem_space == MemSpace::Host) {
        fp16_store_ = std::make_unique<Fp16WeightStore>();
        Fp16WeightStore* st = fp16_store_.get();
        load_weights(weights_path, loaded_arch_.layers, &load_memory_stats,
                     [st](ILayer* l) { st->compress_layer(l); });
        st->compress_shared(loaded_arch_.layers);
        if (kt_dev_getenv("KITTEN_FP16_STORE_VERBOSE"))
            fprintf(stderr, "fp16 weights: %d buffers fp16 (%.1f MiB fp32 -> %.1f MiB fp16), %d kept fp32 (%.1f MiB)\n",
                    st->buffers_fp16(), st->fp32_removed() / 1048576.0, st->fp16_bytes() / 1048576.0,
                    st->buffers_kept(), st->fp32_kept() / 1048576.0);
    } else {
        load_weights(weights_path, loaded_arch_.layers, &load_memory_stats);
    }
    pack_lock.unlock();
    weight_load_staging_peak_bytes_ = load_memory_stats.peak_tensor_bytes;
    plan_fused_epilogues();
    fused_live_.assign(loaded_arch_.arch.nodes.size(), 0);
    trace_rss_ = kt_dev_getenv("KITTEN_TRACE_RSS") != nullptr;
    plan_inplace_convs();
    pool_ = BufferPool::build(GraphSlots::build(loaded_arch_.arch, &slot_inplace_),
                              loaded_arch_.arch, loaded_arch_.layers);
    update_buffers({});   // initial sizing: uses InputSpec shapes (variable dims → 1)

    for (auto& layer : loaded_arch_.layers)
        if (layer) layer->upload_weights(*alloc_);
}

void InferenceModel::set_speed(float speed) {
    if (!std::isfinite(speed) || speed <= 0.0f)
        throw std::runtime_error("speed must be a positive finite value");

    speed_ = speed;
    const float duration_scale = 1.0f / speed_;
    const size_t layer_count = std::min(loaded_arch_.layers.size(), loaded_arch_.arch.nodes.size());
    for (size_t i = 0; i < layer_count; ++i) {
        if (loaded_arch_.layers[i] &&
            loaded_arch_.arch.nodes[i].op == OpType::LengthRegulateFp32) {
            auto* layer = static_cast<LengthRegulateFp32*>(loaded_arch_.layers[i].get());
            layer->set_duration_scale(duration_scale);
        }
    }
}

// ──────────────────────────────────────────────────────────────
// GPU-aware buffer helpers
// ──────────────────────────────────────────────────────────────
static void buf_write(Allocator* alloc, void* dst, const void* src, size_t bytes) {
    if (alloc->mem_space() == MemSpace::Host)
        ::memcpy(dst, src, bytes);
    else
        alloc->copy_h2d(dst, src, bytes);
}
static void buf_read(Allocator* alloc, void* dst, const void* src, size_t bytes) {
    if (alloc->mem_space() == MemSpace::Host)
        ::memcpy(dst, src, bytes);
    else
        alloc->copy_d2h(dst, src, bytes);
}

// ──────────────────────────────────────────────────────────────
// forward — dict of named NHWC input IOTensors → dict of output IOTensors
// ──────────────────────────────────────────────────────────────
std::unordered_map<std::string, ModelIOTensor>
InferenceModel::forward(const std::unordered_map<std::string, ModelIOTensor>& inputs,
                        int step_pos, bool is_decode)
{
    reset_forward_memory_accounting();

    // ── Grow buffers to fit these inputs (no-op if already large enough) ──
    update_buffers(inputs);
    record_engine_transient_peak();

    // ── Copy inputs into device/host buffers ──────────────────
    for (const auto& inp : loaded_arch_.arch.all_inputs) {
        auto it = inputs.find(inp.name);
        if (it == inputs.end()) continue;
        const ModelIOTensor& tensor = it->second;
        int slot = pool_.slots.slot_map.at(inp.name);

        // NHWC compact: last dim = features (d0), first = primary (d1), second = secondary (d2)
        const Shape shape = spec_to_shape(tensor.shape);
        if (inference_memory_mode_releases_transients(memory_mode_))
            pool_.grow(slot, tensor.numel(), *alloc_);
        buf_write(alloc_, pool_.raw_ptr(slot), tensor.data, tensor.nbytes());
        pool_.slot_shapes[slot] = shape;
        remember_active_tensor_bytes(inp.name, tensor.nbytes());
    }
    record_manual_transient_peak(active_tensor_total_bytes_, 0, 0);
    record_engine_transient_peak();

    ScratchPads scratch = scratch_.make_pads();
    scratch.step_pos  = step_pos;
    scratch.is_decode = is_decode;
    execute_graph(scratch);

    // ── Collect named outputs ─────────────────────────────────
    std::unordered_map<std::string, ModelIOTensor> result;
    const bool track_output_copies = memory_report_enabled_;
    size_t output_copy_bytes = 0;
    for (const auto& oname : loaded_arch_.arch.output_names) {
        auto slot_it = pool_.slots.slot_map.find(oname);
        if (slot_it == pool_.slots.slot_map.end()) continue;
        int slot = slot_it->second;
        const Shape& output_shape = pool_.slot_shapes[slot];
        const size_t out_bytes = shape_bytes(output_shape, pool_.slot_dtype[slot]);
        ModelIOTensor output_tensor;
        output_tensor.shape = shape_to_compact(output_shape);
        output_tensor.dtype = pool_.slot_dtype[slot].dtype_str();
        output_tensor.owned_.resize(out_bytes);
        if (track_output_copies) output_copy_bytes += out_bytes;
        buf_read(alloc_, output_tensor.owned_.data(), pool_.buffers[slot].ptr(), out_bytes);
        output_tensor.data = output_tensor.owned_.data();
        result[oname] = std::move(output_tensor);
    }
    if (track_output_copies) {
        last_output_copy_bytes_ = output_copy_bytes;
        peak_output_copy_bytes_ = std::max(peak_output_copy_bytes_, output_copy_bytes);
    }
    if (inference_memory_mode_releases_transients(memory_mode_)) {
        alloc_->sync();
        release_transient_buffers(true);
        // hand the freed activation memory back to the OS between requests. The released tensors
        // smaller than glibc's (dynamic) mmap threshold live in the heap; free() keeps their pages resident, so the
        // next request's peak sits on top of up to ~60 MiB of retained pages. malloc_trim(0) madvises every free
        // page in the heap (not only the top). Values are untouched (lossless). KITTEN_MALLOC_TRIM=0 turns it off.
        static const bool trim = [] { const char* e = kt_dev_getenv("KITTEN_MALLOC_TRIM"); return !e || e[0] != '0'; }();
        if (trim) {
#if defined(__GLIBC__)
            malloc_trim(0);
#elif defined(__APPLE__)
            malloc_zone_pressure_relief(nullptr, 0);
#endif
        }
    }
    return result;
}

// ──────────────────────────────────────────────────────────────
// reset_state — allocate / reset KV caches in all stateful layers
// ──────────────────────────────────────────────────────────────
void InferenceModel::reset_state(int max_seq_len) {
    for (auto& layer : loaded_arch_.layers)
        if (layer && layer->is_stateful())
            layer->reset_state(max_seq_len, *alloc_);
}

ModelMemoryReport InferenceModel::memory_report() const {
    ModelMemoryReport report;
    report.model_name = loaded_arch_.arch.model;
    report.backend_name = active_backend() && active_backend()->name ? active_backend()->name : "unknown";
    report.backend_mem_space = alloc_ ? alloc_->mem_space() : MemSpace::Host;
    report.memory_mode = memory_mode_;
    report.weight_load_staging_peak_bytes = weight_load_staging_peak_bytes_;
    report.last_output_copy_bytes = last_output_copy_bytes_;
    report.peak_output_copy_bytes = peak_output_copy_bytes_;
    report.activation_slot_count = pool_.slots.num_slots;
    report.manual_min_transient_peak_bytes = manual_min_transient_peak_bytes_;
    report.manual_min_activation_peak_bytes = manual_min_activation_peak_bytes_;
    report.manual_min_scratch_peak_bytes = manual_min_scratch_peak_bytes_;
    report.manual_min_staging_peak_bytes = manual_min_staging_peak_bytes_;
    report.engine_transient_peak_bytes = engine_transient_peak_bytes_;
    report.engine_transient_over_min_bytes =
        engine_transient_peak_bytes_ > manual_min_transient_peak_bytes_
            ? engine_transient_peak_bytes_ - manual_min_transient_peak_bytes_
            : 0;

    for (const auto& b : pool_.buffers) {
        if (!b.valid()) continue;
        report.activation_slots_bytes += b.bytes();
        report.active_activation_slots += 1;
    }
    for (const auto& b : scratch_.buffers)
        report.scratch_buffers_bytes += buf_bytes(b);
    for (const auto& b : h_stage_ins_)
        report.staging_buffers_bytes += buf_bytes(b);
    report.staging_buffers_bytes += buf_bytes(h_stage_out_);
    for (const auto& b : h_stage_scratch_)
        report.staging_buffers_bytes += buf_bytes(b);

    for (const auto& layer : loaded_arch_.layers) {
        LayerMemoryUsage usage = layer_memory_usage(layer.get());
        report.persistent_weights_bytes += usage.persistent_weights;
        report.layer_local_scratch_bytes += usage.local_scratch;
        report.persistent_state_bytes += usage.persistent_state;
    }
    return report;
}

void InferenceModel::print_memory_report(const char* context, FILE* out) const {
    if (!out) out = stdout;
    const ModelMemoryReport r = memory_report();
    const char* ctx = (context && context[0]) ? context : "-";
    fprintf(out,
            "MEMORY_REPORT model=%s context=%s backend=%s total_current_mb=%.3f "
            "memory_mode=%s "
            "persistent_weights_mb=%.3f activation_slots_mb=%.3f scratch_buffers_mb=%.3f "
            "layer_local_scratch_mb=%.3f persistent_state_mb=%.3f staging_buffers_mb=%.3f "
            "last_output_copy_mb=%.3f peak_output_copy_mb=%.3f weight_load_staging_peak_mb=%.3f "
            "manual_min_transient_peak_mb=%.3f manual_min_activation_peak_mb=%.3f "
            "manual_min_scratch_peak_mb=%.3f manual_min_staging_peak_mb=%.3f "
            "engine_transient_peak_mb=%.3f engine_over_manual_mb=%.3f "
            "activation_slots=%d active_activation_slots=%d\n",
            r.model_name.c_str(), ctx, r.backend_name.c_str(), mib(r.current_tracked_bytes()),
            inference_memory_mode_name(r.memory_mode),
            mib(r.persistent_weights_bytes), mib(r.activation_slots_bytes),
            mib(r.scratch_buffers_bytes), mib(r.layer_local_scratch_bytes),
            mib(r.persistent_state_bytes), mib(r.staging_buffers_bytes),
            mib(r.last_output_copy_bytes), mib(r.peak_output_copy_bytes),
            mib(r.weight_load_staging_peak_bytes),
            mib(r.manual_min_transient_peak_bytes),
            mib(r.manual_min_activation_peak_bytes),
            mib(r.manual_min_scratch_peak_bytes),
            mib(r.manual_min_staging_peak_bytes),
            mib(r.engine_transient_peak_bytes),
            mib(r.engine_transient_over_min_bytes),
            r.activation_slot_count, r.active_activation_slots);

    fprintf(out, "\n=== Memory report: %s (%s, memory_mode=%s) ===\n",
            r.model_name.c_str(), ctx, inference_memory_mode_name(r.memory_mode));
    fprintf(out, "Current tracked memory after the latest forward pass:\n");
    fprintf(out, "  persistent weights        %9.2f MiB\n", mib(r.persistent_weights_bytes));
    fprintf(out, "  activation slots          %9.2f MiB  (%d/%d slots allocated)\n",
            mib(r.activation_slots_bytes), r.active_activation_slots, r.activation_slot_count);
    fprintf(out, "  shared scratch buffers    %9.2f MiB\n", mib(r.scratch_buffers_bytes));
    fprintf(out, "  layer-local scratch       %9.2f MiB\n", mib(r.layer_local_scratch_bytes));
    fprintf(out, "  persistent state          %9.2f MiB\n", mib(r.persistent_state_bytes));
    fprintf(out, "  host staging buffers      %9.2f MiB\n", mib(r.staging_buffers_bytes));
    fprintf(out, "  last output copies        %9.2f MiB  (peak %.2f MiB)\n",
            mib(r.last_output_copy_bytes), mib(r.peak_output_copy_bytes));
    fprintf(out, "  current tracked total     %9.2f MiB\n", mib(r.current_tracked_bytes()));
    fprintf(out, "Largest one-tensor weight-load staging buffer: %.2f MiB\n\n",
            mib(r.weight_load_staging_peak_bytes));
    fprintf(out, "Latest forward transient peak accounting:\n");
    fprintf(out, "  manual minimum peak       %9.2f MiB  (activations %.2f, scratch %.2f, staging %.2f)\n",
            mib(r.manual_min_transient_peak_bytes),
            mib(r.manual_min_activation_peak_bytes),
            mib(r.manual_min_scratch_peak_bytes),
            mib(r.manual_min_staging_peak_bytes));
    fprintf(out, "  engine allocated peak     %9.2f MiB\n",
            mib(r.engine_transient_peak_bytes));
    fprintf(out, "  engine over manual        %9.2f MiB\n\n",
            mib(r.engine_transient_over_min_bytes));

    struct Entry {
        std::string name;
        std::string op;
        LayerMemoryUsage usage;
        size_t total = 0;
    };
    std::vector<Entry> entries;
    entries.reserve(loaded_arch_.layers.size());
    for (const auto& layer : loaded_arch_.layers) {
        if (!layer) continue;
        LayerMemoryUsage usage = layer_memory_usage(layer.get());
        size_t total = usage.persistent_weights + usage.local_scratch + usage.persistent_state;
        if (total == 0) continue;
        auto op_it = layer_ops_.find(layer->name);
        entries.push_back({layer->name, op_it == layer_ops_.end() ? "" : op_it->second,
                           usage, total});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) {
                  if (a.total != b.total) return a.total > b.total;
                  return a.name < b.name;
              });
    const size_t limit = std::min<size_t>(8, entries.size());
    if (limit > 0) {
        fprintf(out, "Top layer-owned memory holders:\n");
        for (size_t i = 0; i < limit; ++i) {
            const Entry& e = entries[i];
            fprintf(out,
                    "MEMORY_LAYER model=%s context=%s rank=%zu name=%s op=%s total_mb=%.3f "
                    "persistent_weights_mb=%.3f layer_local_scratch_mb=%.3f persistent_state_mb=%.3f\n",
                    r.model_name.c_str(), ctx, i + 1, e.name.c_str(),
                    e.op.empty() ? "-" : e.op.c_str(), mib(e.total),
                    mib(e.usage.persistent_weights), mib(e.usage.local_scratch),
                    mib(e.usage.persistent_state));
            fprintf(out, "  %zu. %-36s %-18s %9.2f MiB  weights %8.2f  local scratch %8.2f  state %8.2f\n",
                    i + 1, e.name.c_str(), e.op.empty() ? "-" : e.op.c_str(),
                    mib(e.total), mib(e.usage.persistent_weights),
                    mib(e.usage.local_scratch), mib(e.usage.persistent_state));
        }
        fprintf(out, "\n");
    }
}

void InferenceModel::reset_profile() {
    op_ms_.clear();
    layer_ms_.clear();
}

void InferenceModel::release_transient_buffers(bool release_activations) {
    if (release_activations) {
        for (auto& b : pool_.buffers)
            b.release();
        for (auto& sh : pool_.slot_shapes)
            sh = {};
    }
    for (auto& b : scratch_.buffers)
        b.release();
    for (auto& b : h_stage_ins_)
        b.release();
    h_stage_out_.release();
    for (auto& b : h_stage_scratch_)
        b.release();
}

void InferenceModel::set_memory_mode(InferenceMemoryMode mode) {
    if (memory_mode_ == mode) return;
    memory_mode_ = mode;
    if (inference_memory_mode_releases_transients(memory_mode_))
        release_transient_buffers(true);
}
