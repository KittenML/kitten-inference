// Optional Qualcomm NPU path of the C API (kt08_npu_*, include/kitten_tts08.h) for Kitten TTS 2 mini, nano and pico.
//
// Split of one chunk (the same computation as kt08_synthesize_tiny):
//   CPU  "front" graph: ALBERT, text encoder, durations, length regulation, F0 and energy, harmonic source + STFT
//   NPU  decoder + vocoder body (the conv_post output), from a QNN HTP context binary compiled ahead of time for the
//        phone's chip by engines/lite/mobile/npu/prepare_npu.py, in fixed lengths ("buckets", e.g. 160 frames = 4 s)
//   CPU  "post" graph: magnitude activation, sin, inverse STFT; then the package's chunk finishing.
// The CPU graphs are subsets of the converter's engine graph (same node names, same weights file).
//
// Body inputs (names fixed by prepare_npu.py), for a bucket of B frames at 40 per second and a chunk of L <= B frames:
//   asr [hidden, B], f0 [2B], n [2B], har [22, 120B+1], style [128]: zero past the chunk's length;
//   m40 [B], m80 [2B], m800 [20B], m4800 [120B+1]: 1 inside the chunk, 0 past it;
//   s40, s80, s800 [1] = B / L and s4800 [1] = (120B+1) / (120L+1): rescale padded means to the chunk's means.
// Output: post [22, 120B+1] (or [120B+1, 22]); the first 120L+1 frames are the chunk's.
//
// Qualcomm's libraries are loaded at run time (dlopen), so the library links nothing of theirs; building this file
// needs only the QNN SDK's headers (KITTEN_QNN_SDK). Without them the functions return KT08_ERR_UNSUPPORTED.
#include "kitten_tts08.h"
#include "capi_internal.hpp"

#include <cstring>
#include <string>

#if defined(KITTEN_HAVE_QNN)

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "utils/json.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using kt08_internal::set_error;

constexpr int kHarCh = 22;      // n_fft + 2 channels of the harmonic source's STFT (all three models: n_fft 20)
constexpr int kStyleDec = 128;  // decoder half of a voice row

// ---------------------------------------------------------------- QNN runtime (one per process, shared)
struct QnnRuntime {
    void* lib_htp = nullptr;
    void* lib_sys = nullptr;
    QNN_INTERFACE_VER_TYPE q{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sq{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    int users = 0;
};

std::mutex& qnn_mutex() {
    static std::mutex m;
    return m;
}
QnnRuntime& qnn() {
    static QnnRuntime r;
    return r;
}

std::string join(const std::string& dir, const char* name) {
    if (dir.empty()) return name;
    return dir.back() == '/' ? dir + name : dir + "/" + name;
}

// Opens Qualcomm's libraries and the HTP backend (first user), or takes another reference. Caller holds qnn_mutex().
bool qnn_acquire(const std::string& dir, std::string& err) {
    QnnRuntime& r = qnn();
    if (r.users > 0) { ++r.users; return true; }
#if defined(__ANDROID__)
    // The NPU side of the runtime (libQnnHtpV<NN>Skel.so) is found through ADSP_LIBRARY_PATH. When the app gives a
    // folder and has not set the variable itself, look there first, then in the system's DSP folders.
    if (!dir.empty() && !std::getenv("ADSP_LIBRARY_PATH")) {
        const std::string v = dir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
        setenv("ADSP_LIBRARY_PATH", v.c_str(), 0);
    }
#endif
    r.lib_htp = dlopen(join(dir, "libQnnHtp.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!r.lib_htp) { err = std::string("cannot load Qualcomm's libQnnHtp.so: ") + dlerror(); return false; }
    r.lib_sys = dlopen(join(dir, "libQnnSystem.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!r.lib_sys) {
        err = std::string("cannot load Qualcomm's libQnnSystem.so: ") + dlerror();
        dlclose(r.lib_htp); r.lib_htp = nullptr;
        return false;
    }
    using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    auto gp = reinterpret_cast<GetProviders>(dlsym(r.lib_htp, "QnnInterface_getProviders"));
    auto gsp = reinterpret_cast<GetSysProviders>(dlsym(r.lib_sys, "QnnSystemInterface_getProviders"));
    bool ok = false, sys_ok = false;
    const QnnInterface_t** pl = nullptr;
    uint32_t np = 0;
    if (gp && gp(&pl, &np) == QNN_SUCCESS)
        for (uint32_t i = 0; i < np && !ok; ++i)
            if (pl[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                pl[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
                r.q = pl[i]->QNN_INTERFACE_VER_NAME;
                ok = true;
            }
    const QnnSystemInterface_t** spl = nullptr;
    uint32_t nsp = 0;
    if (gsp && gsp(&spl, &nsp) == QNN_SUCCESS)
        for (uint32_t i = 0; i < nsp && !sys_ok; ++i)
            if (spl[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
                spl[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
                r.sq = spl[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
                sys_ok = true;
            }
    if (!ok || !sys_ok) {
        err = "Qualcomm's QNN libraries are too old or too new for this build (QNN API " +
              std::to_string(QNN_API_VERSION_MAJOR) + "." + std::to_string(QNN_API_VERSION_MINOR) + " needed)";
    } else if (r.q.backendCreate(nullptr, nullptr, &r.backend) != QNN_SUCCESS || !r.backend) {
        err = "the QNN HTP backend did not start (no Qualcomm NPU, or its driver is not reachable)";
        ok = false;
    } else if (r.q.deviceCreate && r.q.deviceCreate(nullptr, nullptr, &r.device) != QNN_SUCCESS) {
        r.device = nullptr;  // the HTP backend also runs without an explicit device handle
    }
    if (!ok || !sys_ok) {
        if (r.backend && r.q.backendFree) r.q.backendFree(r.backend);
        r.backend = nullptr;
        dlclose(r.lib_sys); dlclose(r.lib_htp);
        r.lib_sys = r.lib_htp = nullptr;
        return false;
    }
    r.users = 1;
    return true;
}

// Drops a reference; the last one frees the backend. The libraries stay loaded: unloading Qualcomm's libraries while
// their worker threads finish can crash the process, and a later kt08_npu_load reuses them.
void qnn_release() {
    QnnRuntime& r = qnn();
    if (r.users <= 0 || --r.users > 0) return;
    if (r.device && r.q.deviceFree) r.q.deviceFree(r.device);
    if (r.backend && r.q.backendFree) r.q.backendFree(r.backend);
    r.device = nullptr;
    r.backend = nullptr;
}

// ---------------------------------------------------------------- tensors
struct TInfo {
    std::string name;
    std::vector<uint32_t> dims;
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    Qnn_DataType_t dt = QNN_DATATYPE_UNDEFINED;
    bool quant = false;
    float scale = 1.0f;
    int32_t offset = 0;
    size_t numel = 1, esize = 0;
};

size_t dt_size(Qnn_DataType_t d) {
    switch (d) {
        case QNN_DATATYPE_FLOAT_32: return 4;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: return 2;
        case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
        default: return 0;  // other types are not used by the body graphs
    }
}

template <class F> auto tensor_v(Qnn_Tensor_t& t, F f) { return t.version == QNN_TENSOR_VERSION_2 ? f(t.v2) : f(t.v1); }

TInfo tinfo(const Qnn_Tensor_t& src) {
    TInfo ti;
    ti.t = src;
    tensor_v(ti.t, [&](auto& v) {
        ti.name = v.name ? v.name : "";
        ti.dt = v.dataType;
        ti.dims.assign(v.dimensions, v.dimensions + v.rank);
        if (v.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&
            v.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            ti.quant = true;
            ti.scale = v.quantizeParams.scaleOffsetEncoding.scale;
            ti.offset = v.quantizeParams.scaleOffsetEncoding.offset;
        }
        return 0;
    });
    for (uint32_t d : ti.dims) ti.numel *= d;
    ti.esize = dt_size(ti.dt);
    return ti;
}

// A tensor description for one execute call: names/dims point into ti (which outlives the call), data into buf.
Qnn_Tensor_t tensor_for_call(const TInfo& ti, void* buf, size_t bytes) {
    Qnn_Tensor_t t = ti.t;
    tensor_v(t, [&](auto& v) {
        v.name = ti.name.c_str();
        v.dimensions = const_cast<uint32_t*>(ti.dims.data());
        v.memType = QNN_TENSORMEMTYPE_RAW;
        v.clientBuf.data = buf;
        v.clientBuf.dataSize = (uint32_t)bytes;
        return 0;
    });
    if (t.version == QNN_TENSOR_VERSION_2) {
        t.v2.isDynamicDimensions = nullptr;  // pointed into the binary info, freed after loading
        t.v2.sparseParams = QNN_SPARSE_PARAMS_INIT;
    }
    return t;
}

// Position of (channel c, step t) in a tensor holding C x T values channel-first or step-first (unit axes ignored).
struct Layout {
    bool cfirst = true;
    int C = 1, T = 1;
};
bool layout_of(const TInfo& ti, int C, int T, Layout& L) {
    L.C = C; L.T = T; L.cfirst = true;
    if ((size_t)C * (size_t)T != ti.numel) return false;
    if (C == 1 || T == 1) return true;
    std::vector<uint32_t> nz;
    for (uint32_t d : ti.dims) if (d != 1) nz.push_back(d);
    if (nz.size() == 2 && (int)nz[0] == C && (int)nz[1] == T) return true;
    if (nz.size() == 2 && (int)nz[0] == T && (int)nz[1] == C) { L.cfirst = false; return true; }
    return false;
}
inline size_t lidx(const Layout& L, int c, int t) { return L.cfirst ? (size_t)c * L.T + t : (size_t)t * L.C + c; }

uint16_t f32_to_f16(float f) {  // round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            x = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e + 127 - 15) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// Writes C x T values get(c, t) into a graph input in its type and layout.
template <class Get> bool put_tensor(const TInfo& ti, std::vector<uint8_t>& buf, int C, int T, Get get, std::string& err) {
    Layout L;
    if (!layout_of(ti, C, T, L) || ti.esize == 0) {
        err = "NPU graph input '" + ti.name + "' has an unexpected shape or type";
        return false;
    }
    buf.assign(ti.numel * ti.esize, 0);
    switch (ti.dt) {
        case QNN_DATATYPE_FLOAT_32: {
            float* p = reinterpret_cast<float*>(buf.data());
            for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t) p[lidx(L, c, t)] = get(c, t);
            return true;
        }
        case QNN_DATATYPE_FLOAT_16: {
            uint16_t* p = reinterpret_cast<uint16_t*>(buf.data());
            for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t) p[lidx(L, c, t)] = f32_to_f16(get(c, t));
            return true;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_8: {
            if (!ti.quant || !(ti.scale > 0)) break;
            const float inv = 1.0f / ti.scale;
            const long hi = ti.dt == QNN_DATATYPE_UFIXED_POINT_16 ? 65535 : 255;
            for (int c = 0; c < C; ++c)
                for (int t = 0; t < T; ++t) {
                    const long q = std::min(hi, std::max(0L, (long)std::lrint(get(c, t) * inv) - ti.offset));
                    if (hi == 255) buf[lidx(L, c, t)] = (uint8_t)q;
                    else reinterpret_cast<uint16_t*>(buf.data())[lidx(L, c, t)] = (uint16_t)q;
                }
            return true;
        }
        default: break;
    }
    err = "NPU graph input '" + ti.name + "' has an unsupported data type";
    return false;
}

float get_value(const TInfo& ti, const uint8_t* buf, size_t i) {
    switch (ti.dt) {
        case QNN_DATATYPE_FLOAT_32: return reinterpret_cast<const float*>(buf)[i];
        case QNN_DATATYPE_FLOAT_16: return f16_to_f32(reinterpret_cast<const uint16_t*>(buf)[i]);
        case QNN_DATATYPE_UFIXED_POINT_16: return (float)((int32_t)reinterpret_cast<const uint16_t*>(buf)[i] + ti.offset) * ti.scale;
        case QNN_DATATYPE_SFIXED_POINT_16: return (float)((int32_t)reinterpret_cast<const int16_t*>(buf)[i] + ti.offset) * ti.scale;
        case QNN_DATATYPE_UFIXED_POINT_8: return (float)((int32_t)buf[i] + ti.offset) * ti.scale;
        case QNN_DATATYPE_SFIXED_POINT_8: return (float)((int32_t)reinterpret_cast<const int8_t*>(buf)[i] + ti.offset) * ti.scale;
        default: return 0.0f;
    }
}

// ---------------------------------------------------------------- compiled body graphs
struct BodyGraph {
    int frames = 0;  // bucket B
    Qnn_GraphHandle_t graph = nullptr;
    std::vector<TInfo> in, out;
};

struct GraphIO {
    std::vector<TInfo> in, out;
};

// One context binary: one graph per length, or several lengths in one context whose graphs share their weights.
struct BodyContext {
    std::string path;
    Qnn_ContextHandle_t ctx = nullptr;
    std::vector<std::pair<std::string, GraphIO>> graphs;  // name -> I/O descriptions
};

// Reads a context binary and creates its context. group: contexts of one object share one spill-fill buffer
// (sf_bytes) when the backend allows it, which matters on phones with little NPU memory.
bool load_context(BodyContext& C, Qnn_ContextHandle_t* group, uint64_t sf_bytes, std::string& err) {
    QnnRuntime& r = qnn();
    const std::string& path = C.path;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "cannot open " + path; return false; }
    const std::streamoff n = f.tellg();
    if (n <= 0) { err = path + " is empty"; return false; }
    std::vector<char> buf((size_t)n);
    f.seekg(0);
    if (!f.read(buf.data(), n)) { err = "cannot read " + path; return false; }
    QnnSystemContext_Handle_t sc = nullptr;
    if (r.sq.systemContextCreate(&sc) != QNN_SUCCESS) { err = "QNN system context failed"; return false; }
    const QnnSystemContext_BinaryInfo_t* bi = nullptr;
    Qnn_ContextBinarySize_t bis = 0;
    bool ok = r.sq.systemContextGetBinaryInfo(sc, buf.data(), (uint64_t)n, &bi, &bis) == QNN_SUCCESS && bi;
    if (ok) {
        uint32_t ng = 0;
        QnnSystemContext_GraphInfo_t* gs = nullptr;
        if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { ng = bi->contextBinaryInfoV1.numGraphs; gs = bi->contextBinaryInfoV1.graphs; }
        else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { ng = bi->contextBinaryInfoV2.numGraphs; gs = bi->contextBinaryInfoV2.graphs; }
        else { ng = bi->contextBinaryInfoV3.numGraphs; gs = bi->contextBinaryInfoV3.graphs; }
        ok = ng >= 1 && gs;
        for (uint32_t k = 0; ok && k < ng; ++k) {
            std::pair<std::string, GraphIO> g;
            auto take = [&](auto& gi) {
                g.first = gi.graphName ? gi.graphName : "";
                for (uint32_t i = 0; i < gi.numGraphInputs; ++i) g.second.in.push_back(tinfo(gi.graphInputs[i]));
                for (uint32_t i = 0; i < gi.numGraphOutputs; ++i) g.second.out.push_back(tinfo(gi.graphOutputs[i]));
                return 0;
            };
            if (gs[k].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) take(gs[k].graphInfoV1);
            else if (gs[k].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) take(gs[k].graphInfoV2);
            else take(gs[k].graphInfoV3);
            ok = g.second.out.size() == 1;
            C.graphs.push_back(std::move(g));
        }
    }
    r.sq.systemContextFree(sc);  // names and dims were copied into TInfo
    if (!ok) { err = path + " is not an NPU graph made by prepare_npu.py"; return false; }

    QnnHtpContext_CustomConfig_t hc;
    std::memset(&hc, 0, sizeof hc);
    QnnContext_Config_t cc;
    std::memset(&cc, 0, sizeof cc);
    const QnnContext_Config_t* cfgs[2] = {nullptr, nullptr};
    if (group && sf_bytes > 0) {
        hc.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
        hc.groupRegistration.firstGroupHandle = *group;
        hc.groupRegistration.maxSpillFillBuffer = sf_bytes;
        cc.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
        cc.customConfig = &hc;
        cfgs[0] = &cc;
    }
    Qnn_ErrorHandle_t e = r.q.contextCreateFromBinary(r.backend, r.device, cfgs[0] ? cfgs : nullptr, buf.data(),
                                                      (Qnn_ContextBinarySize_t)n, &C.ctx, nullptr);
    if (e != QNN_SUCCESS && cfgs[0]) {  // shared spill-fill not supported here: load on its own
        e = r.q.contextCreateFromBinary(r.backend, r.device, nullptr, buf.data(), (Qnn_ContextBinarySize_t)n, &C.ctx,
                                        nullptr);
        group = nullptr;
    }
    if (e != QNN_SUCCESS || !C.ctx) {
        err = "the NPU could not load " + path + " (QNN error " + std::to_string((long)e) +
              "): it was made for another chip or QNN version, or the NPU is out of memory";
        C.ctx = nullptr;
        return false;
    }
    if (group && !*group) *group = C.ctx;
    return true;
}

// The graph `name` (empty: the context's only graph) of a loaded context.
bool retrieve_graph(BodyContext& C, const std::string& name, BodyGraph& G, std::string& err) {
    const std::pair<std::string, GraphIO>* g = nullptr;
    for (const auto& x : C.graphs)
        if (name.empty() ? C.graphs.size() == 1 : x.first == name) g = &x;
    if (!g) { err = "graph '" + name + "' not found in " + C.path; return false; }
    if (qnn().q.graphRetrieve(C.ctx, g->first.c_str(), &G.graph) != QNN_SUCCESS) {
        err = "graph '" + g->first + "' not found in " + C.path;
        return false;
    }
    G.in = g->second.in;
    G.out = g->second.out;
    return true;
}

void set_burst(bool on, uint32_t& power_id) {
    QnnRuntime& r = qnn();
    if (!on || !r.q.deviceGetInfrastructure) return;
    QnnDevice_Infrastructure_t inf = nullptr;
    if (r.q.deviceGetInfrastructure(&inf) != QNN_SUCCESS || !inf) return;
    auto* hi = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(inf);
    if (hi->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    QnnHtpDevice_PerfInfrastructure_t p = hi->perfInfra;
    if (!p.createPowerConfigId || p.createPowerConfigId(0, 0, &power_id) != QNN_SUCCESS) { power_id = 0; return; }
    QnnHtpPerfInfrastructure_PowerConfig_t dc;
    std::memset(&dc, 0, sizeof dc);
    dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& d = dc.dcvsV3Config;
    d.contextId = power_id;
    d.setDcvsEnable = 1; d.dcvsEnable = 0;
    d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    d.setSleepLatency = 1; d.sleepLatency = 40;
    d.setSleepDisable = 1; d.sleepDisable = 1;
    d.setBusParams = 1;
    d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    d.setCoreParams = 1;
    d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    QnnHtpPerfInfrastructure_PowerConfig_t lat;
    std::memset(&lat, 0, sizeof lat);
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    lat.rpcControlLatencyConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t* cfg[3] = {&dc, &lat, nullptr};
    if (p.setPowerConfig(power_id, cfg) != QNN_SUCCESS) {
        if (p.destroyPowerConfigId) p.destroyPowerConfigId(power_id);
        power_id = 0;
    }
}

void clear_burst(uint32_t power_id) {
    QnnRuntime& r = qnn();
    if (!power_id || !r.q.deviceGetInfrastructure) return;
    QnnDevice_Infrastructure_t inf = nullptr;
    if (r.q.deviceGetInfrastructure(&inf) != QNN_SUCCESS || !inf) return;
    auto* hi = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(inf);
    if (hi->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF && hi->perfInfra.destroyPowerConfigId)
        hi->perfInfra.destroyPowerConfigId(power_id);
}

}  // namespace

struct kt08_npu {
    kt08_model* front = nullptr;
    kt08_model* post = nullptr;
    kt08_model* fallback = nullptr;  // owned by the app
    std::vector<BodyContext> contexts;
    std::vector<BodyGraph> graphs;   // ascending frames
    uint32_t power_id = 0;
    bool runtime = false;            // holds a QNN runtime reference
    ~kt08_npu() {
        std::lock_guard<std::mutex> lock(qnn_mutex());
        QnnRuntime& r = qnn();
        graphs.clear();
        for (auto& c : contexts)
            if (c.ctx && r.q.contextFree) r.q.contextFree(c.ctx, nullptr);
        contexts.clear();
        if (runtime) {
            clear_burst(power_id);
            qnn_release();
        }
        if (front) kt08_model_free(front);
        if (post) kt08_model_free(post);
    }
};

extern "C" {

void kt08_npu_options_init(kt08_npu_options* options) {
    if (!options) return;
    std::memset(options, 0, sizeof *options);
    options->struct_size = sizeof *options;
    options->high_performance = 1;
}

kt08_status kt08_npu_load(const kt08_npu_options* options, kt08_npu** out) {
    if (!out) return set_error(KT08_ERR_ARGUMENT, "out is NULL");
    *out = nullptr;
    if (!options) return set_error(KT08_ERR_ARGUMENT, "options are required (model_dir and weights_path)");
    kt08_npu_options opt;
    kt08_npu_options_init(&opt);
    const size_t n = std::min<size_t>(options->struct_size ? options->struct_size : sizeof opt, sizeof opt);
    if (n < sizeof(uint32_t)) return set_error(KT08_ERR_ARGUMENT, "options->struct_size is too small");
    std::memcpy(&opt, options, n);
    if (!opt.model_dir || !*opt.model_dir || !opt.weights_path || !*opt.weights_path)
        return set_error(KT08_ERR_ARGUMENT, "options.model_dir and options.weights_path are required");
    if (opt.threads < 0) return set_error(KT08_ERR_ARGUMENT, "threads must be 0 (default) or more");
    try {
        const std::string dir = opt.model_dir;
        JsonValue man;
        {
            std::ifstream probe(join(dir, "npu.json"));
            if (!probe) return set_error(KT08_ERR_IO, "cannot open " + join(dir, "npu.json") + " (made by prepare_npu.py)");
        }
        man = json_load(join(dir, "npu.json"));
        if (!man.is_object() || !man.contains("graphs") || !man["graphs"].is_array() || man["graphs"].arr.empty())
            return set_error(KT08_ERR_IO, join(dir, "npu.json") + " lists no NPU graphs");
        auto npu = std::make_unique<kt08_npu>();
        npu->fallback = opt.cpu_fallback;
        // Declared after npu: on an early return the lock is released before npu's destructor takes it again.
        std::lock_guard<std::mutex> lock(qnn_mutex());
        std::string err;
        // Qualcomm's runtime first: on a phone without it the app learns that before any model is loaded.
        if (!qnn_acquire(opt.qnn_lib_dir ? opt.qnn_lib_dir : "", err)) return set_error(KT08_ERR_UNSUPPORTED, err);
        npu->runtime = true;
        kt08_options mo;
        kt08_options_init(&mo);
        mo.threads = opt.threads;
        const std::string front = join(dir, man.contains("front") ? man["front"].str.c_str() : "front.json");
        const std::string post = join(dir, man.contains("post") ? man["post"].str.c_str() : "post.json");
        kt08_status st = kt08_model_load(front.c_str(), opt.weights_path, &mo, &npu->front);
        if (st != KT08_OK) return st;
        kt08_options po = mo;
        po.threads = 1;  // the inverse STFT is small; one thread avoids waking the others
        st = kt08_model_load(post.c_str(), opt.weights_path, &po, &npu->post);
        if (st != KT08_OK) return st;
        const uint64_t sf = man.contains("spill_fill_bytes") ? (uint64_t)man["spill_fill_bytes"].as_double() : 0;
        Qnn_ContextHandle_t group = nullptr;
        size_t n_files = 0;
        {
            std::vector<std::string> files;
            for (const JsonValue& g : man["graphs"].arr)
                if (std::find(files.begin(), files.end(), g["context"].str) == files.end()) files.push_back(g["context"].str);
            n_files = files.size();
        }
        Qnn_ContextHandle_t* gp = (sf > 0 && n_files > 1) ? &group : nullptr;
        npu->contexts.reserve(n_files);
        for (const JsonValue& g : man["graphs"].arr) {
            BodyGraph G;
            G.frames = g["frames"].as_int();
            if (G.frames <= 0) return set_error(KT08_ERR_IO, "npu.json: a graph without frames");
            const std::string path = join(dir, g["context"].str.c_str());
            BodyContext* C = nullptr;
            for (auto& c : npu->contexts)
                if (c.path == path) C = &c;
            if (!C) {
                npu->contexts.emplace_back();
                npu->contexts.back().path = path;
                if (!load_context(npu->contexts.back(), gp, sf, err)) {
                    npu->contexts.pop_back();
                    if (npu->graphs.empty()) return set_error(KT08_ERR_UNSUPPORTED, err);
                    continue;  // a longer graph that does not fit: its chunks go to the CPU fallback
                }
                C = &npu->contexts.back();
            }
            if (!retrieve_graph(*C, g.contains("graph") ? g["graph"].str : std::string(), G, err))
                return set_error(KT08_ERR_IO, err);
            npu->graphs.push_back(std::move(G));
        }
        std::sort(npu->graphs.begin(), npu->graphs.end(),
                  [](const BodyGraph& a, const BodyGraph& b) { return a.frames < b.frames; });
        set_burst(opt.high_performance != 0, npu->power_id);
        *out = npu.release();
        return KT08_OK;
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory while loading the NPU path");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_IO, std::string("NPU model folder: ") + e.what());
    }
}

void kt08_npu_free(kt08_npu* npu) { delete npu; }

size_t kt08_npu_max_frames(const kt08_npu* npu) {
    return npu && !npu->graphs.empty() ? (size_t)npu->graphs.back().frames : 0;
}

kt08_status kt08_synthesize_tiny_npu(kt08_npu* npu, const int32_t* ids, size_t n_ids, const float* voice,
                                     size_t voice_rows, size_t style_row, float speed, int first_chunk,
                                     float** out_audio, size_t* out_samples, int* out_on_npu) {
    if (out_on_npu) *out_on_npu = 0;
    if (!out_audio || !out_samples) return set_error(KT08_ERR_ARGUMENT, "out_audio and out_samples are required");
    *out_audio = nullptr;
    *out_samples = 0;
    if (!npu || !npu->front || npu->graphs.empty()) return set_error(KT08_ERR_ARGUMENT, "npu is NULL");
    if (!ids || n_ids == 0) return set_error(KT08_ERR_ARGUMENT, "ids are required");
    if (!voice || voice_rows == 0) return set_error(KT08_ERR_ARGUMENT, "a voice is required");
    if (!(speed >= KT08_TINY_SPEED_MIN && speed <= KT08_TINY_SPEED_MAX))
        return set_error(KT08_ERR_ARGUMENT, "speed must be from 0.25 to 4 (1.0 is normal)");
    const size_t max_tokens = kt08_model_max_tokens(npu->front);
    if (max_tokens && n_ids > max_tokens)
        return set_error(KT08_ERR_TOO_LONG, "this chunk has " + std::to_string(n_ids) +
                                                " phoneme tokens, more than the model's limit of " +
                                                std::to_string(max_tokens) + "; split the text into shorter chunks");
    try {
        // CPU front: everything before the decoder.
        const size_t row = std::min(style_row, voice_rows - 1);
        const float* style = voice + row * KT08_STYLE_DIM;
        std::unordered_map<std::string, std::vector<float>> in;
        in["phoneme_ids"] = std::vector<float>(ids, ids + n_ids);
        in["style_dec"] = std::vector<float>(style, style + kStyleDec);
        in["style_pred"] = std::vector<float>(style + kStyleDec, style + KT08_STYLE_DIM);
        auto fo = kt08_internal::forward_named(npu->front, in, speed);
        auto get = [&](const char* name) -> const ModelIOTensor& {
            auto it = fo.find(name);
            if (it == fo.end()) throw std::runtime_error(std::string("the front graph has no output ") + name);
            return it->second;
        };
        const ModelIOTensor& t_asr = get("asr");
        const ModelIOTensor& t_f0 = get("t_f0");
        const ModelIOTensor& t_n = get("t_n");
        const ModelIOTensor& t_har = get("t_har");
        const ModelIOTensor& t_dur = get("t_dur");
        const int L = (int)(t_f0.numel() / 2);
        if (L <= 0 || t_asr.numel() % (size_t)L) return set_error(KT08_ERR_RUNTIME, "unexpected front graph output sizes");
        const int H = (int)(t_asr.numel() / (size_t)L);
        const int TL = 120 * L + 1;
        if (t_har.numel() != (size_t)kHarCh * TL || t_n.numel() != (size_t)2 * L)
            return set_error(KT08_ERR_RUNTIME, "unexpected front graph output sizes");

        BodyGraph* G = nullptr;
        for (auto& g : npu->graphs)
            if (L <= g.frames) { G = &g; break; }
        if (!G) {
            if (!npu->fallback)
                return set_error(KT08_ERR_TOO_LONG, "this chunk is " + std::to_string(L) + " frames long, more than the NPU "
                                 "graphs' " + std::to_string(npu->graphs.back().frames) + " (25 ms each): give "
                                 "kt08_npu_options.cpu_fallback, or split the text into shorter chunks");
            return kt08_synthesize_tiny(npu->fallback, ids, n_ids, voice, voice_rows, style_row, speed, first_chunk,
                                        out_audio, out_samples);
        }

        // Pack the body inputs for bucket B.
        const int B = G->frames;
        const float* asr = t_asr.begin();
        const float* f0 = t_f0.begin();
        const float* nn = t_n.begin();
        const float* har = t_har.begin();
        const float sB = (float)B / (float)L, s4800 = (float)(120 * B + 1) / (float)TL;
        std::vector<std::vector<uint8_t>> ib(G->in.size());
        std::string err;
        for (size_t k = 0; k < G->in.size(); ++k) {
            const TInfo& ti = G->in[k];
            const std::string& nm = ti.name;
            auto& b = ib[k];
            bool ok;
            if (nm == "asr") ok = put_tensor(ti, b, H, B, [&](int c, int t) { return t < L ? asr[(size_t)t * H + c] : 0.0f; }, err);
            else if (nm == "f0") ok = put_tensor(ti, b, 1, 2 * B, [&](int, int t) { return t < 2 * L ? f0[t] : 0.0f; }, err);
            else if (nm == "n") ok = put_tensor(ti, b, 1, 2 * B, [&](int, int t) { return t < 2 * L ? nn[t] : 0.0f; }, err);
            else if (nm == "har") ok = put_tensor(ti, b, kHarCh, 120 * B + 1, [&](int c, int t) { return t < TL ? har[(size_t)t * kHarCh + c] : 0.0f; }, err);
            else if (nm == "style") ok = put_tensor(ti, b, kStyleDec, 1, [&](int c, int) { return style[c]; }, err);
            else if (nm == "m40") ok = put_tensor(ti, b, 1, B, [&](int, int t) { return t < L ? 1.0f : 0.0f; }, err);
            else if (nm == "m80") ok = put_tensor(ti, b, 1, 2 * B, [&](int, int t) { return t < 2 * L ? 1.0f : 0.0f; }, err);
            else if (nm == "m800") ok = put_tensor(ti, b, 1, 20 * B, [&](int, int t) { return t < 20 * L ? 1.0f : 0.0f; }, err);
            else if (nm == "m4800") ok = put_tensor(ti, b, 1, 120 * B + 1, [&](int, int t) { return t < TL ? 1.0f : 0.0f; }, err);
            else if (nm == "s40" || nm == "s80" || nm == "s800") ok = put_tensor(ti, b, 1, 1, [&](int, int) { return sB; }, err);
            else if (nm == "s4800") ok = put_tensor(ti, b, 1, 1, [&](int, int) { return s4800; }, err);
            else { err = "the NPU graph has an unknown input '" + nm + "'"; ok = false; }
            if (!ok) return set_error(KT08_ERR_RUNTIME, err);
        }
        const TInfo& to = G->out[0];
        Layout Lo;
        if (to.esize == 0 || !layout_of(to, kHarCh, 120 * B + 1, Lo))
            return set_error(KT08_ERR_RUNTIME, "the NPU graph's output has an unexpected shape or type");
        std::vector<uint8_t> ob(to.numel * to.esize, 0);

        // NPU body.
        std::vector<Qnn_Tensor_t> qi, qo;
        for (size_t k = 0; k < G->in.size(); ++k) qi.push_back(tensor_for_call(G->in[k], ib[k].data(), ib[k].size()));
        qo.push_back(tensor_for_call(to, ob.data(), ob.size()));
        const Qnn_ErrorHandle_t e = qnn().q.graphExecute(G->graph, qi.data(), (uint32_t)qi.size(), qo.data(), 1,
                                                         nullptr, nullptr);
        if (e != QNN_SUCCESS)
            return set_error(KT08_ERR_RUNTIME, "the NPU failed to run the decoder (QNN error " + std::to_string((long)e) + ")");

        // CPU post: [frames, 22] -> waveform; then the package's chunk finishing.
        std::unordered_map<std::string, std::vector<float>> pin;
        std::vector<float>& post = pin["t_gen_post"];
        post.resize((size_t)TL * kHarCh);
        for (int t = 0; t < TL; ++t)
            for (int c = 0; c < kHarCh; ++c) post[(size_t)t * kHarCh + c] = get_value(to, ob.data(), lidx(Lo, c, t));
        auto po = kt08_internal::forward_named(npu->post, pin, 1.0f);
        auto ia = po.find("t_audio");
        if (ia == po.end()) return set_error(KT08_ERR_RUNTIME, "the post graph has no t_audio output");
        const kt08_status st = kt08_internal::finish_tiny_chunk(ia->second.begin(), ia->second.numel(), t_dur.begin(),
                                                                t_dur.numel(), speed, first_chunk, out_audio, out_samples);
        if (st == KT08_OK && out_on_npu) *out_on_npu = 1;
        return st;
    } catch (const std::bad_alloc&) {
        return set_error(KT08_ERR_MEMORY, "out of memory during synthesis");
    } catch (const std::exception& e) {
        return set_error(KT08_ERR_RUNTIME, e.what());
    } catch (...) {
        return set_error(KT08_ERR_RUNTIME, "unknown error during synthesis");
    }
}

}  // extern "C"

#else  // !KITTEN_HAVE_QNN

struct kt08_npu {};

extern "C" {

void kt08_npu_options_init(kt08_npu_options* options) {
    if (!options) return;
    std::memset(options, 0, sizeof *options);
    options->struct_size = sizeof *options;
    options->high_performance = 1;
}

kt08_status kt08_npu_load(const kt08_npu_options*, kt08_npu** out) {
    if (out) *out = nullptr;
    return kt08_internal::set_error(KT08_ERR_UNSUPPORTED, "this library was built without NPU support (build it with "
                                                          "the Qualcomm QNN SDK: see the mobile README)");
}

void kt08_npu_free(kt08_npu*) {}

size_t kt08_npu_max_frames(const kt08_npu*) { return 0; }

kt08_status kt08_synthesize_tiny_npu(kt08_npu*, const int32_t*, size_t, const float*, size_t, size_t, float, int,
                                     float** out_audio, size_t* out_samples, int* out_on_npu) {
    if (out_audio) *out_audio = nullptr;
    if (out_samples) *out_samples = 0;
    if (out_on_npu) *out_on_npu = 0;
    return kt08_internal::set_error(KT08_ERR_UNSUPPORTED, "this library was built without NPU support");
}

}  // extern "C"

#endif  // KITTEN_HAVE_QNN
