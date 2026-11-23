// C API of the engine's graph models, for the NVIDIA GPU library that kitten-inference loads with ctypes when a
// Kitten TTS 2 mini, nano or pico model is asked for backend="cuda" (src/kitten_inference/tts2_tiny/_cuda.py).
// It is the Python extension's InferenceModel (python/bindings.cpp: the dict forward, set_speed) without Python, so
// one library serves every Python version, like the Kitten TTS 2 engine next to it in the package. Built only as the
// kt08_cuda shared library (CMake option KITTEN_BUILD_CUDA_LIBRARY).
//
//   kt08g_api_version()                  1
//   kt08g_init("cuda")                   0, or -1 with kt08g_last_error() saying why (no driver, old driver, no GPU)
//   kt08g_device()                       "<GPU name> (compute capability X.Y, driver CUDA A.B)"
//   kt08g_load(arch, weights)            a model, or NULL (kt08g_last_error)
//   kt08g_set_speed(model, speed)        0 / -1
//   kt08g_forward(model, n, names, data, sizes)   0 / -1; float32 inputs, shapes from the arch file as in Python
//   kt08g_output_count / _name / _data   the outputs of the last forward, valid until the next one
//   kt08g_free(model)
#include "engine/model.hpp"
#include "backend/backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define KT08G_API extern "C" __declspec(dllexport)
#else
#  define KT08G_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string t_error;

int fail(std::string msg) {
    t_error = std::move(msg);
    return -1;
}

struct GraphModel {
    std::unique_ptr<InferenceModel> model;
    std::vector<std::pair<std::string, std::vector<float>>> outputs;
};

// The CUDA 12.8 runtime is linked into this library; the driver must support it (R570 or newer).
constexpr int kNeedDriver = 12080;

}  // namespace

bool kt08_cuda_register_backend();   // backend/cuda/cuda_backend.cu (KITTEN_CUDA_LAZY_REGISTER)

KT08G_API int kt08g_api_version(void) { return 1; }

KT08G_API const char* kt08g_last_error(void) { return t_error.c_str(); }

KT08G_API int kt08g_init(const char* backend) {
    try {
        const std::string name = backend ? backend : "";
        if (name == "cuda") {
            int driver = 0;
            if (cudaDriverGetVersion(&driver) != cudaSuccess || driver == 0)
                return fail("no NVIDIA driver was found (libcuda could not be loaded)");
            if (driver < kNeedDriver) {
                char buf[256];
                std::snprintf(buf, sizeof buf, "the NVIDIA driver supports CUDA %d.%d; this needs CUDA 12.8, which "
                              "NVIDIA driver R570 or newer has", driver / 1000, (driver % 1000) / 10);
                return fail(buf);
            }
            int count = 0;
            const cudaError_t e = cudaGetDeviceCount(&count);
            if (e != cudaSuccess || count == 0)
                return fail(std::string("no NVIDIA GPU is available (") +
                            (e != cudaSuccess ? cudaGetErrorString(e) : "0 devices") + ")");
            if (!kt08_cuda_register_backend())
                return fail("the CUDA backend could not be registered");
        }
        if (!set_active_backend(name.c_str()))
            return fail("backend '" + name + "' is not in this library");
        return 0;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
}

KT08G_API const char* kt08g_device(void) {
    static thread_local std::string text;
    int dev = 0, driver = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return "";
    cudaDriverGetVersion(&driver);
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s (compute capability %d.%d, driver CUDA %d.%d)", p.name, p.major, p.minor,
                  driver / 1000, (driver % 1000) / 10);
    text = buf;
    return text.c_str();
}

KT08G_API void* kt08g_load(const char* arch_path, const char* weights_path) {
    try {
        auto g = std::make_unique<GraphModel>();
        g->model = std::make_unique<InferenceModel>(arch_path, weights_path);
        return g.release();
    } catch (const std::exception& ex) {
        fail(ex.what());
        return nullptr;
    }
}

KT08G_API void kt08g_free(void* model) { delete static_cast<GraphModel*>(model); }

KT08G_API int kt08g_set_speed(void* model, float speed) {
    try {
        static_cast<GraphModel*>(model)->model->set_speed(speed);
        return 0;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
}

// Shapes as python/bindings.cpp's dict forward: fixed inputs take the arch file's shape; inputs with a variable
// dimension get {H, W, C}, {H, C} or {C} from the element count.
KT08G_API int kt08g_forward(void* model, int n, const char* const* names, const float* const* data,
                            const int64_t* sizes) {
    try {
        auto* g = static_cast<GraphModel*>(model);
        InferenceModel& self = *g->model;
        std::unordered_map<std::string, const InputSpec*> spec_map;
        for (const auto& s : self.inputs()) spec_map[s.name] = &s;
        std::unordered_map<std::string, ModelIOTensor> named;
        for (int i = 0; i < n; ++i) {
            const std::string name = names[i];
            const int64_t size = sizes[i];
            std::vector<int> shape;
            auto sit = spec_map.find(name);
            if (sit != spec_map.end()) {
                const InputSpec& spec = *sit->second;
                bool fixed = true;
                for (int d : spec.shape) if (d == 0) { fixed = false; break; }
                if (fixed) {
                    shape = spec.shape;
                } else {
                    int C = spec.shape.empty() ? 1 : spec.shape.back();
                    int W = (int)spec.shape.size() >= 3 ? spec.shape[1] : 1;
                    W = std::max(W, 1);
                    int H = (int)(size / ((int64_t)C * W));
                    if (W > 1)      shape = {H, W, C};
                    else if (H > 1) shape = {H, C};
                    else            shape = {C};
                }
            } else {
                shape = {(int)size};
            }
            named[name] = ModelIOTensor(data[i], std::move(shape));
        }
        auto outputs = self.forward(named);
        g->outputs.clear();
        for (auto& [name, ot] : outputs) g->outputs.emplace_back(name, std::vector<float>(ot.begin(), ot.end()));
        return 0;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
}

KT08G_API int kt08g_output_count(void* model) { return (int)static_cast<GraphModel*>(model)->outputs.size(); }

KT08G_API const char* kt08g_output_name(void* model, int i) {
    return static_cast<GraphModel*>(model)->outputs.at(i).first.c_str();
}

KT08G_API const float* kt08g_output_data(void* model, int i, int64_t* size) {
    const auto& v = static_cast<GraphModel*>(model)->outputs.at(i).second;
    *size = (int64_t)v.size();
    return v.data();
}
