#pragma once
// Per-load overrides of the opt-in engine switches that are otherwise read from the environment
// (KITTEN_FP16_STORE, KITTEN_FP16_COMPUTE). The C API (src/capi) sets them on the loading thread for the duration of
// one InferenceModel constructor, so an app chooses them per model without touching the process environment.
// -1 = not set (the environment variable decides, as before); 0 = off; 1 = on.
struct KtLoadOverrides {
    int fp16_store = -1;
    int fp16_compute = -1;
};

inline KtLoadOverrides& kt_load_overrides() {
    static thread_local KtLoadOverrides o;
    return o;
}

// Sets the overrides for the lifetime of the scope (the constructor call), then restores the previous values.
class KtLoadOverridesScope {
public:
    explicit KtLoadOverridesScope(const KtLoadOverrides& o) : saved_(kt_load_overrides()) { kt_load_overrides() = o; }
    ~KtLoadOverridesScope() { kt_load_overrides() = saved_; }
    KtLoadOverridesScope(const KtLoadOverridesScope&) = delete;
    KtLoadOverridesScope& operator=(const KtLoadOverridesScope&) = delete;

private:
    KtLoadOverrides saved_;
};
