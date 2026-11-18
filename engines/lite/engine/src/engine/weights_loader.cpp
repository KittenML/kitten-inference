#include "engine/dev_env.hpp"
#include "weights_loader.hpp"
#include "weight_cache.hpp"
#include "utils/file_open.hpp"
#include <algorithm>
#include <chrono>
#include <typeinfo>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define KT_WEIGHTS_CAN_MMAP 1
#else
#define KT_WEIGHTS_CAN_MMAP 0
#endif

namespace {

// The weights file, read-only: memory-mapped on Linux and Android, else read with stdio (Windows, Apple platforms, a
// failed mmap, or KITTEN_WEIGHTS_MMAP=0). Apple platforms map only with KITTEN_WEIGHTS_MMAP=1: there madvise(DONTNEED)
// does not drop the mapped file pages, so the whole file counts in the resident size while loading (iPhone SE 2020:
// ru_maxrss 598 MiB mapped vs 356 MiB with stdio; phys_footprint the same), for no measurable load-time gain.
//
// Every layer copies or repacks its tensors into its own arrays at load, so nothing points into the file afterwards:
// the mapping is closed when loading ends. While loading, the pages of the tensors a layer has read are dropped again
// after that layer (madvise DONTNEED), so the mapped file never adds more than about one layer's weights to the
// process's resident memory. The values are the same bytes either way, so the output does not change.
class WeightFile {
public:
    WeightFile(const std::string& path) : path_(path) {
#if KT_WEIGHTS_CAN_MMAP
        const char* e = std::getenv("KITTEN_WEIGHTS_MMAP");
#if defined(__APPLE__)
        const bool use_mmap = e && e[0] == '1';
#else
        const bool use_mmap = !(e && e[0] == '0');
#endif
        if (use_mmap) open_mapped();
#endif
        if (!map_) open_stdio();
    }
    ~WeightFile() { close(); }
    WeightFile(const WeightFile&) = delete;
    WeightFile& operator=(const WeightFile&) = delete;

    bool mapped() const { return map_ != nullptr; }
    long long size() const { return size_; }  // -1: unknown (stdio, ftell failed)

    void read_at(uint64_t offset, void* dst, size_t n, const char* what) {
        if (map_) {
            if (offset > (uint64_t)size_ || n > (uint64_t)size_ - offset)
                throw std::runtime_error(std::string("read error at ") + what);
            memcpy(dst, map_ + offset, n);
            touched(offset, n);
            return;
        }
        if (offset != pos_) {
            if (offset > (uint64_t)LONG_MAX || fseek(f_, (long)offset, SEEK_SET) != 0)
                throw std::runtime_error("Cannot read weights: " + path_);
        }
        if (fread(dst, 1, n, f_) != n)
            throw std::runtime_error(std::string("read error at ") + what);
        pos_ = offset + n;
    }

    // The stored bytes in place (mapped files only; the range was checked against the file size by the caller).
    const uint8_t* view(uint64_t offset, size_t n) {
        touched(offset, n);
        return map_ + offset;
    }

    // Drops the pages that lie wholly inside the ranges read since the last call (mapped files only).
    void release_touched() {
#if KT_WEIGHTS_CAN_MMAP
        if (!map_) return;
        for (const auto& r : touched_) {
            const uint64_t lo = (r.first + page_ - 1) / page_ * page_;
            const uint64_t hi = (r.first + r.second) / page_ * page_;
            if (hi > lo) madvise(const_cast<uint8_t*>(map_) + lo, (size_t)(hi - lo), MADV_DONTNEED);
        }
#endif
        touched_.clear();
    }

    void close() {
#if KT_WEIGHTS_CAN_MMAP
        if (map_) { munmap(const_cast<uint8_t*>(map_), (size_t)size_); map_ = nullptr; }
#endif
        if (f_) {
            // Check the return value (fclose can fail on network filesystems).
            if (fclose(f_) != 0)
                fprintf(stderr, "Warning: fclose failed for %s: %s\n", path_.c_str(), strerror(errno));
            f_ = nullptr;
        }
    }

private:
    void touched(uint64_t offset, size_t n) {
        if (map_ && n) touched_.push_back({offset, (uint64_t)n});
    }

#if KT_WEIGHTS_CAN_MMAP
    void open_mapped() {
        const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        struct stat st;
        if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
            void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p != MAP_FAILED) {
                map_ = static_cast<const uint8_t*>(p);
                size_ = (long long)st.st_size;
                const long ps = sysconf(_SC_PAGESIZE);
                page_ = ps > 0 ? (uint64_t)ps : 4096;
                madvise(p, (size_t)st.st_size, MADV_SEQUENTIAL);  // read ahead: tensors are mostly read in file order
            }
        }
        ::close(fd);  // the mapping stays valid without the descriptor
    }
#endif

    void open_stdio() {
        f_ = fopen_utf8(path_, "rb");
        if (!f_) throw std::runtime_error("Cannot open weights: " + path_);
        // The file's size, to check every index entry against it (unknown when ftell cannot report it).
        if (fseek(f_, 0, SEEK_END) == 0) size_ = (long long)ftell(f_);
        if (fseek(f_, 0, SEEK_SET) != 0) throw std::runtime_error("Cannot read weights: " + path_);
        pos_ = 0;
    }

    std::string path_;
    const uint8_t* map_ = nullptr;
    FILE* f_ = nullptr;
    long long size_ = -1;
    uint64_t pos_ = 0;
    uint64_t page_ = 4096;
    std::vector<std::pair<uint64_t, uint64_t>> touched_;
};

}  // namespace

void load_weights(const std::string& weights_path,
                  std::vector<std::unique_ptr<ILayer>>& layers,
                  WeightLoadMemoryStats* memory_stats,
                  const std::function<void(ILayer*)>& after_layer)
{
    clear_packed_weight_cache();

    WeightFile file(weights_path);
    const long long file_size = file.size();
    if (memory_stats) memory_stats->mapped = file.mapped();

    char magic[4];
    file.read_at(0, magic, 4, "magic");
    if (memcmp(magic, "W1X1", 4) != 0)
        throw std::runtime_error("Bad magic in " + weights_path + " (expected W1X1)");

    uint32_t version, num_entries;
    file.read_at(4, &version,     4, "version");
    file.read_at(8, &num_entries, 4, "num_entries");

    struct IndexEntry {
        char     name[64];
        uint64_t offset;
        uint32_t nbytes;
        uint32_t _pad;
    };
    if (file_size >= 0 && (unsigned long long)num_entries * sizeof(IndexEntry) > (unsigned long long)file_size)
        throw std::runtime_error("Bad index in " + weights_path + " (more entries than the file can hold)");
    std::unordered_map<std::string, IndexEntry> index;
    index.reserve(num_entries);
    for (uint32_t i = 0; i < num_entries; ++i) {
        IndexEntry e;
        file.read_at(12 + (uint64_t)i * sizeof(IndexEntry), &e, sizeof(e), "index_entry");
        e.name[63] = '\0';
        if (e.nbytes < 4 || (file_size >= 0 && (e.offset > (uint64_t)file_size ||
                                                e.nbytes > (uint64_t)file_size - e.offset)))
            throw std::runtime_error("Bad index entry for tensor " + std::string(e.name) + " in " + weights_path +
                                     " (outside the file)");
        index[std::string(e.name)] = e;
    }
    file.release_touched();

    // Finds a tensor, reads and checks its header; on success gives where its data starts and how many bytes it has.
    auto locate = [&](const std::string& tensor_name, std::vector<int32_t>& shape_out,
                      uint64_t& data_offset, size_t& data_bytes) -> bool {
        auto it = index.find(tensor_name);
        if (it == index.end()) return false;
        const IndexEntry& e = it->second;
        uint8_t head[4];  // dtype, ordering, ndim (uint16)
        file.read_at(e.offset, head, 4, "tensor_header");
        uint16_t ndim;
        memcpy(&ndim, head + 2, 2);
        if ((size_t)e.nbytes < 4 + (size_t)ndim * 4)
            throw std::runtime_error("Bad tensor " + tensor_name + " in " + weights_path + " (header too long)");
        shape_out.resize(ndim);
        if (ndim > 0)
            file.read_at(e.offset + 4, shape_out.data(), (size_t)ndim * 4, "tensor_shape");
        data_bytes = (size_t)e.nbytes - 4 - (size_t)ndim * 4;
        data_offset = e.offset + 4 + (uint64_t)ndim * 4;
        // The shape and the data must agree: 1, 2, 4 or 8 bytes per value.
        size_t values = 1;
        for (int32_t d : shape_out) {
            if (d < 1 || values > ((size_t)1 << 40) / (size_t)d)
                throw std::runtime_error("Bad tensor " + tensor_name + " in " + weights_path + " (shape)");
            values *= (size_t)d;
        }
        const size_t per_value = data_bytes % values == 0 ? data_bytes / values : 0;
        if (per_value != 1 && per_value != 2 && per_value != 4 && per_value != 8)
            throw std::runtime_error("Bad tensor " + tensor_name + " in " + weights_path + " (" +
                                     std::to_string(data_bytes) + " bytes for " + std::to_string(values) +
                                     " values)");
        return true;
    };

    // read_tensor captures the file by reference — must be called synchronously within
    // this scope; layers must not store the lambda for deferred use.
    ReadTensorFn read_tensor = [&](const std::string& tensor_name,
                                    std::vector<int32_t>& shape_out,
                                    std::vector<uint8_t>& data_out) -> bool {
        uint64_t offset;
        size_t data_bytes;
        if (!locate(tensor_name, shape_out, offset, data_bytes)) return false;
        data_out.resize(data_bytes);
        if (memory_stats && data_bytes > memory_stats->peak_tensor_bytes)
            memory_stats->peak_tensor_bytes = data_bytes;
        file.read_at(offset, data_out.data(), data_bytes, "tensor_data");
        return true;
    };

    // Mapped file: read_f32_tensor (layer.hpp) copies straight from the pages, without the staging vector above.
    TensorViewSource view_source;
    view_source.read = &read_tensor;
    view_source.view = [&](const std::string& tensor_name, std::vector<int32_t>& shape_out,
                           const uint8_t*& bytes, size_t& n) -> bool {
        uint64_t offset;
        if (!locate(tensor_name, shape_out, offset, n)) return false;
        bytes = file.view(offset, n);
        return true;
    };
    struct ViewSourceScope {
        const TensorViewSource* saved;
        explicit ViewSourceScope(const TensorViewSource* s) : saved(kt_tensor_view_source()) { kt_tensor_view_source() = s; }
        ~ViewSourceScope() { kt_tensor_view_source() = saved; }
    } view_scope(file.mapped() ? &view_source : nullptr);

    // KT08_LOAD_PROFILE=1: seconds spent in each layer class's load_weights (read + repack) and in releasing the pages, to stderr
    static const bool load_prof = [] { const char* e = kt_dev_getenv("KT08_LOAD_PROFILE"); return e && e[0] == '1'; }();
    std::unordered_map<std::string, double> lp_ms;
    double lp_release_ms = 0;
    auto lp_now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    for (auto& layer : layers) {
        const double t0 = load_prof ? lp_now() : 0.0;
        layer->load_weights(read_tensor);
        const double t1 = load_prof ? lp_now() : 0.0;
        file.release_touched();
        if (load_prof) { lp_ms[typeid(*layer).name()] += t1 - t0; lp_release_ms += lp_now() - t1; }
        if (after_layer) after_layer(layer.get());
    }
    if (load_prof) {
        for (auto& kv : lp_ms) std::fprintf(stderr, "LOADPROF %s %.1f ms\n", kv.first.c_str(), kv.second);
        std::fprintf(stderr, "LOADPROF release_touched %.1f ms\n", lp_release_ms);
    }
    clear_packed_weight_cache();
    file.close();

    // printf("Loaded W1X1 weights from %s  (%u tensors)\n",
    //        weights_path.c_str(), num_entries);
}
