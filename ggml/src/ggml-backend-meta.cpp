#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_GATHER1:
            return "GATHER1";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        // note: use the member, not the moved-from constructor parameter of the same name
        for (size_t i = 0; i < this->simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (this->simple_devs[i]);
            description += ggml_backend_dev_description(this->simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        // Host (pinned) buffers are forwarded from the member devices when they
        // all share one host buffer type (see get_host_buffer_type); required
        // for the model loader's chunked async-upload fast path.
        /* .host_buffer           = */ ggml_backend_meta_device_get_host_buffer_type(dev) != nullptr,
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ true, // Meta events fan out to member-device events.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}


static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    auto * evs = new std::vector<ggml_backend_event_t>();
    evs->reserve(meta_dev_ctx->simple_devs.size());
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_event_t ev = ggml_backend_event_new(simple_dev);
        if (ev == nullptr) {
            for (ggml_backend_event_t e : *evs) {
                ggml_backend_event_free(e);
            }
            delete evs;
            return nullptr;
        }
        evs->push_back(ev);
    }
    return new ggml_backend_event {dev, evs};
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * evs = (std::vector<ggml_backend_event_t> *) event->context;
    for (ggml_backend_event_t e : *evs) {
        ggml_backend_event_free(e);
    }
    delete evs;
    delete event;
    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * evs = (std::vector<ggml_backend_event_t> *) event->context;
    for (ggml_backend_event_t e : *evs) {
        ggml_backend_event_synchronize(e);
    }
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        // note: use the member, not the moved-from constructor parameter of the same name
        for (size_t i = 0; i < this->simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(this->simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

// Memoize split-state computation, which recurses through each tensor's source
// subtree and re-derives shared subtrees O(consumers) times. The cache is active
// for the duration of one immutable graph compute, activated via
// ggml_backend_meta_split_cache_begin/end() around the whole sched split-walk
// (see ggml_backend_sched_compute_splits) so it also covers the hot inter-split
// copy path (get_tensor_backend / cpy_tensor), not just per-split graph_compute.
// Results are always correct because the graph is immutable within the scope;
// outside a scope the pointer is null and we fall through to the uncached impl.
struct ggml_backend_meta_split_cache_entry { ggml_backend_meta_split_state ss[2]; bool valid[2] = {false, false}; };
static thread_local std::unordered_map<const ggml_tensor *, ggml_backend_meta_split_cache_entry> * g_meta_split_cache = nullptr;

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state_impl(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    if (g_meta_split_cache == nullptr) {
        return ggml_backend_meta_get_split_state_impl(stc, tensor, assume_sync);
    }
    const int idx = assume_sync ? 1 : 0;
    auto it = g_meta_split_cache->find(tensor);
    if (it != g_meta_split_cache->end() && it->second.valid[idx]) {
        return it->second.ss[idx];
    }
    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state_impl(stc, tensor, assume_sync);
    auto & e = (*g_meta_split_cache)[tensor];
    e.ss[idx] = ss;
    e.valid[idx] = true;
    return ss;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state_impl(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.

    // hybrid TP x PP: a tensor living outside this meta device (another pipeline stage's
    // buffer, or a host/plain-GPU buffer) is a complete replica from this device's
    // perspective; boundary copies broadcast it to all simple backends.
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }

    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        // batched mat-mul with both operands split identically along a batch axis
        // (e.g. MLA k_b/v_b absorption: per-head matrices x per-head activations).
        // Each device holds complete matrices for its subset of batch entries, so
        // the result inherits the same batch-axis split with no communication.
        if ((src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_3) &&
                src_ss[1].axis == src_ss[0].axis) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        fprintf(stderr, "TPHYBRID mul_mat unhandled: node=%s src0=%s(axis=%d) src1=%s(axis=%d)\n",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        GGML_ABORT("fatal error");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        // Position-sharded sparse gather (LLAMA_KV_SHARD + DSA decode): rows
        // distributed on axis 1, indices are mirrored GLOBAL positions. Each
        // member gathers only the rows it owns (the CUDA kernel rebases via
        // op_params and ZERO-FILLS foreign rows), so the member-sum is the
        // complete gather -> PARTIAL; the standard allreduce completes it.
        // ~n_sel*row_bytes moved per boundary instead of the whole cache.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // Position-sharded KV cache (LLAMA_KV_SHARD): src[0]=values (mirrored),
        // src[1]=indices (mirrored), src[2]=destination cache split on the row
        // axis. Each member executes the same set_rows with per-member-rebased
        // indices; the CUDA kernel skips rows outside [0, n_rows_dst) so
        // out-of-shard writes are no-ops. Result inherits the cache's state.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            return src_ss[2];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                src_ss[1].axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                !split_states_equal(src_ss[0], src_ss[2])) {
            fprintf(stderr, "META set_rows mismatch: node=%s dst=%s(axis %d) idx=%s(axis %d) val=%s(axis %d)\n",
                tensor->name, tensor->src[0]->name, (int) src_ss[0].axis,
                tensor->src[1]->name, (int) src_ss[1].axis,
                tensor->src[2]->name, (int) src_ss[2].axis);
        }
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // fully mirrored attention (e.g. bisection/debug): output mirrored
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        // MLA/MQA: a single shared KV head may be mirrored on all devices while the
        // query heads are split; each device then runs a complete local MQA attention.
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                                 src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_mirrored || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(kv_mirrored || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        // hybrid TP x PP: a tensor living outside this meta device (another pipeline
        // stage's buffer, or a host/plain-GPU buffer) is a complete replica from this
        // device's perspective; boundary copies broadcast it to all simple backends.
        if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                // Position-sharded KV (LLAMA_KV_SHARD): a cont named
                // "kvgather*" over an axis-1-split tensor marks a gather
                // boundary - the subgraph splitter breaks here and the
                // boundary code exchanges the halves into the full
                // (mirrored) per-member buffers. Consumers (assume_sync)
                // see the result as mirrored; allocation is full-size.
                if (tensor->op == GGML_OP_CONT &&
                        (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                         src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2) &&
                        strncmp(tensor->name, "kvgather", 8) == 0) {
                    split_state = {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_GATHER1, {0}, {1}, 1};
                    break;
                }
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARANGE: {
                // source-less generator: every member computes the identical
                // full tensor -> mirrored
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                // DSA fused indexer: with the lid cache mirrored (kv_shard) and
                // the indexer weights mirrored, every input is member-identical
                // -> both members compute the identical full score tensor
                for (size_t si = 0; si < GGML_MAX_SRC; si++) {
                    if (tensor->src[si] != nullptr) {
                        GGML_ASSERT(src_ss[si].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_PARTIAL: {
                // each member fills its own slot of the doubled dst (peer slot
                // zeroed); the standard allreduce completes both slots
                split_state = {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
            } break;
            case GGML_OP_FLASH_ATTN_COMBINE: {
                // consumes the completed (mirrored) partial pair; result is
                // computed identically on every member
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                // An offset smaller than the stride of one split unit stays within each unit and must not be
                // scaled (e.g. the MLA q_pe view: a within-head offset into rows split by head).
                if (t_ij->view_offs < tensor->nb[split_dim]) {
                    split_internal_offset = true;
                }
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    // Gather-boundary nodes (position-sharded KV) are filled by the boundary
    // exchange code, never computed per member (their per-member src is a
    // half while the dst is full-size).
    if (tensor->op == GGML_OP_CONT && strncmp(tensor->name, "kvgather", 8) == 0) {
        for (size_t j = 0; j < n_simple_bufs; j++) {
            simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    // Row-ranged access to a single-segment axis-1 split of an effectively-2D
    // tensor: the position-sharded KV cache, where member j owns the global
    // row range [base_j, base_j + ne_j) in member order (rotation pinned 0).
    // llama state save/load (--cache-ram / --slot-save-path) writes
    // per-sequence CELL RANGES, which the whole-tensor chunk splice below
    // asserts against. Ownership comes from the PARENT cache's split state:
    // a view's own split state distributes rows proportionally (S1 lesson).
    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_1 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            offset % tensor->nb[1] == 0 && size % tensor->nb[1] == 0 &&
            !(offset == 0 && size == ggml_nbytes(tensor))) {
        const ggml_tensor * root = tensor->view_src != nullptr ? tensor->view_src : tensor;
        const ggml_backend_meta_split_state rss = ggml_backend_meta_get_split_state(root, /*assume_sync =*/ false);
        if (rss.axis == GGML_BACKEND_SPLIT_AXIS_1 && rss.n_segments == 1 && root->ne[1] == tensor->ne[1]) {
            const size_t  row_size  = tensor->nb[1];
            const int64_t row_start = (int64_t) (offset / row_size);
            const int64_t row_count = (int64_t) (size / row_size);
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);
            size_t  data_off = 0;
            int64_t base_j   = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const int64_t ne_j  = rss.ne[j];
                const int64_t r_beg = std::max<int64_t>(row_start, base_j);
                const int64_t r_end = std::min<int64_t>(row_start + row_count, base_j + ne_j);
                if (r_beg < r_end && simple_tensor != nullptr) {
                    ggml_backend_tensor_set(simple_tensor, (const char *) data + data_off,
                        (size_t) (r_beg - base_j) * row_size, (size_t) (r_end - r_beg) * row_size);
                    data_off += (size_t) (r_end - r_beg) * row_size;
                }
                base_j += ne_j;
            }
            GGML_ASSERT(data_off == size);
            return;
        }
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    // Row-ranged access to a single-segment axis-1 split (position-sharded KV
    // cache) — mirror of the set_tensor fast path; see comment there.
    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_1 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            offset % tensor->nb[1] == 0 && size % tensor->nb[1] == 0 &&
            !(offset == 0 && size == ggml_nbytes(tensor))) {
        const ggml_tensor * root = tensor->view_src != nullptr ? tensor->view_src : tensor;
        const ggml_backend_meta_split_state rss = ggml_backend_meta_get_split_state(root, /*assume_sync =*/ false);
        if (rss.axis == GGML_BACKEND_SPLIT_AXIS_1 && rss.n_segments == 1 && root->ne[1] == tensor->ne[1]) {
            const size_t  row_size  = tensor->nb[1];
            const int64_t row_start = (int64_t) (offset / row_size);
            const int64_t row_count = (int64_t) (size / row_size);
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);
            size_t  data_off = 0;
            int64_t base_j   = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const int64_t ne_j  = rss.ne[j];
                const int64_t r_beg = std::max<int64_t>(row_start, base_j);
                const int64_t r_end = std::min<int64_t>(row_start + row_count, base_j + ne_j);
                if (r_beg < r_end && simple_tensor != nullptr) {
                    ggml_backend_tensor_get(simple_tensor, (char *) data + data_off,
                        (size_t) (r_beg - base_j) * row_size, (size_t) (r_end - r_beg) * row_size);
                    data_off += (size_t) (r_end - r_beg) * row_size;
                }
                base_j += ne_j;
            }
            GGML_ASSERT(data_off == size);
            return;
        }
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        if (bufs.back() == nullptr) {
            // propagate allocation failure so callers (e.g. the scheduler's
            // pipeline-parallel reserve) can retry with a smaller footprint
            // instead of aborting the process
            for (ggml_backend_buffer_t b : bufs) {
                if (b != nullptr) {
                    ggml_backend_buffer_free(b);
                }
            }
            return nullptr;
        }
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    // Launch pool: host-side kernel enqueueing is serial over pair members and
    // shows up as cuLaunchKernel throttling during prefill. Member launches
    // within a subgraph are independent (CUDA is thread-safe; the CUDA backend
    // sets its device per call), so members 1..n-1 launch on persistent
    // workers while member 0 launches on the caller. Handoff is a short spin
    // (arm->run gap is ~us) with a condition-variable fallback so idle workers
    // don't burn CPU between tokens. GGML_META_PARALLEL_LAUNCH=0 disables.
    struct launch_worker {
        std::thread             thread;
        std::mutex              mtx;
        std::condition_variable cv;
        std::atomic<int>        state{0};        // 0 idle, 1 armed, 2 done, -1 shutdown
        ggml_backend_t          job_backend = nullptr;
        ggml_cgraph *           job_cgraph  = nullptr;
        ggml_status             job_status  = GGML_STATUS_SUCCESS;

        void run() {
            for (;;) {
                int spins = 0;
                while (state.load(std::memory_order_acquire) == 0 && spins < 2000) {
                    spins++;
                    std::this_thread::yield();
                }
                if (state.load(std::memory_order_acquire) == 0) {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return state.load(std::memory_order_acquire) != 0; });
                }
                const int s = state.load(std::memory_order_acquire);
                if (s == -1) {
                    return;
                }
                if (s == 1) {
                    job_status = ggml_backend_graph_compute_async(job_backend, job_cgraph);
                    state.store(2, std::memory_order_release);
                }
            }
        }

        void arm(ggml_backend_t backend, ggml_cgraph * cgraph) {
            job_backend = backend;
            job_cgraph  = cgraph;
            {
                std::lock_guard<std::mutex> lock(mtx);
                state.store(1, std::memory_order_release);
            }
            cv.notify_one();
        }

        ggml_status join() {
            while (state.load(std::memory_order_acquire) != 2) {
                std::this_thread::yield();
            }
            state.store(0, std::memory_order_release);
            return job_status;
        }

        void shutdown() {
            if (!thread.joinable()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                state.store(-1, std::memory_order_release);
            }
            cv.notify_one();
            thread.join();
        }
    };
    std::vector<std::unique_ptr<launch_worker>> launch_workers; // size n_backends-1 once started
    bool launch_pool_started = false;

    // per-subgraph boundary kind: 1 = gather (position-sharded KV halves
    // exchange), 0 = normal (allreduce / none); _node holds the meta-graph
    // node index of the gather node for that subgraph (-1 otherwise)
    std::vector<uint8_t> subgraph_gather;
    std::vector<int>     subgraph_gather_node;
    // per-member events used to order gather-boundary P2P copies against
    // both members' in-flight kernels (lazily created)
    std::vector<ggml_backend_event_t> gather_events;
    // GGML_META_STAGE_KEEP_EXECS=1 bisect: captured graph execs whose
    // destruction is deferred to the next graph_compute call (tests the
    // destroy-while-launch-in-flight driver theory without adding syncs)
    std::vector<std::pair<ggml_backend_t, void *>> kept_execs;

    // Pinned staging ring for host->meta COMPUTE-buffer uploads (graph inputs).
    // cudaMemcpyAsync from pinned host memory reads the source at EXECUTION
    // time; when the host runs far ahead of the GPU (per-stage CUDA graphs
    // remove the launch-queue throttle), the caller may overwrite the source
    // (set_inputs input leafs) while uploads from a previous ubatch are still
    // pending -> long-context corruption. Bouncing through this ring makes the
    // upload consume the source at ENQUEUE time; slot reuse is guarded by
    // per-member events (ring depth 4 >> pipeline depth 2, waits ~never fire).
    // Weights loads (USAGE_WEIGHTS buffers) bypass the ring.
    struct staging_slot {
        ggml_backend_buffer_ptr           buf;
        size_t                            size = 0;
        std::vector<ggml_backend_event_t> evs;      // per member, lazily created
        bool                              pending = false;
    };
    // Size-tiered rings: small uploads (pos/embd/out_ids) cycle several times
    // per ubatch — a deep ring keeps the host multiple ubatches ahead (the
    // 7-pair pipeline needs ~7 chunks of host lead at the front). Large
    // uploads (the KQ mask, ~n_kv*n_ub*2 bytes) are one per pair per ubatch —
    // a shallow dedicated ring bounds pinned memory while still giving
    // 4 ubatches of lead.
    static constexpr size_t staging_small_max = 32u << 20;
    std::array<staging_slot, 12> staging_small;
    std::array<staging_slot, 4>  staging_large;
    size_t staging_small_next = 0;
    size_t staging_large_next = 0;

    void * staging_acquire(size_t size) {
        const bool large = size > staging_small_max;
        staging_slot & s = large ? staging_large[staging_large_next] : staging_small[staging_small_next];
        if (large) {
            staging_large_next = (staging_large_next + 1) % staging_large.size();
        } else {
            staging_small_next = (staging_small_next + 1) % staging_small.size();
        }
        if (s.pending) {
            for (ggml_backend_event_t ev : s.evs) {
                if (ev != nullptr) {
                    ggml_backend_event_synchronize(ev);
                }
            }
            s.pending = false;
        }
        if (s.size < size) {
            ggml_backend_dev_t dev0 = ggml_backend_get_device(backend_configs[0].backend);
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev0);
            if (host_buft == nullptr) {
                return nullptr; // no pinned host buffers on this backend
            }
            // small slots get 2x slack to avoid regrow churn; large slots
            // (masks grow monotonically with n_kv) get 1.25x
            const size_t alloc_size = large ? size + size/4 : std::max<size_t>(size, 4u << 20) * 2;
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(host_buft, alloc_size);
            if (b == nullptr) {
                return nullptr;
            }
            s.buf.reset(b);
            s.size = alloc_size;
        }
        return ggml_backend_buffer_get_base(s.buf.get());
    }

    bool staging_commit_ring(void * base, staging_slot * ring, size_t n) {
        for (size_t i = 0; i < n; i++) {
            staging_slot & s = ring[i];
            if (s.buf && ggml_backend_buffer_get_base(s.buf.get()) == base) {
                if (s.evs.size() < backend_configs.size()) {
                    s.evs.resize(backend_configs.size(), nullptr);
                }
                for (size_t j = 0; j < backend_configs.size(); j++) {
                    if (s.evs[j] == nullptr) {
                        s.evs[j] = ggml_backend_event_new(ggml_backend_get_device(backend_configs[j].backend));
                    }
                    if (s.evs[j] != nullptr) {
                        ggml_backend_event_record(s.evs[j], backend_configs[j].backend);
                    }
                }
                s.pending = true;
                return true;
            }
        }
        return false;
    }

    void staging_commit(void * base) {
        if (staging_commit_ring(base, staging_small.data(), staging_small.size()) ||
            staging_commit_ring(base, staging_large.data(), staging_large.size())) {
            return;
        }
        GGML_ABORT("staging_commit: unknown staging base");
    }

    void launch_pool_start() {
        // Opt-in (GGML_META_PARALLEL_LAUNCH=1): measured neutral for prefill on
        // the clean rig (815 t/s with and without) — the launch bottleneck is
        // the per-device pending-launch queue, not the serial member loop.
        static const bool enabled = [] {
            const char * env = getenv("GGML_META_PARALLEL_LAUNCH");
            return env != nullptr && atoi(env) != 0;
        }();
        launch_pool_started = true;
        if (!enabled || backend_configs.size() < 2) {
            return;
        }
        launch_workers.resize(backend_configs.size() - 1);
        for (auto & w : launch_workers) {
            w = std::make_unique<launch_worker>();
            w->thread = std::thread([worker = w.get()] { worker->run(); });
        }
    }

    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    // Outer-capture (whole-subgraph-sequence CUDA graph) hooks, fetched from
    // the simple backend's registry.  All-or-nothing: capture_* stay null if
    // the backend doesn't provide the full set.
    bool   (*capture_begin)(ggml_backend_t)          = nullptr;
    void * (*capture_end)(ggml_backend_t)            = nullptr;
    bool   (*capture_launch)(ggml_backend_t, void *) = nullptr;
    void   (*capture_free)(ggml_backend_t, void *)   = nullptr;
    void   (*capture_abort)(ggml_backend_t)          = nullptr;

    // Cached captured graphs, keyed by cgraph uid.  uids are monotonic and
    // never reused, so a stale entry can never be matched again; evict oldest
    // beyond a small cap (spec-decode verify passes cycle a few shapes).
    struct outer_graph_entry {
        uint64_t            uid  = 0;
        int                 warm = 0;
        bool                capturable = true;
        std::vector<void *> execs; // one per simple backend; empty until captured
        uint64_t            last_used = 0;
    };
    std::vector<outer_graph_entry> outer_graphs;
    uint64_t                       outer_graph_clock   = 0;
    bool                           outer_graph_enabled = false;

    // creation-order ordinal (pair index for the TPxPP hybrid); bisect aid
    int ctx_ord = -1;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        static std::atomic<int> next_ord{0};
        ctx_ord = next_ord++;
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }

        if (comm_ctx != nullptr && getenv("GGML_META_GRAPH_OFF") == nullptr) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            capture_begin  = (bool   (*)(ggml_backend_t))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_begin");
            capture_end    = (void * (*)(ggml_backend_t))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_end");
            capture_launch = (bool   (*)(ggml_backend_t, void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_launch");
            capture_free   = (void   (*)(ggml_backend_t, void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_free");
            capture_abort  = (void   (*)(ggml_backend_t))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_capture_abort");
            outer_graph_enabled = capture_begin && capture_end && capture_launch && capture_free && capture_abort;
        }
    }

    ~ggml_backend_meta_context() {
        for (auto & e : gather_events) {
            if (e != nullptr) {
                ggml_backend_event_free(e);
            }
        }
        for (auto & ke : kept_execs) {
            if (capture_free != nullptr) {
                capture_free(ke.first, ke.second);
            }
        }
        for (auto & s : staging_small) {
            for (auto & ev : s.evs) {
                if (ev != nullptr) {
                    ggml_backend_event_free(ev);
                }
            }
        }
        for (auto & s : staging_large) {
            for (auto & ev : s.evs) {
                if (ev != nullptr) {
                    ggml_backend_event_free(ev);
                }
            }
        }
        for (auto & w : launch_workers) {
            if (w) {
                w->shutdown();
            }
        }
        for (auto & e : outer_graphs) {
            for (size_t j = 0; j < e.execs.size(); j++) {
                if (e.execs[j] != nullptr && capture_free != nullptr) {
                    capture_free(backend_configs[j].backend, e.execs[j]);
                }
            }
        }
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    // Graph-input uploads (COMPUTE buffers) bounce through the pinned staging
    // ring so the source is consumed at enqueue time — the caller may reuse
    // the source buffer while the async H2D of a prior ubatch is still
    // pending (see staging ring comment). Weights loads bypass.
    static const bool no_staging = [](){
        const char * e = getenv("GGML_META_NO_INPUT_STAGING");
        return e != nullptr && atoi(e) != 0;
    }();
    void * staged = nullptr;
    if (!no_staging && tensor->buffer != nullptr &&
            ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        ggml_backend_meta_context * meta_ctx = (ggml_backend_meta_context *) backend->context;
        staged = meta_ctx->staging_acquire(size);
        if (staged != nullptr) {
            memcpy(staged, data, size);
            data = staged;
        }
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Contiguous layout above the split axis: the tensor is a sequence
            // of "planes" of chunk_size_full bytes; within each plane, member j
            // owns the contiguous byte range [member_off, member_off + chunk_size_j).
            // Supports arbitrary [offset, offset + size) ranges (needed by the
            // model loader's chunked async uploads) by intersecting the range
            // with each member's slice in every plane it touches.
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            const int64_t p_first =  offset            / chunk_size_full;
            const int64_t p_last  = (offset + size - 1) / chunk_size_full;
            size_t member_off = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                // Per member: partial head/tail planes copied individually, the
                // run of fully-covered planes in one strided 2D copy (a 64MB
                // chunk can span tens of thousands of planes for axis-0 splits).
                auto copy_partial = [&](int64_t p) {
                    const size_t slice_beg = p*chunk_size_full + member_off;
                    const size_t r_beg = std::max<size_t>(offset,        slice_beg);
                    const size_t r_end = std::min<size_t>(offset + size, slice_beg + chunk_size_j);
                    if (r_beg < r_end) {
                        ggml_backend_tensor_set_async(simple_backend, simple_tensor,
                            (const char *) data + (r_beg - offset),
                            (size_t) p*chunk_size_j + (r_beg - slice_beg), r_end - r_beg);
                    }
                };
                if (p_first == p_last) {
                    // single plane: copy_partial clips to both range ends
                    copy_partial(p_first);
                } else {
                    // full planes: member slice entirely inside [offset, offset+size)
                    int64_t pf_first = p_first;
                    int64_t pf_last  = p_last;
                    if ((size_t) p_first*chunk_size_full + member_off < offset) {
                        copy_partial(p_first);
                        pf_first = p_first + 1;
                    }
                    if ((size_t) p_last*chunk_size_full + member_off + chunk_size_j > offset + size) {
                        copy_partial(p_last);
                        pf_last = p_last - 1;
                    }
                    if (pf_last >= pf_first) {
                        ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor,
                            (const char *) data + (pf_first*chunk_size_full + member_off - offset),
                            (size_t) pf_first*chunk_size_j, chunk_size_j,
                            pf_last - pf_first + 1, chunk_size_j, chunk_size_full);
                    }
                }
                member_off += chunk_size_j;
            }
            GGML_ASSERT(member_off == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }

    if (staged != nullptr) {
        ((ggml_backend_meta_context *) backend->context)->staging_commit(staged);
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    // Split-state memo cache is scoped around the whole sched split-walk by
    // ggml_backend_sched_compute_splits (begin/end), covering both this per-split
    // compute and the inter-split copy path. No per-split scope here.
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);

                // KV-shard index rebase: a SET_ROWS whose destination is an
                // axis1-sharded cache receives GLOBAL row ids (mirrored idx
                // input). Member j owns rows [base_j, base_j + ne[j]); encode
                // base_j in op_params[0] so the CUDA kernel can rebase and
                // its bounds guard drops out-of-shard rows. CUDA-only: the
                // CPU set_rows path does not read this parameter.
                if (node->op == GGML_OP_SET_ROWS && node->src[2] != nullptr) {
                    ggml_tensor * dst_root = node->src[2]->view_src != nullptr ? node->src[2]->view_src : node->src[2];
                    const ggml_backend_meta_split_state dss = ggml_backend_meta_get_split_state(dst_root, /*assume_sync =*/ false);
                    if (dss.axis == GGML_BACKEND_SPLIT_AXIS_1 && dss.n_segments == 1) {
                        int64_t base_j = 0;
                        for (size_t jj = 0; jj < j; jj++) {
                            base_j += dss.ne[jj];
                        }
                        GGML_ASSERT(base_j <= INT32_MAX);
                        ggml_set_op_params_i32(bcj.nodes[i], 0, (int32_t) base_j);
                        static int inject_logged = 0;
                        if (inject_logged < 4) {
                            inject_logged++;
                            fprintf(stderr, "[KVSBASE] inject member=%zu node=%s base=%lld ptr=%p\n",
                                    j, node->name, (long long) base_j, (void *) bcj.nodes[i]);
                        }
                    }
                }

                // KV-shard sparse gather rebase: a GET_ROWS whose source is an
                // axis1-sharded cache receives GLOBAL row ids. Encode member
                // base in op_params[0] and set op_params[1]=1 ("shard mode"):
                // the CUDA kernel rebases and ZERO-FILLS out-of-shard rows, so
                // the PARTIAL member results sum to the complete gather.
                if (node->op == GGML_OP_GET_ROWS && node->src[0] != nullptr) {
                    ggml_tensor * src_root = node->src[0]->view_src != nullptr ? node->src[0]->view_src : node->src[0];
                    const ggml_backend_meta_split_state gss = ggml_backend_meta_get_split_state(src_root, /*assume_sync =*/ false);
                    if (gss.axis == GGML_BACKEND_SPLIT_AXIS_1 && gss.n_segments == 1) {
                        int64_t base_j = 0;
                        for (size_t jj = 0; jj < j; jj++) {
                            base_j += gss.ne[jj];
                        }
                        GGML_ASSERT(base_j <= INT32_MAX);
                        ggml_set_op_params_i32(bcj.nodes[i], 0, (int32_t) base_j);
                        ggml_set_op_params_i32(bcj.nodes[i], 1, 1);
                        // member OWNED extent from the PARENT cache split (a
                        // view's own ne distributes rows proportionally and is
                        // WRONG for ownership -- the S1 lesson)
                        ggml_set_op_params_i32(bcj.nodes[i], 2, (int32_t) gss.ne[j]);
                        static int gr_logged = 0;
                        if (gr_logged < 4) {
                            gr_logged++;
                            fprintf(stderr, "[KVSGROWS] inject member=%zu node=%s base=%lld\n",
                                    j, node->name, (long long) base_j);
                        }
                    }
                }

                // S2b partial attention: tell the node which member slot of
                // the [DV+2, H, T, 2] dst it produces (op_params[8]); the
                // epilogue zeroes the peer slot and the standard allreduce
                // completes both.
                if (node->op == GGML_OP_FLASH_ATTN_PARTIAL) {
                    ggml_set_op_params_i32(bcj.nodes[i], 8, (int32_t) j);
                    static int fp_logged = 0;
                    if (fp_logged < 4) {
                        fp_logged++;
                        fprintf(stderr, "[S2BFA] inject member=%zu node=%s\n", j, node->name);
                    }
                }
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // GGML_META_LOG_SUBGRAPHS=1: one line per (re)split naming every
            // subgraph boundary node — fragmentation diagnosis for prefill.
            static const bool log_subgraphs = getenv("GGML_META_LOG_SUBGRAPHS") != nullptr;
            std::string sg_bounds;

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool is_gather = split_state.axis == GGML_BACKEND_SPLIT_AXIS_GATHER1;
                const bool new_subgraph = i + 1 == cgraph->n_nodes ||
                    split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL || is_gather;
                if (!new_subgraph) {
                    continue;
                }
                if (backend_ctx->subgraph_gather.size() <= n_subgraphs) {
                    backend_ctx->subgraph_gather.resize(n_subgraphs + 1, 0);
                    backend_ctx->subgraph_gather_node.resize(n_subgraphs + 1, -1);
                }
                backend_ctx->subgraph_gather[n_subgraphs] = is_gather ? 1 : 0;
                backend_ctx->subgraph_gather_node[n_subgraphs] = is_gather ? i : -1;

                if (log_subgraphs) {
                    sg_bounds += " ";
                    sg_bounds += is_gather ? "G:" : "P:";
                    sg_bounds += ggml_op_name(node->op);
                    sg_bounds += "(";
                    sg_bounds += node->name;
                    sg_bounds += ")";
                }

                const int i_delayed = is_gather ? (int) i : get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);

            if (log_subgraphs) {
                static int sg_logged = 0;
                if (sg_logged < 60) {
                    sg_logged++;
                    fprintf(stderr, "[SGCOUNT] ord=%d nodes=%d subgraphs=%zu bounds:%s\n",
                            backend_ctx->ctx_ord, cgraph->n_nodes, n_subgraphs, sg_bounds.c_str());
                    fflush(stderr);
                }
            }
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    static const bool tp_spike_dump   = getenv("TP_SPIKE_DUMP") != nullptr;
    static const bool tp_spike_timing = getenv("TP_SPIKE_TIMING") != nullptr;
    // TP_SUBMIT_TIMING: like TP_SPIKE_TIMING but WITHOUT the end sync - shows
    // host-side launch/allreduce blocking per compute call at normal speed
    static const bool tp_submit_timing = getenv("TP_SUBMIT_TIMING") != nullptr;
    const bool tp_time = tp_spike_timing || tp_submit_timing;
    int64_t t_launch_us = 0, t_reduce_us = 0, t_start_us = 0;
    if (tp_time) {
        t_start_us = ggml_time_us();
    }

    // -----------------------------------------------------------------------
    // Outer capture: once a uid has run enough passes for every inner CUDA
    // graph to be warmed, capture the whole per-device enqueue sequence
    // (subgraph computes + graph-safe allreduces) into one executable graph
    // per device and replay it with a single launch per pass.  Capturable
    // only when every AR boundary tensor fits the graph-mode slot ring --
    // prefill passes never qualify, decode/verify passes do.
    // -----------------------------------------------------------------------
    ggml_backend_meta_context::outer_graph_entry * og = nullptr;
    bool outer_capture_active = false;

    if (backend_ctx->outer_graph_enabled && cgraph->uid != 0 && backend_ctx->comm_ctx != nullptr &&
            n_backends > 1 && n_backends <= 4 && backend_ctx->n_subgraphs > 1 &&
            !tp_spike_dump && !tp_spike_timing) {
        for (auto & e : backend_ctx->outer_graphs) {
            if (e.uid == cgraph->uid) {
                og = &e;
                break;
            }
        }
        if (og == nullptr) {
            if (backend_ctx->outer_graphs.size() >= 16) {
                size_t lru = 0;
                for (size_t k = 1; k < backend_ctx->outer_graphs.size(); k++) {
                    if (backend_ctx->outer_graphs[k].last_used < backend_ctx->outer_graphs[lru].last_used) {
                        lru = k;
                    }
                }
                auto & ev = backend_ctx->outer_graphs[lru];
                for (size_t j = 0; j < ev.execs.size(); j++) {
                    if (ev.execs[j] != nullptr) {
                        backend_ctx->capture_free(backend_ctx->backend_configs[j].backend, ev.execs[j]);
                    }
                }
                backend_ctx->outer_graphs.erase(backend_ctx->outer_graphs.begin() + lru);
            }
            backend_ctx->outer_graphs.emplace_back();
            og = &backend_ctx->outer_graphs.back();
            og->uid = cgraph->uid;
            // Capturability: every boundary (PARTIAL) node must fit the
            // graph-mode AR staging slot (256 KB), sized for decode/verify.
            auto & bc0 = backend_ctx->backend_configs[0];
            for (size_t i = 0; i + 1 < backend_ctx->n_subgraphs && og->capturable; i++) {
                ggml_cgraph * cg = bc0.cgraphs[i].cgraph_main;
                if (cg->n_nodes < 1 || ggml_nbytes(cg->nodes[cg->n_nodes - 1]) > 256 * 1024) {
                    og->capturable = false;
                }
                // gather boundaries (position-sharded KV) host-synchronize
                // between subgraphs - never capturable
                if (i < backend_ctx->subgraph_gather.size() && backend_ctx->subgraph_gather[i] != 0) {
                    og->capturable = false;
                }
            }
        }
        og->last_used = ++backend_ctx->outer_graph_clock;

        static const bool mg_debug = getenv("GGML_META_GRAPH_DEBUG") != nullptr;
        if (mg_debug) {
            static int dbg_n = 0;
            if (dbg_n < 60 || dbg_n % 500 == 0) {
                fprintf(stderr, "MG: uid=%llu warm=%d execs=%zu cap=%d nsub=%zu entries=%zu\n",
                        (unsigned long long) cgraph->uid, og->warm, og->execs.size(),
                        (int) og->capturable, backend_ctx->n_subgraphs, backend_ctx->outer_graphs.size());
            }
            dbg_n++;
        }

        if (!og->execs.empty()) {
            bool ok = true;
            for (size_t j = 0; j < n_backends && ok; j++) {
                ok = backend_ctx->capture_launch(backend_ctx->backend_configs[j].backend, og->execs[j]);
            }
            if (ok) {
                return GGML_STATUS_SUCCESS;
            }
            for (size_t j = 0; j < og->execs.size(); j++) {
                if (og->execs[j] != nullptr) {
                    backend_ctx->capture_free(backend_ctx->backend_configs[j].backend, og->execs[j]);
                }
            }
            og->execs.clear();
            og->capturable = false;
        } else if (og->capturable && og->warm >= 3) {
            bool ok = true;
            size_t begun = 0;
            for (size_t j = 0; j < n_backends && ok; j++) {
                ok = backend_ctx->capture_begin(backend_ctx->backend_configs[j].backend);
                if (ok) {
                    begun++;
                }
            }
            if (!ok) {
                for (size_t j = 0; j < begun; j++) {
                    backend_ctx->capture_abort(backend_ctx->backend_configs[j].backend);
                }
                og->capturable = false;
            } else {
                outer_capture_active = true;
            }
        } else {
            og->warm++;
        }
    }

    const size_t iga0 = iga;
    const size_t ina0 = ina;

    if (!backend_ctx->launch_pool_started) {
        backend_ctx->launch_pool_start();
    }
    // Per-stage CUDA graphs (GGML_META_STAGE_GRAPH=1): capture each subgraph's
    // ~thousands of kernel launches into one executable graph and launch it
    // once. Capture does not touch the device's pending-launch queue, so the
    // host can enqueue arbitrarily far ahead — this removes the launch-queue
    // wall that caps prefill pipeline depth at ~2 stages. Prefill passes only
    // (uid == 0; decode/verify passes go through the outer capture instead).
    static const bool stage_graph_env = [] {
        const char * e = getenv("GGML_META_STAGE_GRAPH");
        return e != nullptr && atoi(e) != 0;
    }();
    bool stage_graph = stage_graph_env && !tp_spike_dump && !outer_capture_active &&
        backend_ctx->capture_begin && backend_ctx->capture_end &&
        backend_ctx->capture_launch && backend_ctx->capture_free && backend_ctx->capture_abort;
    if (stage_graph) {
        // prefill passes only: capture+instantiate overhead (~10ms/stage) is
        // amortized over large ubatches, not per decode token. Detect prefill
        // by batch width — decode/verify passes stay <= ~16 tokens. Width is
        // read from MATMUL nodes only: their ne[1] is the true token count,
        // while cache writes (SET_ROWS dst = whole cache) and kq masks (query
        // dim padded to 64) carry large ne[1] and misclassified GLM-DSA spec
        // verify batches (T<=8) as prefill — ~500 stage captures PER DECODE
        // STEP, tripling verify cost (measured 170ms vs 62ms/step at 8.5K).
        int64_t max_batch = 0;
        const int n_scan = std::min(cgraph->n_nodes, 48);
        for (int k = 0; k < n_scan; k++) {
            const ggml_tensor * nk = cgraph->nodes[k];
            if (nk->op != GGML_OP_MUL_MAT && nk->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            max_batch = std::max(max_batch, nk->ne[1]);
        }
        stage_graph = max_batch >= 64;
    }
    // stream capture is thread-sensitive: workers must stand down while an
    // outer-capture pass is recording; stage capture also takes precedence
    const bool parallel_launch = !backend_ctx->launch_workers.empty() && !tp_spike_dump &&
        !outer_capture_active && !stage_graph;

    static const int stage_keep_execs = [](){
        const char * e = getenv("GGML_META_STAGE_KEEP_EXECS");
        return e ? atoi(e) : 0;
    }();
    // free execs kept alive since the previous call on this pair
    if (!backend_ctx->kept_execs.empty()) {
        for (auto & ke : backend_ctx->kept_execs) {
            backend_ctx->capture_free(ke.first, ke.second);
        }
        backend_ctx->kept_execs.clear();
    }

    for (int pass = 0; ; pass++) {

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        const int64_t t0 = tp_time ? ggml_time_us() : 0;
        if (parallel_launch) {
            // members 1..n-1 enqueue on persistent workers, member 0 here
            for (size_t j = 1; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                backend_ctx->launch_workers[j - 1]->arm(bcj.backend, bcj.cgraphs[i].cgraph_main);
            }
            auto & bc0 = backend_ctx->backend_configs[0];
            const ggml_status status0 = ggml_backend_graph_compute_async(bc0.backend, bc0.cgraphs[i].cgraph_main);
            ggml_status status_all = status0;
            for (size_t j = 1; j < n_backends; j++) {
                const ggml_status sj = backend_ctx->launch_workers[j - 1]->join();
                if (sj != GGML_STATUS_SUCCESS) {
                    status_all = sj;
                }
            }
            if (status_all != GGML_STATUS_SUCCESS) {
                return status_all;
            }
        } else {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bool captured = false;
            static const int stage_min_nodes = [](){
                const char * e = getenv("GGML_META_STAGE_MIN_NODES");
                const int v = e ? atoi(e) : 0;
                return v > 0 ? v : 8;
            }();
            // Bisect knob for the stage-capture x P2P-gather interaction:
            // GGML_META_STAGE_SKIP=1 -> no captures in graphs containing any
            // gather boundary; =2 -> skip only the producer (ends at gather)
            // and consumer (follows gather) subgraphs.
            static const int stage_skip = [](){
                const char * e = getenv("GGML_META_STAGE_SKIP");
                return e ? atoi(e) : 0;
            }();
            // Range bisect: capture only subgraphs with IMIN <= i <= IMAX.
            static const int stage_imin = [](){
                const char * e = getenv("GGML_META_STAGE_IMIN");
                return e ? atoi(e) : 0;
            }();
            static const int stage_imax = [](){
                const char * e = getenv("GGML_META_STAGE_IMAX");
                return e ? atoi(e) : INT32_MAX;
            }();
            // Pair bisect: capture only in the meta context with this
            // creation ordinal (pair index); -1 = all pairs.
            static const int stage_pair = [](){
                const char * e = getenv("GGML_META_STAGE_PAIR");
                return e ? atoi(e) : -1;
            }();
            bool skip_capture = (int) i < stage_imin || (int) i > stage_imax ||
                (stage_pair >= 0 && backend_ctx->ctx_ord != stage_pair);
            // One-shot node listing of a chosen subgraph (bisect aid):
            static const int dump_sg = [](){
                const char * e = getenv("GGML_META_STAGE_DUMP_SUBGRAPH");
                return e ? atoi(e) : -1;
            }();
            if (dump_sg >= 0 && (int) i == dump_sg && j == 0) {
                static int sg_dumped = 0;
                if (sg_dumped < 2) {
                    sg_dumped++;
                    ggml_cgraph * cgd = bcj.cgraphs[i].cgraph_main;
                    fprintf(stderr, "[SGDUMP] subgraph %zu n_nodes=%d:", i, cgd->n_nodes);
                    for (int nn = 0; nn < cgd->n_nodes; nn++) {
                        fprintf(stderr, " %s(%s)", ggml_op_name(cgd->nodes[nn]->op), cgd->nodes[nn]->name);
                    }
                    fprintf(stderr, "\n");
                    fflush(stderr);
                }
            }
            if (stage_skip == 1) {
                for (size_t g = 0; g < backend_ctx->subgraph_gather.size(); g++) {
                    if (backend_ctx->subgraph_gather[g] != 0) { skip_capture = true; break; }
                }
            } else if (stage_skip == 2) {
                const bool ends_at_gather    = i < backend_ctx->subgraph_gather.size() && backend_ctx->subgraph_gather[i] != 0;
                const bool follows_gather    = i > 0 && i - 1 < backend_ctx->subgraph_gather.size() && backend_ctx->subgraph_gather[i - 1] != 0;
                skip_capture = ends_at_gather || follows_gather;
            } else if (stage_skip == 3) {
                // CUB-theory bisect: skip capture for stages containing ops
                // whose CUDA impls go through CUB device algorithms (argsort/
                // top-k/cumsum/sum/mean) — suspected silently capture-unsafe.
                ggml_cgraph * cgs = bcj.cgraphs[i].cgraph_main;
                for (int nn = 0; nn < cgs->n_nodes; nn++) {
                    const ggml_op op = cgs->nodes[nn]->op;
                    if (op == GGML_OP_ARGSORT || op == GGML_OP_TOP_K ||
                        op == GGML_OP_CUMSUM || op == GGML_OP_SUM ||
                        op == GGML_OP_MEAN) {
                        skip_capture = true;
                        static std::atomic<int> n_cub_skips{0};
                        const int ns = ++n_cub_skips;
                        if (ns <= 2 || ns % 512 == 0) {
                            fprintf(stderr, "[STAGE-SKIP3] skipped CUB stage (op=%s) n=%d\n", ggml_op_name(op), ns);
                        }
                        break;
                    }
                }
            }
            if (!skip_capture && stage_graph && bcj.cgraphs[i].cgraph_main->n_nodes >= stage_min_nodes) {
                if (backend_ctx->capture_begin(bcj.backend)) {
                    const ggml_status cstatus = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                    if (cstatus != GGML_STATUS_SUCCESS) {
                        backend_ctx->capture_abort(bcj.backend);
                        return cstatus;
                    }
                    void * exec = backend_ctx->capture_end(bcj.backend);
                    if (exec != nullptr) {
                        // Bisect knob: host-drain the member right after each
                        // captured launch. Clean under drain => capture only
                        // shifts timing and exposes an external ordering race;
                        // corrupt under drain => the captured execution itself
                        // computes wrong data.
                        static const int stage_drain = [](){
                            const char * e = getenv("GGML_META_STAGE_DRAIN");
                            return e ? atoi(e) : 0;
                        }();
                        // Bisect knob: capture then DISCARD without launching,
                        // falling through to the plain enqueue. Corrupt =>
                        // the act of capturing perturbs stream state; clean =>
                        // the replayed launch is at fault.
                        static const int stage_nolaunch = [](){
                            const char * e = getenv("GGML_META_STAGE_NOLAUNCH");
                            return e ? atoi(e) : 0;
                        }();
                        if (stage_nolaunch) {
                            backend_ctx->capture_free(bcj.backend, exec);
                            exec = nullptr;
                        } else
                        if (backend_ctx->capture_launch(bcj.backend, exec)) {
                            captured = true;
                            if (stage_drain) {
                                ggml_backend_synchronize(bcj.backend);
                            }
                            static std::atomic<int> n_stage_captures{0};
                            const int n = ++n_stage_captures;
                            if (n == 1 || n % 512 == 0) {
                                fprintf(stderr, "META-STAGE-GRAPH: %d stage captures\n", n);
                            }
                        }
                        // cudaGraphExecDestroy defers destruction until the
                        // in-flight launch completes
                        if (exec != nullptr) {
                            if (stage_keep_execs) {
                                backend_ctx->kept_execs.emplace_back(bcj.backend, exec);
                            } else {
                                backend_ctx->capture_free(bcj.backend, exec);
                            }
                        }
                    }
                    // capture_end failure: nothing was executed; fall through
                    // to the plain enqueue below
                }
            }
            if (captured) {
                continue;
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            if (tp_spike_dump) {
                ggml_backend_synchronize(bcj.backend);
                ggml_cgraph * cg = bcj.cgraphs[i].cgraph_main;
                for (int n = 0; n < cg->n_nodes; n++) {
                    ggml_tensor * t = cg->nodes[n];
                    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->view_src != nullptr) {
                        continue;
                    }
                    const int64_t nel = ggml_nelements(t);
                    if (nel == 0) {
                        continue;
                    }
                    const int64_t n_sample = std::min<int64_t>(nel, 65536);
                    std::vector<float> buf(n_sample);
                    ggml_backend_tensor_get(t, buf.data(), 0, n_sample*sizeof(float));
                    double s = 0.0;
                    for (float x : buf) {
                        s += (double) x;
                    }
                    fprintf(stderr, "TPDUMP sub=%zu dev=%zu node=%d op=%s name=%s ne=[%lld,%lld,%lld] sum=%.6g vals=[%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g]\n",
                        i, j, n, ggml_op_name(t->op), t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], s,
                        n_sample > 0 ? buf[0] : 0.f, n_sample > 1 ? buf[1] : 0.f,
                        n_sample > 2 ? buf[2] : 0.f, n_sample > 3 ? buf[3] : 0.f,
                        n_sample > 4 ? buf[4] : 0.f, n_sample > 5 ? buf[5] : 0.f,
                        n_sample > 6 ? buf[6] : 0.f, n_sample > 7 ? buf[7] : 0.f);
                }
            }
        }
        }

        const int64_t t1 = tp_time ? ggml_time_us() : 0;

        const bool gather_boundary = n_backends > 1 &&
            i < backend_ctx->subgraph_gather.size() && backend_ctx->subgraph_gather[i] != 0;
        if (gather_boundary) {
            // Position-sharded KV gather. Per-member VIEW tensors of the
            // split cache mis-map row counts (a view's rows get distributed
            // proportionally), so ranges are computed from the PARENT cache's
            // split state instead: member j owns parent_ss.ne[j] rows, in
            // member order (rotation forced to 0 for the sharded cache).
            // Correctness-first host bounce; S2 replaces this boundary with
            // partial-attention merge.
            ggml_tensor * meta_g     = cgraph->nodes[backend_ctx->subgraph_gather_node[i]];
            ggml_tensor * meta_view  = meta_g->src[0];
            ggml_tensor * meta_cache = meta_view->view_src != nullptr ? meta_view->view_src : meta_view;
            GGML_ASSERT(meta_view->view_offs == 0 && "sharded KV gather expects views starting at row 0");
            const size_t row_bytes = meta_cache->nb[1];
            const int64_t view_rows = (int64_t) (ggml_nbytes(meta_g) / row_bytes);
            const ggml_backend_meta_split_state cache_ss = ggml_backend_meta_get_split_state(meta_cache, /*assume_sync =*/ false);
            GGML_ASSERT(cache_ss.n_segments == 1);

            std::vector<size_t> take_bytes(n_backends);
            {
                int64_t rows_left = view_rows;
                for (size_t j = 0; j < n_backends; j++) {
                    const int64_t take = std::min(rows_left, cache_ss.ne[j]);
                    rows_left -= take;
                    take_bytes[j] = (size_t) take * row_bytes;
                }
                GGML_ASSERT(rows_left == 0);
            }

            static const int gdump = [](){
                const char * e = getenv("LLAMA_KV_SHARD_DUMP");
                return e ? atoi(e) : 0;
            }();

            if (gdump <= 0) {
                // S2a fast path: exchange cache halves member-to-member with
                // stream-ordered device copies; the host never blocks.
                //
                // Ordering requirements, both satisfied on-device via events:
                // 1. A copy must run after the SRC member's subgraph<=i kernels
                //    (its set_rows) - guaranteed by issuing on the src stream.
                // 2. A copy writing member k's gather node must run after
                //    member k's own in-flight subgraph<=i kernels, because the
                //    gather node's galloc region can alias their memory (the
                //    S1 host-write race). Guaranteed by making every src
                //    stream wait on every other member's recorded event first.
                // Subsequent subgraphs on the dst stream are ordered after the
                // incoming copy by cpy_tensor_async's own event chain.
                auto & gev = backend_ctx->gather_events;
                if (gev.size() < n_backends) {
                    gev.resize(n_backends, nullptr);
                }
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    if (gev[j] == nullptr) {
                        gev[j] = ggml_backend_event_new(ggml_backend_get_device(bcj.backend));
                        GGML_ASSERT(gev[j] != nullptr);
                    }
                    ggml_backend_event_record(gev[j], bcj.backend);
                }
                for (size_t j = 0; j < n_backends; j++) {
                    for (size_t k = 0; k < n_backends; k++) {
                        if (j != k) {
                            ggml_backend_event_wait(backend_ctx->backend_configs[j].backend, gev[k]);
                        }
                    }
                }
                for (size_t idst = 0; idst < n_backends; idst++) {
                    auto & bci = backend_ctx->backend_configs[idst];
                    ggml_cgraph * cgi = bci.cgraphs[i].cgraph_main;
                    ggml_tensor * gnode = cgi->nodes[cgi->n_nodes - 1];
                    size_t off_j = 0;
                    for (size_t j = 0; j < n_backends; j++) {
                        if (take_bytes[j] == 0) {
                            continue;
                        }
                        ggml_tensor * cache_j = ggml_backend_meta_buffer_simple_tensor(meta_cache, j);
                        ggml_tensor tsrc{};
                        tsrc.type = GGML_TYPE_I8;
                        tsrc.ne[0] = (int64_t) take_bytes[j]; tsrc.ne[1] = tsrc.ne[2] = tsrc.ne[3] = 1;
                        tsrc.nb[0] = 1; tsrc.nb[1] = tsrc.nb[2] = tsrc.nb[3] = take_bytes[j];
                        tsrc.data   = cache_j->data;
                        tsrc.buffer = cache_j->buffer;
                        ggml_tensor tdst = tsrc;
                        tdst.data   = (char *) gnode->data + off_j;
                        tdst.buffer = gnode->buffer;
                        ggml_backend_tensor_copy_async(backend_ctx->backend_configs[j].backend, bci.backend, &tsrc, &tdst);
                        off_j += take_bytes[j];
                    }
                }
                // Bisect knob: GGML_META_GATHER_SYNC=1 drains both members
                // after the copies (makes the P2P path as strongly ordered as
                // the S1 host bounce, at full-stall cost).
                static const int gsync = [](){
                    const char * e = getenv("GGML_META_GATHER_SYNC");
                    return e ? atoi(e) : 0;
                }();
                if (gsync != 0) {
                    for (size_t j = 0; j < n_backends; j++) {
                        ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    }
                }
            } else {
                // Debug path (LLAMA_KV_SHARD_DUMP=<nrows>): host bounce with
                // full member syncs + block checksums of the exchanged rows.
                std::vector<std::vector<char>> bounce(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    if (take_bytes[j] == 0) {
                        continue;
                    }
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_backend_synchronize(bcj.backend);
                    ggml_tensor * cache_j = ggml_backend_meta_buffer_simple_tensor(meta_cache, j);
                    bounce[j].resize(take_bytes[j]);
                    ggml_backend_tensor_get(cache_j, bounce[j].data(), 0, take_bytes[j]);
                }
                // sync ALL members before host-writing gather nodes (galloc
                // aliasing - see fast-path comment; this was the S1 race)
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }
                fprintf(stderr, "[KVSGATH] %s view_rows=%lld row_bytes=%zu ss.ne={%lld,%lld}\n",
                        meta_g->name, (long long) view_rows, row_bytes,
                        (long long) cache_ss.ne[0], (long long) (n_backends > 1 ? cache_ss.ne[1] : 0));
                for (size_t j = 0; j < n_backends; j++) {
                    const int64_t rows_j = std::min<int64_t>((int64_t) (take_bytes[j] / row_bytes), gdump);
                    fprintf(stderr, "[KVSGATH]   member%zu take_rows=%zu blocks16:", j, take_bytes[j] / row_bytes);
                    for (int64_t r0 = 0; r0 < rows_j; r0 += 16) {
                        uint64_t h = 1469598103934665603ULL;
                        const size_t lo = (size_t) r0 * row_bytes;
                        const size_t hi = std::min((size_t) take_bytes[j], (size_t) (r0 + 16) * row_bytes);
                        for (size_t k = lo; k < hi; k++) {
                            h = (h ^ (uint8_t) bounce[j][k]) * 1099511628211ULL;
                        }
                        fprintf(stderr, " %03lld:%08x", (long long) r0, (uint32_t) (h & 0xffffffff));
                    }
                    fprintf(stderr, "\n");
                }
                fflush(stderr);
                for (size_t idst = 0; idst < n_backends; idst++) {
                    auto & bci = backend_ctx->backend_configs[idst];
                    ggml_cgraph * cgi = bci.cgraphs[i].cgraph_main;
                    ggml_tensor * gnode = cgi->nodes[cgi->n_nodes - 1];
                    size_t off_j = 0;
                    for (size_t j = 0; j < n_backends; j++) {
                        if (take_bytes[j] > 0) {
                            ggml_backend_tensor_set(gnode, bounce[j].data(), off_j, take_bytes[j]);
                            off_j += take_bytes[j];
                        }
                    }
                }
            }
        }

        if (!gather_boundary && n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        if (tp_time) {
            const int64_t t2 = ggml_time_us();
            t_launch_us += t1 - t0;
            t_reduce_us += t2 - t1;
        }
    }

    if (!outer_capture_active) {
        break;
    }
    // End capture on every device and instantiate.  Capture only records --
    // nothing has executed yet on ANY device -- so an end/instantiate failure
    // can be recovered by re-running the whole pass eagerly.
    {
        og->execs.assign(n_backends, nullptr);
        bool end_ok = true;
        for (size_t j = 0; j < n_backends; j++) {
            og->execs[j] = backend_ctx->capture_end(backend_ctx->backend_configs[j].backend);
            end_ok = end_ok && og->execs[j] != nullptr;
        }
        if (end_ok) {
            // Execute this pass for real via the freshly captured graphs.
            for (size_t j = 0; j < n_backends; j++) {
                if (!backend_ctx->capture_launch(backend_ctx->backend_configs[j].backend, og->execs[j])) {
                    GGML_ABORT("meta outer-graph launch failed right after instantiate");
                }
            }
            break;
        }
        for (size_t j = 0; j < og->execs.size(); j++) {
            if (og->execs[j] != nullptr) {
                backend_ctx->capture_free(backend_ctx->backend_configs[j].backend, og->execs[j]);
            }
        }
        og->execs.clear();
        og->capturable = false;
        outer_capture_active = false;
        iga = iga0;
        ina = ina0;
        GGML_ASSERT(pass == 0);
        // fall through: loop again, executing eagerly this time
    }
    }

    if (tp_submit_timing) {
        // no sync: reports host-side blocking inside this compute call only
        const int64_t t_host = ggml_time_us() - t_start_us;
        if (t_host > 100000) {
            fprintf(stderr, "TPSUBMIT host=%.1fms launch=%.1fms reduce=%.1fms subgraphs=%zu\n",
                    t_host/1000.0, t_launch_us/1000.0, t_reduce_us/1000.0, backend_ctx->n_subgraphs);
            fflush(stderr);
        }
    }

    if (tp_spike_timing) {
        // synchronize to attribute the GPU-side tail to this compute call
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
        }
        const int64_t t_total = ggml_time_us() - t_start_us;
        static int64_t acc_total = 0, acc_launch = 0, acc_reduce = 0;
        static int     n_tok = 0;
        acc_total += t_total; acc_launch += t_launch_us; acc_reduce += t_reduce_us;
        n_tok++;
        if (n_tok % 50 == 0) {
            fprintf(stderr, "TPTIMING n=%d subgraphs=%zu avg_total=%lldus host_launch=%lldus host_reduce=%lldus gpu_tail=%lldus\n",
                n_tok, backend_ctx->n_subgraphs,
                (long long)(acc_total/n_tok), (long long)(acc_launch/n_tok), (long long)(acc_reduce/n_tok),
                (long long)((acc_total - acc_launch - acc_reduce)/n_tok));
        }
    }

    // LLAMA_GLM_SPARSE_DUMP: dump layer-0 DSA indexer_score / top_k selection.
    // Works through the meta backend (unlike the sched eval callback, which never
    // sees nodes internal to a meta split): scan this split's cgraph, sync, and
    // read the reconstructed tensor via the meta buffer's get_tensor.
    static const bool glm_dump = getenv("LLAMA_GLM_SPARSE_DUMP") != nullptr;
    if (glm_dump) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            const bool want = strcmp(node->name, "top_k-0") == 0 || strcmp(node->name, "indexer_score-0") == 0 || strcmp(node->name, "glm_mask-0") == 0 || strcmp(node->name, "glm_setrows-0") == 0;
            if (!want) {
                continue;
            }
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
            }
            const size_t esz = ggml_type_size(node->type);
            size_t cap = ggml_nelements(node);
            if (cap > 4096) cap = 4096;
            while (esz > 0 && cap * esz > ggml_nbytes(node)) cap--;   // bounds-safe
            if (node->type == GGML_TYPE_I32) {
                std::vector<int32_t> v(cap);
                ggml_backend_tensor_get(node, v.data(), 0, cap * sizeof(int32_t));
                int32_t mn = v[0], mx = v[0];
                long lo = 0;
                for (size_t k = 0; k < cap; k++) { if (v[k] < mn) mn = v[k]; if (v[k] > mx) mx = v[k]; if (v[k] >= 0 && v[k] < 64) lo++; }
                fprintf(stderr, "[GLMDUMP] %s I32 ne=[%lld,%lld] min=%d max=%d (#<64=%ld) first24=",
                        node->name, (long long)node->ne[0], (long long)node->ne[1], mn, mx, lo);
                for (size_t k = 0; k < 24 && k < cap; k++) fprintf(stderr, "%d ", v[k]);
                fprintf(stderr, "\n"); fflush(stderr);
            } else if (node->type == GGML_TYPE_F16) {
                // mask: F16 zero (0x0000) = unmasked, 0xFC00 = -inf (masked)
                std::vector<uint16_t> v(cap);
                ggml_backend_tensor_get(node, v.data(), 0, cap * sizeof(uint16_t));
                long unmasked = 0, masked = 0;
                for (size_t k = 0; k < cap; k++) { if (v[k] == 0) unmasked++; else masked++; }
                fprintf(stderr, "[GLMDUMP] %s F16 ne=[%lld,%lld] unmasked(==0)=%ld masked=%ld pos0_31=",
                        node->name, (long long)node->ne[0], (long long)node->ne[1], unmasked, masked);
                for (size_t k = 0; k < 32 && k < cap; k++) fprintf(stderr, "%c", v[k] == 0 ? 'U' : '.');
                fprintf(stderr, "\n"); fflush(stderr);
            } else {
                std::vector<float> v(cap);
                ggml_backend_tensor_get(node, v.data(), 0, cap * sizeof(float));
                float mn = 1e30f, mx = -1e30f; long unmasked = 0, masked = 0;
                for (size_t k = 0; k < cap; k++) {
                    const float x = v[k];
                    if (x > -1e30f && x < mn) mn = x;
                    if (x > mx && x < 1e30f) mx = x;
                    if (x == 0.0f) unmasked++; else masked++;
                }
                fprintf(stderr, "[GLMDUMP] %s F32 ne=[%lld,%lld] finite_min=%.4f max=%.4f unmasked(==0)=%ld masked=%ld pos0_23=",
                        node->name, (long long)node->ne[0], (long long)node->ne[1], mn, mx, unmasked, masked);
                for (size_t k = 0; k < 24 && k < cap; k++) fprintf(stderr, "%c", v[k] == 0.0f ? 'U' : (v[k] <= -1e30f ? '.' : '?'));
                fprintf(stderr, "\n"); fflush(stderr);
            }
        }
    }

    // KV-shard contents-diff harness: LLAMA_KV_SHARD_DUMP=<nrows> dumps
    // per-member block checksums of the layer-0 K cache after every graph.
    // Works under both KVSHARD=0 (mirrored) and =1 (sharded) so outputs can
    // be diffed directly. Value-gated (atoi>0), never enabled by presence.
    {
        static const int dump_rows = [](){
            const char * e = getenv("LLAMA_KV_SHARD_DUMP");
            return e ? atoi(e) : 0;
        }();
        if (dump_rows > 0) {
            std::vector<ggml_tensor *> caches;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * n = cgraph->nodes[i];
                if (n->op != GGML_OP_SET_ROWS || n->src[2] == nullptr) {
                    continue;
                }
                ggml_tensor * dst = n->src[2];
                ggml_tensor * root = dst->view_src != nullptr ? dst->view_src : dst;
                if (strstr(root->name, "cache_k_l") != nullptr &&
                        std::find(caches.begin(), caches.end(), root) == caches.end()) {
                    caches.push_back(root);
                }
            }
            for (ggml_tensor * meta_cache : caches) {
                const ggml_backend_meta_split_state css = ggml_backend_meta_get_split_state(meta_cache, /*assume_sync =*/ false);
                const size_t row_bytes = meta_cache->nb[1];
                fprintf(stderr, "[KVSDUMP] graph=%d cache=%s axis=%d ne1=%lld ss.ne={%lld,%lld} row_bytes=%zu\n",
                        (int) cgraph->uid, meta_cache->name, (int) css.axis, (long long) meta_cache->ne[1],
                        (long long) css.ne[0], (long long) (n_backends > 1 ? css.ne[1] : 0), row_bytes);
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_tensor * cache_j = ggml_backend_meta_buffer_simple_tensor(meta_cache, j);
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    const int64_t rows_j = std::min<int64_t>(dump_rows, cache_j->ne[1]);
                    std::vector<char> buf((size_t) rows_j * row_bytes);
                    ggml_backend_tensor_get(cache_j, buf.data(), 0, buf.size());
                    fprintf(stderr, "[KVSDUMP]   member%zu ne1=%lld blocks16:", j, (long long) cache_j->ne[1]);
                    for (int64_t r0 = 0; r0 < rows_j; r0 += 16) {
                        uint64_t h = 1469598103934665603ULL;
                        const size_t lo = (size_t) r0 * row_bytes;
                        const size_t hi = std::min(buf.size(), (size_t) (r0 + 16) * row_bytes);
                        for (size_t k = lo; k < hi; k++) {
                            h = (h ^ (uint8_t) buf[k]) * 1099511628211ULL;
                        }
                        fprintf(stderr, " %03lld:%08x", (long long) r0, (uint32_t) (h & 0xffffffff));
                    }
                    fprintf(stderr, "\n");
                }
                fflush(stderr);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}


// MCPY-DIAG: counters for the pipeline-overlap machinery. Printed every 64
// async-copy calls and on the first few failures, to establish whether the
// sched routes inter-stage copies through the async path at all.
static int mcpy_calls = 0, mcpy_ok = 0, mcpy_fail[5] = {0,0,0,0,0};
static int mev_record = 0, mev_wait = 0;

static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    mev_record++;
    ggml_backend_meta_context * meta_ctx = (ggml_backend_meta_context *) backend->context;
    auto * evs = (std::vector<ggml_backend_event_t> *) event->context;
    GGML_ASSERT(evs->size() == meta_ctx->backend_configs.size());
    for (size_t i = 0; i < evs->size(); i++) {
        ggml_backend_event_record((*evs)[i], meta_ctx->backend_configs[i].backend);
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    mev_wait++;
    ggml_backend_meta_context * meta_ctx = (ggml_backend_meta_context *) backend->context;
    auto * evs = (std::vector<ggml_backend_event_t> *) event->context;
    // Cross-stage wait: every member stream of this meta backend waits on every
    // member event of the (possibly different) source meta device. CUDA stream
    // waits are cross-device capable.
    for (auto & bc : meta_ctx->backend_configs) {
        for (ggml_backend_event_t e : *evs) {
            ggml_backend_event_wait(bc.backend, e);
        }
    }
}


static bool ggml_backend_meta_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    // Inter-stage handoff for the TPxPP hybrid: without this, the generic
    // fallback host-synchronizes BOTH stages on every split boundary, which
    // serializes the whole pipeline. Delegate member-wise to the simple
    // backends' async copies (CUDA handles cross-device ordering + peer DMA).
    mcpy_calls++;
    static const bool mcpy_diag = getenv("MCPY_DIAG") != nullptr;
    if (mcpy_diag && mcpy_calls % 64 == 0) {
        fprintf(stderr, "MCPY-DIAG calls=%d ok=%d fail[nonmeta,n,buf,axis,member]=%d,%d,%d,%d,%d ev_rec=%d ev_wait=%d\n",
                mcpy_calls, mcpy_ok, mcpy_fail[0], mcpy_fail[1], mcpy_fail[2], mcpy_fail[3], mcpy_fail[4],
                mev_record, mev_wait);
    }
    if (!ggml_backend_is_meta(backend_src) || !ggml_backend_is_meta(backend_dst)) {
        // Host -> meta input uploads (kq_mask, positions, ...): route through
        // the member-wise async set path instead of failing into the sched's
        // synchronous fallback (event_synchronize + device-syncing plain copy
        // per split input, ~2 dozen host blocks per ubatch during prefill).
        if (ggml_backend_is_meta(backend_dst) &&
                dst->buffer != nullptr && ggml_backend_buffer_is_meta(dst->buffer) &&
                src->buffer != nullptr && ggml_backend_buffer_is_host(src->buffer) &&
                ggml_is_contiguous(src) && ggml_is_contiguous(dst) &&
                ggml_are_same_shape(src, dst) && src->type == dst->type) {
            // only for split states set_tensor_async can map; anything else
            // (PARTIAL, multi-segment, replicated rows) keeps the safe fallback
            const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(dst, /*assume_sync =*/ false);
            const bool supported = ss.n_segments == 1 && ss.nr[0] == 1 &&
                (ss.axis == GGML_BACKEND_SPLIT_AXIS_0 || ss.axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                 ss.axis == GGML_BACKEND_SPLIT_AXIS_2 || ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            if (supported) {
                ggml_backend_meta_set_tensor_async(backend_dst, dst, src->data, 0, ggml_nbytes(src));
                mcpy_ok++;
                return true;
            }
        }
        mcpy_fail[0]++;
        return false;
    }
    const size_t n_src = ggml_backend_meta_n_backends(backend_src);
    const size_t n_dst = ggml_backend_meta_n_backends(backend_dst);
    if (n_src != n_dst) {
        mcpy_fail[1]++;
        return false;
    }
    if (!ggml_backend_buffer_is_meta(src->buffer) || !ggml_backend_buffer_is_meta(dst->buffer)) {
        mcpy_fail[2]++;
        return false;
    }
    // Only mirror-to-mirror copies are member-wise correct; anything else
    // falls back to the safe synchronous gather path.
    if (ggml_backend_meta_get_split_state(src, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
        ggml_backend_meta_get_split_state(dst, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        mcpy_fail[3]++;
        return false;
    }
    for (size_t i = 0; i < n_src; i++) {
        ggml_tensor * s = ggml_backend_meta_buffer_simple_tensor(const_cast<ggml_tensor *>(src), i);
        ggml_tensor * d = ggml_backend_meta_buffer_simple_tensor(dst, i);
        if (s == nullptr || d == nullptr) {
            mcpy_fail[4]++;
            return false;
        }
        ggml_backend_tensor_copy_async(
            ggml_backend_meta_simple_backend(backend_src, i),
            ggml_backend_meta_simple_backend(backend_dst, i),
            s, d);
    }
    mcpy_ok++;
    return true;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_meta_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
