#include <cuda_runtime.h>
#include <cudnn.h>

#include "cudnn_frontend.h"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t PAGE_TABLE_K_UID = 5;
constexpr int64_t PAGE_TABLE_V_UID = 6;
constexpr int64_t SEQ_LEN_Q_UID = 7;
constexpr int64_t SEQ_LEN_KV_UID = 8;
constexpr int64_t QO_RAGGED_OFFSET_UID = 9;

enum class AlignMode {
    TOP_LEFT,
    BOTTOM_RIGHT,
};

enum class QStrideMode {
    SAMPLE_BHSD,   // [b,h,s,d] with standard BHSD contiguous strides
    RUNNER_TOKEN,  // runner-style token-major reinterpret stride
};

enum class KVStrideMode {
    SAMPLE_BHSD,   // [blk,h,bs,d] standard strides
    RUNNER_BSHD,   // runner current strides (head/seq swapped)
};

struct ProbeCase {
    std::string name;
    fe::AttentionImplementation_t impl;
    AlignMode align;
    std::optional<int32_t> right_bound;
    bool ragged_q{true};
    bool ragged_o{false};
    QStrideMode q_stride{QStrideMode::SAMPLE_BHSD};
    KVStrideMode kv_stride{KVStrideMode::SAMPLE_BHSD};
    bool dynamic_shape{false};
    std::vector<fe::HeurMode_t> heur_modes{fe::HeurMode_t::A};
};

struct ProbeShape {
    int64_t b{1};
    int64_t hq{12};
    int64_t hk{2};
    int64_t hv{2};
    int64_t sq{9};
    int64_t skv{16};
    int64_t d{128};
    int64_t block_size{16};
    int64_t num_blocks{2048};
    int64_t table_size{1};
    float scale{1.0f / std::sqrt(128.0f)};
};

const char *impl_name(fe::AttentionImplementation_t impl) {
    switch (impl) {
    case fe::AttentionImplementation_t::AUTO:
        return "AUTO";
    case fe::AttentionImplementation_t::COMPOSITE:
        return "COMPOSITE";
    case fe::AttentionImplementation_t::UNIFIED:
        return "UNIFIED";
    }
    return "UNKNOWN";
}

const char *align_name(AlignMode m) {
    return m == AlignMode::TOP_LEFT ? "TOP_LEFT" : "BOTTOM_RIGHT";
}

std::string heur_modes_name(const std::vector<fe::HeurMode_t> &modes) {
    std::string out;
    for (size_t i = 0; i < modes.size(); ++i) {
        if (i > 0) {
            out += "+";
        }
        switch (modes[i]) {
        case fe::HeurMode_t::A:
            out += "A";
            break;
        case fe::HeurMode_t::B:
            out += "B";
            break;
        case fe::HeurMode_t::FALLBACK:
            out += "FALLBACK";
            break;
        default:
            out += "?";
            break;
        }
    }
    return out;
}

void check_cuda(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) {
        return;
    }
    std::cerr << "[probe] CUDA error at " << what << ": " << cudaGetErrorString(rc) << std::endl;
    std::exit(2);
}

void check_cudnn(cudnnStatus_t rc, const char *what) {
    if (rc == CUDNN_STATUS_SUCCESS) {
        return;
    }
    std::cerr << "[probe] cuDNN error at " << what << ": " << cudnnGetErrorString(rc) << std::endl;
    std::exit(2);
}

std::shared_ptr<fe::graph::Graph> build_graph(const ProbeShape &shape, const ProbeCase &cfg) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(cfg.dynamic_shape);

    std::shared_ptr<fe::graph::Tensor_attributes> q_ragged = nullptr;
    if (cfg.ragged_q) {
        q_ragged = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("QO_ragged_offset")
                                     .set_uid(QO_RAGGED_OFFSET_UID)
                                     .set_dim({shape.b + 1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
    }

    auto q_attrs = fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID).set_dim({shape.b, shape.hq, shape.sq, shape.d});
    if (cfg.q_stride == QStrideMode::SAMPLE_BHSD) {
        q_attrs.set_stride({shape.hq * shape.sq * shape.d, shape.sq * shape.d, shape.d, 1});
    } else {
        // Matches current NovaInfer runner graph settings.
        q_attrs.set_stride({shape.hq * shape.d, shape.d, shape.hq * shape.d, 1});
    }
    if (q_ragged != nullptr) {
        q_attrs.set_ragged_offset(q_ragged);
    }
    auto Q = graph->tensor(q_attrs);

    auto k_attrs = fe::graph::Tensor_attributes()
                       .set_name("container_K")
                       .set_uid(K_UID)
                       .set_dim({shape.num_blocks, shape.hk, shape.block_size, shape.d});
    auto v_attrs = fe::graph::Tensor_attributes()
                       .set_name("container_V")
                       .set_uid(V_UID)
                       .set_dim({shape.num_blocks, shape.hv, shape.block_size, shape.d});
    if (cfg.kv_stride == KVStrideMode::SAMPLE_BHSD) {
        k_attrs.set_stride({shape.hk * shape.block_size * shape.d, shape.block_size * shape.d, shape.d, 1});
        v_attrs.set_stride({shape.hv * shape.block_size * shape.d, shape.block_size * shape.d, shape.d, 1});
    } else {
        // Matches current NovaInfer runner graph settings.
        k_attrs.set_stride({shape.hk * shape.block_size * shape.d, shape.d, shape.hk * shape.d, 1});
        v_attrs.set_stride({shape.hv * shape.block_size * shape.d, shape.d, shape.hv * shape.d, 1});
    }
    auto K = graph->tensor(k_attrs);
    auto V = graph->tensor(v_attrs);

    auto seq_q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("seq_q")
                                   .set_uid(SEQ_LEN_Q_UID)
                                   .set_dim({shape.b, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_kv")
                                    .set_uid(SEQ_LEN_KV_UID)
                                    .set_dim({shape.b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    auto page_k = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("page_table_k")
                                    .set_uid(PAGE_TABLE_K_UID)
                                    .set_dim({shape.b, 1, shape.table_size, 1})
                                    .set_stride({shape.table_size, shape.table_size, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    auto page_v = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("page_table_v")
                                    .set_uid(PAGE_TABLE_V_UID)
                                    .set_dim({shape.b, 1, shape.table_size, 1})
                                    .set_stride({shape.table_size, shape.table_size, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));

    auto sdpa = fe::graph::SDPA_attributes()
                    .set_name("probe_sdpa")
                    .set_generate_stats(false)
                    .set_attn_scale(shape.scale)
                    .set_implementation(cfg.impl);
    if (cfg.align == AlignMode::TOP_LEFT) {
        sdpa.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT);
    } else {
        sdpa.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT);
    }
    if (cfg.right_bound.has_value()) {
        sdpa.set_diagonal_band_right_bound(*cfg.right_bound);
    }
    sdpa.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    sdpa.set_paged_attention_k_table(page_k);
    sdpa.set_paged_attention_v_table(page_v);
    sdpa.set_paged_attention_max_seq_len_kv(static_cast<int>(shape.skv));

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa);
    (void)Stats;
    O->set_output(true).set_uid(O_UID).set_dim({shape.b, shape.hq, shape.sq, shape.d});
    if (cfg.q_stride == QStrideMode::SAMPLE_BHSD) {
        O->set_stride({shape.hq * shape.sq * shape.d, shape.sq * shape.d, shape.d, 1});
    } else {
        O->set_stride({shape.hq * shape.d, shape.d, shape.hq * shape.d, 1});
    }
    if (cfg.ragged_q && cfg.ragged_o && q_ragged != nullptr) {
        O->set_ragged_offset(q_ragged);
    }
    return graph;
}

void run_case(cudnnHandle_t handle, const ProbeShape &shape, const ProbeCase &cfg) {
    std::cout << "[case] " << cfg.name << " impl=" << impl_name(cfg.impl) << " align=" << align_name(cfg.align)
              << " right_bound=" << (cfg.right_bound.has_value() ? std::to_string(*cfg.right_bound) : "none")
              << " ragged_q=" << int(cfg.ragged_q) << " ragged_o=" << int(cfg.ragged_o)
              << " q_stride=" << (cfg.q_stride == QStrideMode::SAMPLE_BHSD ? "sample" : "runner")
              << " kv_stride=" << (cfg.kv_stride == KVStrideMode::SAMPLE_BHSD ? "sample" : "runner")
              << " dynamic_shape=" << int(cfg.dynamic_shape)
              << " heur=" << heur_modes_name(cfg.heur_modes) << std::endl;

    auto graph = build_graph(shape, cfg);
    auto st = graph->build(handle, cfg.heur_modes);
    if (st.is_good()) {
        std::cout << "  -> PASS" << std::endl;
    } else {
        std::cout << "  -> FAIL code=" << static_cast<int>(st.get_code()) << " msg=" << st.get_message() << std::endl;
    }
}

} // namespace

int main() {
    int dev = 0;
    check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    std::cout << "[probe] device=" << dev << " name=" << prop.name << " sm=" << prop.major << prop.minor
              << " cudnn=" << cudnnGetVersion() << std::endl;

    cudnnHandle_t handle{};
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");

    ProbeShape shape{};
    std::cout << "[probe] shape_small"
              << " b=" << shape.b << " hq=" << shape.hq << " hk=" << shape.hk << " sq=" << shape.sq << " skv=" << shape.skv
              << " d=" << shape.d << " block_size=" << shape.block_size << " num_blocks=" << shape.num_blocks
              << " table_size=" << shape.table_size << std::endl;

    ProbeShape shape_large = shape;
    shape_large.num_blocks = 65536;
    std::cout << "[probe] shape_large"
              << " b=" << shape_large.b << " hq=" << shape_large.hq << " hk=" << shape_large.hk << " sq=" << shape_large.sq
              << " skv=" << shape_large.skv << " d=" << shape_large.d << " block_size=" << shape_large.block_size
              << " num_blocks=" << shape_large.num_blocks << " table_size=" << shape_large.table_size << std::endl;

    std::vector<ProbeCase> cases = {
        // Official sample-like prefill
        {"sample_like_top_left_auto", fe::AttentionImplementation_t::AUTO, AlignMode::TOP_LEFT, 0, true, false,
         QStrideMode::SAMPLE_BHSD, KVStrideMode::SAMPLE_BHSD, false, {fe::HeurMode_t::A}},
        {"sample_like_top_left_composite", fe::AttentionImplementation_t::COMPOSITE, AlignMode::TOP_LEFT, 0, true, false,
         QStrideMode::SAMPLE_BHSD, KVStrideMode::SAMPLE_BHSD, false, {fe::HeurMode_t::A}},
        // Runner-like prefill
        {"runner_like_top_left_auto", fe::AttentionImplementation_t::AUTO, AlignMode::TOP_LEFT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, false, {fe::HeurMode_t::A}},
        {"runner_like_bottom_right_auto", fe::AttentionImplementation_t::AUTO, AlignMode::BOTTOM_RIGHT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, false, {fe::HeurMode_t::A}},
        {"runner_like_top_left_composite", fe::AttentionImplementation_t::COMPOSITE, AlignMode::TOP_LEFT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, false, {fe::HeurMode_t::A}},
        {"runner_like_bottom_right_composite", fe::AttentionImplementation_t::COMPOSITE, AlignMode::BOTTOM_RIGHT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, false, {fe::HeurMode_t::A}},
        {"runner_like_bottom_right_composite_A_FALLBACK", fe::AttentionImplementation_t::COMPOSITE, AlignMode::BOTTOM_RIGHT, 0,
         true, true, QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, false, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}},
        // Mixed stride sanity checks
        {"mixed_top_left_composite_sample_stride", fe::AttentionImplementation_t::COMPOSITE, AlignMode::TOP_LEFT, 0, true, true,
         QStrideMode::SAMPLE_BHSD, KVStrideMode::SAMPLE_BHSD, false, {fe::HeurMode_t::A}},
        {"mixed_bottom_right_composite_sample_stride", fe::AttentionImplementation_t::COMPOSITE, AlignMode::BOTTOM_RIGHT, 0,
         true, true, QStrideMode::SAMPLE_BHSD, KVStrideMode::SAMPLE_BHSD, false, {fe::HeurMode_t::A}},
        // Dynamic-shape variants (closer to NovaInfer runtime graph path)
        {"runner_like_top_left_composite_dynamic", fe::AttentionImplementation_t::COMPOSITE, AlignMode::TOP_LEFT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, true, {fe::HeurMode_t::A}},
        {"runner_like_bottom_right_composite_dynamic", fe::AttentionImplementation_t::COMPOSITE, AlignMode::BOTTOM_RIGHT, 0, true, true,
         QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, true, {fe::HeurMode_t::A}},
        {"runner_like_bottom_right_composite_dynamic_A_FALLBACK", fe::AttentionImplementation_t::COMPOSITE, AlignMode::BOTTOM_RIGHT, 0,
         true, true, QStrideMode::RUNNER_TOKEN, KVStrideMode::RUNNER_BSHD, true, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}},
    };

    std::cout << "[probe] ---- testing shape_small ----" << std::endl;
    for (const auto &c : cases) {
        run_case(handle, shape, c);
    }

    std::cout << "[probe] ---- testing shape_large (num_blocks=65536) ----" << std::endl;
    for (const auto &c : cases) {
        run_case(handle, shape_large, c);
    }

    check_cudnn(cudnnDestroy(handle), "cudnnDestroy");
    return 0;
}
