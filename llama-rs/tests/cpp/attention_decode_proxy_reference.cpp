#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> make_input(int hidden_features, int sequence_length) {
    std::vector<float> values(static_cast<size_t>(hidden_features) *
                              static_cast<size_t>(sequence_length));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i + 3) % 29) * 0.0625f;
    }
    return values;
}

std::vector<float> make_q_weights(int hidden_features, int query_features) {
    std::vector<float> values(static_cast<size_t>(hidden_features) *
                              static_cast<size_t>(query_features));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i % 31) * 0.01f;
    }
    return values;
}

std::vector<float> make_k_weights(int hidden_features, int kv_features) {
    std::vector<float> values(static_cast<size_t>(hidden_features) *
                              static_cast<size_t>(kv_features));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i + 7) % 29) * 0.011f;
    }
    return values;
}

std::vector<float> make_v_weights(int hidden_features, int kv_features) {
    std::vector<float> values(static_cast<size_t>(hidden_features) *
                              static_cast<size_t>(kv_features));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i + 13) % 23) * 0.013f;
    }
    return values;
}

std::vector<float> make_o_weights(int hidden_features, int query_features) {
    std::vector<float> values(static_cast<size_t>(hidden_features) *
                              static_cast<size_t>(query_features));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i + 17) % 19) * 0.009f;
    }
    return values;
}

std::vector<float> project_input(const std::vector<float>& weights,
                                 const std::vector<float>& input,
                                 int hidden_features,
                                 int output_features,
                                 int sequence_length) {
    std::vector<float> output(static_cast<size_t>(output_features) *
                              static_cast<size_t>(sequence_length),
                              0.0f);
    for (int seq = 0; seq < sequence_length; ++seq) {
        const int input_base = seq * hidden_features;
        const int out_base = seq * output_features;
        for (int out_feature = 0; out_feature < output_features; ++out_feature) {
            const int weight_base = out_feature * hidden_features;
            float acc = 0.0f;
            for (int hidden = 0; hidden < hidden_features; ++hidden) {
                acc += weights[static_cast<size_t>(weight_base + hidden)] *
                       input[static_cast<size_t>(input_base + hidden)];
            }
            output[static_cast<size_t>(out_base + out_feature)] = acc;
        }
    }
    return output;
}

ggml_backend_t init_backend(const std::string& backend_name) {
    if (backend_name == "cpu") {
        if (auto* backend =
                ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)) {
            return backend;
        }
        return ggml_backend_init_by_name("CPU", nullptr);
    }
    if (backend_name == "metal") {
        if (auto* backend = ggml_backend_init_by_name("Metal", nullptr)) {
            return backend;
        }
        if (auto* backend = ggml_backend_init_by_name("MTL0", nullptr)) {
            return backend;
        }
        if (auto* backend =
                ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr)) {
            return backend;
        }
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9 && argc != 12) {
        std::cerr
            << "usage: attention_decode_proxy_reference <cpu|metal> <hidden> <q_heads> <kv_heads> <q_len> <kv_len> <warmup> <iters> [<stepwise_kv_start> <stepwise_steps> <past_start>]\n";
        return 2;
    }

    const std::string backend_name = argv[1];
    const int hidden_features = std::atoi(argv[2]);
    const int query_head_count = std::atoi(argv[3]);
    const int kv_head_count = std::atoi(argv[4]);
    const int query_length = std::atoi(argv[5]);
    const int key_value_length = std::atoi(argv[6]);
    const int warmup_iters = std::atoi(argv[7]);
    const int bench_iters = std::atoi(argv[8]);
    const bool stepwise_enabled = argc == 12;
    const int stepwise_kv_start = stepwise_enabled ? std::atoi(argv[9]) : 0;
    const int stepwise_steps = stepwise_enabled ? std::atoi(argv[10]) : 0;
    const int stepwise_past_start = stepwise_enabled ? std::atoi(argv[11]) : 0;

    if (hidden_features <= 0 || query_head_count <= 0 || kv_head_count <= 0 ||
        query_length <= 0 || key_value_length <= 0 || warmup_iters < 0 || bench_iters <= 0) {
        std::cerr << "invalid argument values\n";
        return 2;
    }
    if (hidden_features % query_head_count != 0) {
        std::cerr << "hidden must be divisible by q_heads\n";
        return 2;
    }
    if (query_head_count % kv_head_count != 0) {
        std::cerr << "q_heads must be divisible by kv_heads\n";
        return 2;
    }
    if (stepwise_enabled) {
        if (query_length != 1) {
            std::cerr << "stepwise mode requires q_len=1\n";
            return 2;
        }
        if (stepwise_kv_start <= 0 || stepwise_steps <= 0 || stepwise_past_start < 0) {
            std::cerr << "invalid stepwise arguments\n";
            return 2;
        }
        if (stepwise_past_start + 1 != stepwise_kv_start) {
            std::cerr << "stepwise mode expects stepwise_kv_start == past_start + 1\n";
            return 2;
        }
        const int required_kv = stepwise_kv_start + stepwise_steps - 1;
        if (required_kv != key_value_length) {
            std::cerr
                << "stepwise mode expects kv_len == stepwise_kv_start + stepwise_steps - 1\n";
            return 2;
        }
    }

    const int head_dimension = hidden_features / query_head_count;
    const int query_features = hidden_features;
    const int kv_features = head_dimension * kv_head_count;
    const int kv_group_size = query_head_count / kv_head_count;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));

    const auto total_start = std::chrono::steady_clock::now();

    ggml_backend_load_all();
    ggml_backend_t backend = init_backend(backend_name);
    if (!backend) {
        std::cerr << "failed to initialize backend: " << backend_name << "\n";
        return 2;
    }

    // Keep this intentionally generous to avoid build-time shape-specific tuning
    // in the reference benchmark.
    const size_t ctx_size = 512ull * 1024ull * 1024ull;
    std::vector<uint8_t> ctx_buffer(ctx_size);
    const ggml_init_params params{
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "ggml_init failed\n";
        ggml_backend_free(backend);
        return 2;
    }

    ggml_tensor* w_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_features, query_features);
    ggml_tensor* w_o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, query_features, hidden_features);
    ggml_tensor* x_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_features, query_length);
    ggml_tensor* k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_features, key_value_length);
    ggml_tensor* v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_features, key_value_length);
    ggml_tensor* mask = stepwise_enabled
                            ? ggml_new_tensor_2d(
                                  ctx, GGML_TYPE_F32, key_value_length, query_length)
                            : nullptr;

    ggml_tensor* q = ggml_mul_mat(ctx, w_q, x_q);
    if (!q) {
        std::cerr << "failed to build Q projection\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    const size_t q_row_stride = static_cast<size_t>(query_features) * sizeof(float);
    const size_t kv_row_stride = static_cast<size_t>(kv_features) * sizeof(float);
    const size_t o_row_stride = static_cast<size_t>(query_features) * sizeof(float);

    ggml_tensor* output_projection = nullptr;
    for (int head = 0; head < query_head_count; ++head) {
        const int key_head = head / kv_group_size;
        const size_t query_offset =
            static_cast<size_t>(head) * static_cast<size_t>(head_dimension) *
            static_cast<size_t>(query_length) * sizeof(float);
        const size_t key_offset =
            static_cast<size_t>(key_head) * static_cast<size_t>(head_dimension) *
            static_cast<size_t>(key_value_length) * sizeof(float);

        ggml_tensor* q_head =
            ggml_view_2d(ctx, q, head_dimension, query_length, q_row_stride, query_offset);
        ggml_tensor* k_head = ggml_view_2d(
            ctx, k, head_dimension, key_value_length, kv_row_stride, key_offset);
        ggml_tensor* v_head = ggml_view_2d(
            ctx, v, head_dimension, key_value_length, kv_row_stride, key_offset);

        ggml_tensor* scores = ggml_mul_mat(ctx, k_head, q_head);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask, attention_scale, 0.0f);
        ggml_tensor* v_t = ggml_transpose(ctx, v_head);
        ggml_tensor* v_t_cont = ggml_cont(ctx, v_t);
        ggml_tensor* head_output = ggml_mul_mat(ctx, v_t_cont, probs);
        ggml_tensor* w_o_head = ggml_view_2d(
            ctx, w_o, head_dimension, hidden_features, o_row_stride, query_offset);
        ggml_tensor* projected = ggml_mul_mat(ctx, w_o_head, head_output);
        output_projection = output_projection ? ggml_add(ctx, output_projection, projected)
                                             : projected;
    }

    if (!output_projection) {
        std::cerr << "failed to build attention output\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_projection);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::cerr << "ggml_backend_alloc_ctx_tensors failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    const auto q_weights = make_q_weights(hidden_features, query_features);
    const auto k_weights = make_k_weights(hidden_features, kv_features);
    const auto v_weights = make_v_weights(hidden_features, kv_features);
    const auto o_weights = make_o_weights(hidden_features, query_features);
    const auto query_input = make_input(hidden_features, query_length);
    const auto key_value_input = make_input(hidden_features, key_value_length);
    const auto projected_k = project_input(
        k_weights, key_value_input, hidden_features, kv_features, key_value_length);
    const auto projected_v = project_input(
        v_weights, key_value_input, hidden_features, kv_features, key_value_length);

    ggml_backend_tensor_set(w_q, q_weights.data(), 0, ggml_nbytes(w_q));
    ggml_backend_tensor_set(w_o, o_weights.data(), 0, ggml_nbytes(w_o));
    ggml_backend_tensor_set(x_q, query_input.data(), 0, ggml_nbytes(x_q));
    ggml_backend_tensor_set(k, projected_k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, projected_v.data(), 0, ggml_nbytes(v));

    std::vector<float> mask_values;
    if (stepwise_enabled) {
        mask_values.resize(static_cast<size_t>(query_length) *
                           static_cast<size_t>(key_value_length));
    }

    auto run_stepwise = [&](int repeats_per_step) -> bool {
        if (repeats_per_step == 0) {
            return true;
        }
        for (int step = 0; step < stepwise_steps; ++step) {
            const int step_past = stepwise_past_start + step;
            size_t offset = 0;
            for (int query = 0; query < query_length; ++query) {
                const int max_allowed = step_past + query;
                for (int key = 0; key < key_value_length; ++key) {
                    mask_values[offset++] = key <= max_allowed ? 0.0f : -1.0e9f;
                }
            }
            ggml_backend_tensor_set(mask, mask_values.data(), 0, ggml_nbytes(mask));
            for (int repeat = 0; repeat < repeats_per_step; ++repeat) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                    return false;
                }
            }
        }
        return true;
    };
    auto run_decode = [&](int repeats) -> bool {
        for (int i = 0; i < repeats; ++i) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }
        return true;
    };

    if (stepwise_enabled) {
        if (!run_stepwise(warmup_iters)) {
            std::cerr << "warmup compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 2;
        }
        if (!run_stepwise(bench_iters)) {
            std::cerr << "bench compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 2;
        }
    } else {
        if (!run_decode(warmup_iters)) {
            std::cerr << "warmup compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 2;
        }
        if (!run_decode(bench_iters)) {
            std::cerr << "bench compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 2;
        }
    }

    std::vector<float> output(static_cast<size_t>(ggml_nelements(output_projection)));
    ggml_backend_tensor_get(
        output_projection, output.data(), 0, ggml_nbytes(output_projection));
    double checksum = 0.0;
    for (size_t i = 0; i < std::min<size_t>(output.size(), 16); ++i) {
        checksum += static_cast<double>(output[i]);
    }

    const auto total_end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    const double denominator = stepwise_enabled
                                   ? static_cast<double>(bench_iters) *
                                         static_cast<double>(stepwise_steps)
                                   : static_cast<double>(bench_iters);
    const double avg_ms = elapsed_ms / denominator;

    std::cout << std::fixed << std::setprecision(6)
              << "mode=" << (stepwise_enabled ? "stepwise" : "decode")
              << " backend=" << backend_name << " hidden=" << hidden_features
              << " q_heads=" << query_head_count << " kv_heads=" << kv_head_count
              << " q_len=" << query_length << " kv_len=" << key_value_length
              << " warmup=" << warmup_iters << " iters=" << bench_iters
              << " steps=" << (stepwise_enabled ? stepwise_steps : 0)
              << " avg_ms=" << avg_ms << " checksum=" << checksum << "\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
