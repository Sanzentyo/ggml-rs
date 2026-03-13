#include "ggml-backend.h"
#include "ggml.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static std::vector<float> make_input(int hidden) {
    std::vector<float> input(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) {
        input[static_cast<size_t>(i)] = static_cast<float>((i + 5) % 19) * 0.125f;
    }
    return input;
}

static std::vector<float> make_gate_weights(int hidden, int ffn) {
    std::vector<float> out(static_cast<size_t>(hidden) * static_cast<size_t>(ffn));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(i % 17) * 0.03125f;
    }
    return out;
}

static std::vector<float> make_up_weights(int hidden, int ffn) {
    std::vector<float> out(static_cast<size_t>(hidden) * static_cast<size_t>(ffn));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i + 11) % 23) * 0.015625f;
    }
    return out;
}

static std::vector<float> make_down_weights(int hidden, int ffn) {
    std::vector<float> out(static_cast<size_t>(hidden) * static_cast<size_t>(ffn));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i + 7) % 29) * 0.0078125f;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: mlp_reference <hidden_features> <ffn_features>\n";
        return 2;
    }

    const int hidden = std::atoi(argv[1]);
    const int ffn = std::atoi(argv[2]);
    if (hidden <= 0 || ffn <= 0) {
        std::cerr << "hidden/ffn must be positive\n";
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_t backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::cerr << "ggml_backend_cpu_init failed\n";
        return 2;
    }

    const size_t ctx_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                            ggml_graph_overhead() + 1024 * 1024;
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

    ggml_tensor* w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn);
    ggml_tensor* w_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn);
    ggml_tensor* w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn, hidden);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);

    ggml_tensor* gate = ggml_mul_mat(ctx, w_gate, x);
    ggml_tensor* up = ggml_mul_mat(ctx, w_up, x);
    ggml_tensor* activated = ggml_silu(ctx, gate);
    ggml_tensor* fused = ggml_mul(ctx, activated, up);
    ggml_tensor* y = ggml_mul_mat(ctx, w_down, fused);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::cerr << "ggml_backend_alloc_ctx_tensors failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    const auto gate_w = make_gate_weights(hidden, ffn);
    const auto up_w = make_up_weights(hidden, ffn);
    const auto down_w = make_down_weights(hidden, ffn);
    const auto input = make_input(hidden);

    ggml_backend_tensor_set(w_gate, gate_w.data(), 0, ggml_nbytes(w_gate));
    ggml_backend_tensor_set(w_up, up_w.data(), 0, ggml_nbytes(w_up));
    ggml_backend_tensor_set(w_down, down_w.data(), 0, ggml_nbytes(w_down));
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "ggml_backend_graph_compute failed with status " << static_cast<int>(status)
                  << "\n";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    std::vector<float> output(static_cast<size_t>(ggml_nelements(y)));
    ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));

    std::cout << std::setprecision(9);
    for (size_t i = 0; i < output.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << output[i];
    }
    std::cout << "\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
