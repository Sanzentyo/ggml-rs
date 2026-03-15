#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int GPTJ_VOCAB = 64;
constexpr int GPTJ_EMBED = 32;
constexpr size_t GPTJ_CTX_BYTES = 64ull * 1024 * 1024;

constexpr int MAGIKA_VOCAB = 257;
constexpr int MAGIKA_HIDDEN = 96;
constexpr int MAGIKA_BEG_SIZE = 512;
constexpr int MAGIKA_MID_SIZE = 512;
constexpr int MAGIKA_END_SIZE = 512;
constexpr int MAGIKA_INPUT_SIZE = MAGIKA_BEG_SIZE + MAGIKA_MID_SIZE + MAGIKA_END_SIZE;
constexpr int MAGIKA_PADDING_TOKEN = 256;
constexpr size_t MAGIKA_CTX_BYTES = 64ull * 1024 * 1024;

const std::vector<std::string> MAGIKA_LABELS = {
    "ai", "apk", "csv", "elf", "html", "java", "jpeg", "json",
    "pdf", "png", "python", "rust", "sql", "txt", "xml", "zip",
};

struct synth_rng {
    uint64_t state;

    explicit synth_rng(uint64_t seed) : state(seed ^ 0x9E3779B97F4A7C15ull) {}

    uint64_t next_u64() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    float next_f32_signed() {
        const uint32_t unit_bits = static_cast<uint32_t>(next_u64() >> 40);
        const float unit = static_cast<float>(unit_bits) / static_cast<float>((1u << 24) - 1u);
        return std::fma(unit, 2.0f, -1.0f);
    }
};

std::vector<float> synth_values(uint64_t seed, size_t len, float scale) {
    synth_rng rng(seed);
    std::vector<float> out(len);
    for (size_t i = 0; i < len; ++i) {
        out[i] = rng.next_f32_signed() * scale;
    }
    return out;
}

void write_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values, const char * name) {
    const size_t expected = static_cast<size_t>(ggml_nelements(tensor));
    if (values.size() != expected) {
        throw std::runtime_error(std::string("size mismatch for tensor ") + name);
    }
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
}

void write_tensor_i32(ggml_tensor * tensor, const std::vector<int32_t> & values, const char * name) {
    const size_t expected = static_cast<size_t>(ggml_nelements(tensor));
    if (values.size() != expected) {
        throw std::runtime_error(std::string("size mismatch for tensor ") + name);
    }
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(int32_t));
}

std::vector<float> read_tensor_f32(const ggml_tensor * tensor) {
    const size_t len = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> out(len);
    std::memcpy(out.data(), tensor->data, len * sizeof(float));
    return out;
}

template <typename T>
std::string join_values(const std::vector<T> & values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << values[i];
    }
    return oss.str();
}

std::vector<std::pair<size_t, float>> top_k(const std::vector<float> & values, size_t k) {
    std::vector<size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (values[lhs] == values[rhs]) {
            return lhs < rhs;
        }
        return values[lhs] > values[rhs];
    });

    std::vector<std::pair<size_t, float>> out;
    out.reserve(std::min(k, order.size()));
    for (size_t i = 0; i < std::min(k, order.size()); ++i) {
        out.emplace_back(order[i], values[order[i]]);
    }
    return out;
}

std::string format_topk_logits(const std::vector<std::pair<size_t, float>> & pairs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << pairs[i].first << ':' << pairs[i].second;
    }
    return oss.str();
}

double logits_checksum(const std::vector<float> & logits) {
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        sum += static_cast<double>(logits[i]) * static_cast<double>(i + 1);
    }
    return sum;
}

struct gptj_weights {
    std::vector<float> token_embedding;
    std::vector<float> proj_weight;
    std::vector<float> proj_bias;
    std::vector<float> head_weight;
    std::vector<float> head_bias;
};

gptj_weights build_gptj_weights(uint64_t seed) {
    return {
        synth_values(seed + 11, GPTJ_EMBED * GPTJ_VOCAB, 0.45f),
        synth_values(seed + 17, GPTJ_EMBED * GPTJ_EMBED, 0.35f),
        synth_values(seed + 23, GPTJ_EMBED, 0.12f),
        synth_values(seed + 29, GPTJ_VOCAB * GPTJ_EMBED, 0.30f),
        synth_values(seed + 31, GPTJ_VOCAB, 0.09f),
    };
}

std::vector<int32_t> tokenize_prompt(const std::string & prompt) {
    std::vector<int32_t> tokens;
    tokens.reserve(prompt.size());
    for (unsigned char c : prompt) {
        tokens.push_back(static_cast<int32_t>(c % GPTJ_VOCAB));
    }
    if (tokens.empty()) {
        tokens = {1, 2, 3, 4};
    }
    return tokens;
}

std::vector<float> gptj_eval_logits(const gptj_weights & weights, const std::vector<int32_t> & tokens) {
    ggml_init_params params = {
        /*.mem_size   =*/ GPTJ_CTX_BYTES,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for gptj_eval_logits");
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);

    ggml_tensor * token_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(tokens.size()));
    ggml_tensor * token_embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GPTJ_EMBED, GPTJ_VOCAB);
    ggml_tensor * proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GPTJ_EMBED, GPTJ_EMBED);
    ggml_tensor * proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GPTJ_EMBED);
    ggml_tensor * head_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GPTJ_EMBED, GPTJ_VOCAB);
    ggml_tensor * head_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GPTJ_VOCAB);

    write_tensor_i32(token_tensor, tokens, "token_tensor");
    write_tensor_f32(token_embedding, weights.token_embedding, "token_embedding");
    write_tensor_f32(proj_weight, weights.proj_weight, "proj_weight");
    write_tensor_f32(proj_bias, weights.proj_bias, "proj_bias");
    write_tensor_f32(head_weight, weights.head_weight, "head_weight");
    write_tensor_f32(head_bias, weights.head_bias, "head_bias");

    ggml_tensor * embeddings = ggml_get_rows(ctx, token_embedding, token_tensor);
    ggml_tensor * hidden_linear = ggml_mul_mat(ctx, proj_weight, embeddings);
    ggml_tensor * hidden_bias = ggml_repeat(ctx, proj_bias, hidden_linear);
    ggml_tensor * hidden = ggml_add(ctx, hidden_linear, hidden_bias);
    hidden = ggml_silu(ctx, hidden);

    ggml_tensor * logits_linear = ggml_mul_mat(ctx, head_weight, hidden);
    ggml_tensor * logits_bias = ggml_repeat(ctx, head_bias, logits_linear);
    ggml_tensor * logits = ggml_add(ctx, logits_linear, logits_bias);

    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    std::vector<float> logits_all = read_tensor_f32(logits);
    const size_t start = GPTJ_VOCAB * (tokens.size() - 1);
    std::vector<float> out(logits_all.begin() + static_cast<std::ptrdiff_t>(start),
                           logits_all.begin() + static_cast<std::ptrdiff_t>(start + GPTJ_VOCAB));
    ggml_free(ctx);
    return out;
}

struct gptj_main_options {
    uint64_t seed = 7;
    int n_predict = 8;
    std::string prompt = "ggml-rs synthetic gpt-j";
};

void run_gptj_main_mode(const gptj_main_options & opts) {
    const gptj_weights weights = build_gptj_weights(opts.seed);
    std::vector<int32_t> tokens = tokenize_prompt(opts.prompt);
    const size_t prompt_len = tokens.size();
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(opts.n_predict));
    std::vector<float> final_logits(GPTJ_VOCAB, 0.0f);

    const auto started = std::chrono::steady_clock::now();
    for (int step = 0; step < opts.n_predict; ++step) {
        final_logits = gptj_eval_logits(weights, tokens);
        const auto max_it = std::max_element(final_logits.begin(), final_logits.end());
        const int32_t next_token = static_cast<int32_t>(std::distance(final_logits.begin(), max_it));
        tokens.push_back(next_token);
        generated.push_back(next_token);
    }
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::cout << "mode=gptj-main-synth\n";
    std::cout << "seed=" << opts.seed << "\n";
    std::cout << "prompt_len=" << prompt_len << "\n";
    std::cout << "n_predict=" << opts.n_predict << "\n";
    std::cout << "generated_tokens=" << join_values(generated) << "\n";
    std::cout << "logits_top5=" << format_topk_logits(top_k(final_logits, 5)) << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "logit_checksum=" << logits_checksum(final_logits) << "\n";
    std::cout << "elapsed_us=" << elapsed_us << "\n";
}

struct gptj_quantize_options {
    uint64_t seed = 7;
    size_t tensor_len = 4096;
};

void run_gptj_quantize_mode(const gptj_quantize_options & opts) {
    const auto started = std::chrono::steady_clock::now();
    const std::vector<float> source = synth_values(opts.seed + 43, opts.tensor_len, 1.75f);

    float max_abs = 0.0f;
    for (float value : source) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;

    std::vector<int8_t> quantized(source.size());
    std::vector<float> dequantized(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const float q = std::round(source[i] / scale);
        const float clamped = std::max(-127.0f, std::min(127.0f, q));
        quantized[i] = static_cast<int8_t>(clamped);
        dequantized[i] = static_cast<float>(quantized[i]) * scale;
    }

    int64_t checksum = 0;
    double mse = 0.0;
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < source.size(); ++i) {
        checksum += static_cast<int64_t>(i + 1) * static_cast<int64_t>(quantized[i]);
        const float diff = source[i] - dequantized[i];
        mse += static_cast<double>(diff) * static_cast<double>(diff);
        max_abs_err = std::max(max_abs_err, std::fabs(diff));
    }
    mse /= static_cast<double>(source.size());

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::ostringstream head;
    for (size_t i = 0; i < std::min<size_t>(8, quantized.size()); ++i) {
        if (i > 0) {
            head << ',';
        }
        head << static_cast<int>(quantized[i]);
    }

    std::cout << "mode=gptj-quantize-synth\n";
    std::cout << "seed=" << opts.seed << "\n";
    std::cout << "tensor_len=" << opts.tensor_len << "\n";
    std::cout << std::fixed << std::setprecision(9) << "scale=" << scale << "\n";
    std::cout << "quantized_head=" << head.str() << "\n";
    std::cout << "quantized_checksum=" << checksum << "\n";
    std::cout << std::fixed << std::setprecision(12) << "mse=" << mse << "\n";
    std::cout << std::fixed << std::setprecision(9) << "max_abs_err=" << max_abs_err << "\n";
    std::cout << "elapsed_us=" << elapsed_us << "\n";
}

std::vector<uint8_t> synthetic_file_bytes(uint64_t seed, size_t sample_index) {
    synth_rng rng(seed + static_cast<uint64_t>(sample_index) * 97ull);
    const size_t len = 768 + sample_index * 257;
    std::vector<uint8_t> out(len);
    for (size_t i = 0; i < len; ++i) {
        const uint64_t mixed = rng.next_u64() ^ (static_cast<uint64_t>(i) * 0xA0761D6478BD642Full);
        out[i] = static_cast<uint8_t>(mixed & 0xFF);
    }
    return out;
}

std::vector<uint8_t> read_file_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<int> sampled_tokens(const std::vector<uint8_t> & bytes) {
    std::vector<int> sampled(MAGIKA_INPUT_SIZE, MAGIKA_PADDING_TOKEN);

    const int n_beg = std::min<int>(MAGIKA_BEG_SIZE, static_cast<int>(bytes.size()));
    for (int i = 0; i < n_beg; ++i) {
        sampled[i] = bytes[static_cast<size_t>(i)];
    }

    const int mid_offs = std::max(0, (static_cast<int>(bytes.size()) - MAGIKA_MID_SIZE) / 2);
    const int mid_end = std::min<int>(mid_offs + MAGIKA_MID_SIZE, static_cast<int>(bytes.size()));
    const int mid_len = std::max(0, mid_end - mid_offs);
    const int mid_start = MAGIKA_BEG_SIZE + (MAGIKA_MID_SIZE / 2) - mid_len / 2;
    for (int i = 0; i < mid_len; ++i) {
        sampled[mid_start + i] = bytes[static_cast<size_t>(mid_offs + i)];
    }

    const int end_offs = std::max(0, static_cast<int>(bytes.size()) - MAGIKA_END_SIZE);
    const int end_len = static_cast<int>(bytes.size()) - end_offs;
    const int end_start = MAGIKA_BEG_SIZE + MAGIKA_MID_SIZE + MAGIKA_END_SIZE - end_len;
    for (int i = 0; i < end_len; ++i) {
        sampled[end_start + i] = bytes[static_cast<size_t>(end_offs + i)];
    }

    return sampled;
}

std::vector<float> histogram_features(const std::vector<int> & tokens) {
    std::vector<float> out(MAGIKA_VOCAB, 0.0f);
    for (int token : tokens) {
        out[static_cast<size_t>(token)] += 1.0f;
    }
    const float normalizer = static_cast<float>(tokens.size());
    for (float & value : out) {
        value /= normalizer;
    }
    return out;
}

std::string format_label_summary(const std::vector<float> & probs) {
    const auto best = top_k(probs, 3);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < best.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << MAGIKA_LABELS[best[i].first] << '@' << best[i].second;
    }
    return oss.str();
}

double probability_checksum(const std::vector<float> & probs, size_t sample_count) {
    double checksum = 0.0;
    const size_t label_count = MAGIKA_LABELS.size();
    for (size_t sample = 0; sample < sample_count; ++sample) {
        for (size_t label = 0; label < label_count; ++label) {
            const double value = probs[sample * label_count + label];
            checksum += value * static_cast<double>(sample + 1) * static_cast<double>(label + 1);
        }
    }
    return checksum;
}

struct magika_options {
    uint64_t seed = 7;
    size_t samples = 3;
    std::vector<std::string> files;
};

void run_magika_mode(const magika_options & opts) {
    std::vector<std::vector<uint8_t>> sources;
    if (opts.files.empty()) {
        sources.reserve(opts.samples);
        for (size_t i = 0; i < opts.samples; ++i) {
            sources.push_back(synthetic_file_bytes(opts.seed, i));
        }
    } else {
        sources.reserve(opts.files.size());
        for (const auto & path : opts.files) {
            sources.push_back(read_file_bytes(path));
        }
    }

    const size_t sample_count = sources.size();
    std::vector<float> feature_matrix;
    feature_matrix.reserve(sample_count * MAGIKA_VOCAB);
    for (const auto & source : sources) {
        const auto sampled = sampled_tokens(source);
        const auto features = histogram_features(sampled);
        feature_matrix.insert(feature_matrix.end(), features.begin(), features.end());
    }

    const std::vector<float> dense_w = synth_values(opts.seed + 101, MAGIKA_HIDDEN * MAGIKA_VOCAB, 0.42f);
    const std::vector<float> dense_b = synth_values(opts.seed + 103, MAGIKA_HIDDEN, 0.08f);
    const std::vector<float> head_w = synth_values(opts.seed + 107, MAGIKA_LABELS.size() * MAGIKA_HIDDEN, 0.28f);
    const std::vector<float> head_b = synth_values(opts.seed + 109, MAGIKA_LABELS.size(), 0.07f);

    const auto started = std::chrono::steady_clock::now();

    ggml_init_params params = {
        /*.mem_size   =*/ MAGIKA_CTX_BYTES,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for magika mode");
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MAGIKA_VOCAB, sample_count);
    ggml_tensor * dense_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MAGIKA_VOCAB, MAGIKA_HIDDEN);
    ggml_tensor * dense_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MAGIKA_HIDDEN);
    ggml_tensor * head_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MAGIKA_HIDDEN, MAGIKA_LABELS.size());
    ggml_tensor * head_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MAGIKA_LABELS.size());

    write_tensor_f32(input, feature_matrix, "input");
    write_tensor_f32(dense_weight, dense_w, "dense_weight");
    write_tensor_f32(dense_bias, dense_b, "dense_bias");
    write_tensor_f32(head_weight, head_w, "head_weight");
    write_tensor_f32(head_bias, head_b, "head_bias");

    ggml_tensor * hidden_linear = ggml_mul_mat(ctx, dense_weight, input);
    ggml_tensor * hidden_bias = ggml_repeat(ctx, dense_bias, hidden_linear);
    ggml_tensor * hidden = ggml_add(ctx, hidden_linear, hidden_bias);
    hidden = ggml_silu(ctx, hidden);

    ggml_tensor * logits_linear = ggml_mul_mat(ctx, head_weight, hidden);
    ggml_tensor * logits_bias = ggml_repeat(ctx, head_bias, logits_linear);
    ggml_tensor * logits = ggml_add(ctx, logits_linear, logits_bias);
    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    ggml_build_forward_expand(graph, probs);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    std::vector<float> prob_values = read_tensor_f32(probs);
    ggml_free(ctx);

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::ostringstream labels_top;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        if (sample > 0) {
            labels_top << ';';
        }
        const size_t begin = sample * MAGIKA_LABELS.size();
        const size_t end = begin + MAGIKA_LABELS.size();
        const std::vector<float> row(prob_values.begin() + static_cast<std::ptrdiff_t>(begin),
                                     prob_values.begin() + static_cast<std::ptrdiff_t>(end));
        labels_top << "sample" << sample << ":" << format_label_summary(row);
    }

    std::cout << "mode=magika-main-synth\n";
    std::cout << "seed=" << opts.seed << "\n";
    std::cout << "samples=" << sample_count << "\n";
    std::cout << "labels_top=" << labels_top.str() << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "prob_checksum=" << probability_checksum(prob_values, sample_count) << "\n";
    std::cout << "elapsed_us=" << elapsed_us << "\n";
}

void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " <gptj-main|gptj-quantize|magika-main> [--synthetic] [options] [files...]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }

        const std::string mode = argv[1];
        gptj_main_options gptj_main;
        gptj_quantize_options gptj_quantize;
        magika_options magika;

        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--seed") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--seed requires a value");
                }
                const auto parsed = static_cast<uint64_t>(std::stoull(argv[++i]));
                gptj_main.seed = parsed;
                gptj_quantize.seed = parsed;
                magika.seed = parsed;
            } else if (arg == "--n-predict") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--n-predict requires a value");
                }
                gptj_main.n_predict = std::stoi(argv[++i]);
            } else if (arg == "--prompt") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--prompt requires a value");
                }
                gptj_main.prompt = argv[++i];
            } else if (arg == "--tensor-len") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--tensor-len requires a value");
                }
                gptj_quantize.tensor_len = static_cast<size_t>(std::stoull(argv[++i]));
            } else if (arg == "--samples") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--samples requires a value");
                }
                magika.samples = static_cast<size_t>(std::stoull(argv[++i]));
            } else if (arg == "--synthetic") {
                continue;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("unknown option: " + arg);
            } else {
                magika.files.push_back(arg);
            }
        }

        if (mode == "gptj-main") {
            run_gptj_main_mode(gptj_main);
            return 0;
        }
        if (mode == "gptj-quantize") {
            run_gptj_quantize_mode(gptj_quantize);
            return 0;
        }
        if (mode == "magika-main") {
            run_magika_mode(magika);
            return 0;
        }

        throw std::runtime_error("unknown mode: " + mode);
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
