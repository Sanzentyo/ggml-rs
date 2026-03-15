#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

float synth_value(uint64_t seed, size_t step, size_t index, uint64_t salt) {
    const uint64_t mixed = seed + salt + static_cast<uint64_t>(step) * 131ULL +
                           static_cast<uint64_t>(index) * 17ULL;
    const float raw = static_cast<float>(mixed % 251ULL);
    return (raw - 125.0f) * 0.01f;
}

std::vector<float> make_rhs(size_t n_embd, size_t n_vocab, uint64_t seed) {
    std::vector<float> rhs(n_embd * n_vocab);
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = synth_value(seed, 0, i, 17ULL);
    }
    return rhs;
}

std::vector<float> make_lhs(size_t n_embd, size_t active_batch, uint64_t seed,
                            size_t step) {
    std::vector<float> lhs(n_embd * active_batch);
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = synth_value(seed, step, i, 53ULL);
    }
    return lhs;
}

double step_checksum(const std::vector<float>& rhs, size_t n_embd, size_t n_vocab,
                     size_t active_batch, uint64_t seed, size_t step) {
    const auto lhs = make_lhs(n_embd, active_batch, seed, step);
    std::vector<float> logits(n_vocab * active_batch, 0.0f);

    for (size_t batch = 0; batch < active_batch; ++batch) {
        for (size_t vocab = 0; vocab < n_vocab; ++vocab) {
            float acc = 0.0f;
            const size_t rhs_base = vocab * n_embd;
            const size_t lhs_base = batch * n_embd;
            for (size_t embd = 0; embd < n_embd; ++embd) {
                acc += rhs[rhs_base + embd] * lhs[lhs_base + embd];
            }
            logits[batch * n_vocab + vocab] = acc;
        }
    }

    const size_t sample_len = std::min<size_t>(32, logits.size());
    double checksum = 0.0;
    for (size_t i = 0; i < sample_len; ++i) {
        checksum += static_cast<double>(logits[i]);
    }
    return checksum;
}

void run_quantize(size_t n_embd, size_t n_vocab, uint64_t seed) {
    const auto values = make_rhs(n_embd, n_vocab, seed);
    const auto start = std::chrono::steady_clock::now();

    float max_abs = 0.0f;
    for (float value : values) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;

    std::vector<int8_t> quantized(values.size(), 0);
    double squared_error_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const float scaled = std::round(values[i] / scale);
        const float clamped = std::clamp(scaled, -127.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(clamped);
        const float restored = static_cast<float>(quantized[i]) * scale;
        const double delta = static_cast<double>(values[i] - restored);
        squared_error_sum += delta * delta;
    }

    const double rmse = std::sqrt(squared_error_sum / static_cast<double>(values.size()));
    double checksum = static_cast<double>(scale);
    const size_t sample_len = std::min<size_t>(64, quantized.size());
    for (size_t i = 0; i < sample_len; ++i) {
        checksum += static_cast<double>(quantized[i]);
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double quantize_ms =
        std::chrono::duration<double, std::milli>(elapsed).count();
    const size_t input_bytes = values.size() * sizeof(float);
    const size_t output_bytes = 8 + 4 + quantized.size();

    std::cout << std::setprecision(9)
              << "mode=quantize"
              << " n_embd=" << n_embd
              << " n_vocab=" << n_vocab
              << " input_bytes=" << input_bytes
              << " output_bytes=" << output_bytes
              << " quantize_ms=" << quantize_ms
              << " rmse=" << rmse
              << " checksum=" << checksum << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: gpt2_synthetic_reference <mode> <n_embd> <n_vocab> <n_batch> <n_predict> <seed> [n_parallel] [n_backends]\n";
        return 2;
    }

    const std::string mode = argv[1];
    const size_t n_embd = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    const size_t n_vocab = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));
    const size_t n_batch = static_cast<size_t>(std::strtoull(argv[4], nullptr, 10));
    const size_t n_predict = static_cast<size_t>(std::strtoull(argv[5], nullptr, 10));
    const uint64_t seed = static_cast<uint64_t>(std::strtoull(argv[6], nullptr, 10));

    if (mode == "quantize") {
        if (n_embd == 0 || n_vocab == 0) {
            std::cerr << "n_embd and n_vocab must be positive\n";
            return 2;
        }
        run_quantize(n_embd, n_vocab, seed);
        return 0;
    }

    if (n_embd == 0 || n_vocab == 0 || n_batch == 0 || n_predict == 0) {
        std::cerr << "n_embd, n_vocab, n_batch, and n_predict must be positive\n";
        return 2;
    }

    const size_t n_parallel =
        argc >= 8 ? static_cast<size_t>(std::strtoull(argv[7], nullptr, 10)) : n_batch;
    const size_t n_backends =
        argc >= 9 ? static_cast<size_t>(std::strtoull(argv[8], nullptr, 10)) : 1;

    const size_t active_batch = mode == "batched" ? n_parallel : n_batch;
    if (active_batch == 0 || n_backends == 0) {
        std::cerr << "active batch and backend count must be positive\n";
        return 2;
    }

    const auto rhs = make_rhs(n_embd, n_vocab, seed);
    const auto start = std::chrono::steady_clock::now();

    double checksum = 0.0;
    if (mode == "sched") {
        for (size_t backend_idx = 0; backend_idx < n_backends; ++backend_idx) {
            for (size_t step = backend_idx; step < n_predict; step += n_backends) {
                checksum += step_checksum(rhs, n_embd, n_vocab, active_batch, seed, step);
            }
        }
    } else {
        for (size_t step = 0; step < n_predict; ++step) {
            checksum += step_checksum(rhs, n_embd, n_vocab, active_batch, seed, step);
        }
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double total_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    const double avg_step_ms = total_ms / static_cast<double>(n_predict);
    const double avg_item_ms =
        total_ms / static_cast<double>(n_predict * active_batch);
    const size_t compute_buffer_bytes =
        n_embd * (n_vocab + active_batch) * sizeof(float);

    std::cout << std::setprecision(9)
              << "mode=" << mode
              << " backend=reference"
              << " n_embd=" << n_embd
              << " n_vocab=" << n_vocab
              << " n_batch=" << active_batch
              << " n_predict=" << n_predict
              << " compute_buffer_bytes=" << compute_buffer_bytes
              << " total_ms=" << total_ms
              << " avg_step_ms=" << avg_step_ms
              << " avg_item_ms=" << avg_item_ms
              << " checksum=" << checksum << "\n";

    return 0;
}
