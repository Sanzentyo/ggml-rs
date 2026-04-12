#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    std::fprintf(
        stderr,
        "usage: %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [--prompt-tokens 1,2,3] [prompt]\n",
        argv[0]);
}

static bool parse_prompt_tokens_csv(const char * text, std::vector<llama_token> & out_tokens) {
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        try {
            out_tokens.push_back(static_cast<llama_token>(std::stoi(item)));
        } catch (...) {
            return false;
        }
    }
    return !out_tokens.empty();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello";
    std::vector<llama_token> prompt_tokens;
    bool use_prompt_tokens = false;
    int n_gpu_layers = 99;
    int n_predict = 1;

    int i = 1;
    for (; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0) {
            if (i + 1 >= argc) {
                print_usage(argc, argv);
                return 1;
            }
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc) {
                print_usage(argc, argv);
                return 1;
            }
            try {
                n_predict = std::stoi(argv[++i]);
            } catch (...) {
                print_usage(argc, argv);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-ngl") == 0) {
            if (i + 1 >= argc) {
                print_usage(argc, argv);
                return 1;
            }
            try {
                n_gpu_layers = std::stoi(argv[++i]);
            } catch (...) {
                print_usage(argc, argv);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--prompt-tokens") == 0) {
            if (i + 1 >= argc || !parse_prompt_tokens_csv(argv[++i], prompt_tokens)) {
                print_usage(argc, argv);
                return 1;
            }
            use_prompt_tokens = true;
        } else {
            break;
        }
    }

    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    if (i < argc) {
        prompt = argv[i++];
        for (; i < argc; ++i) {
            prompt += " ";
            prompt += argv[i];
        }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    ggml_backend_dev_t cpu_only_devices[2] = { nullptr, nullptr };
    if (n_gpu_layers == 0) {
        cpu_only_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_only_devices[0] == nullptr) {
            std::fprintf(stderr, "failed to resolve CPU backend device\n");
            return 1;
        }
        model_params.devices = cpu_only_devices;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!use_prompt_tokens) {
        const int n_prompt =
            -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            std::fprintf(stderr, "failed to size prompt tokenization\n");
            llama_model_free(model);
            return 1;
        }

        prompt_tokens.resize(static_cast<size_t>(n_prompt));
        if (llama_tokenize(
                vocab,
                prompt.c_str(),
                prompt.size(),
                prompt_tokens.data(),
                prompt_tokens.size(),
                true,
                true) < 0) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            llama_model_free(model);
            return 1;
        }
    }

    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "prompt token list must not be empty\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(prompt_tokens.size() + n_predict - 1);
    ctx_params.n_batch = static_cast<uint32_t>(prompt_tokens.size());
    ctx_params.no_perf = true;
    if (n_gpu_layers == 0) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.offload_kqv = false;
        ctx_params.op_offload = false;
    }
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch) != 0) {
            std::fprintf(stderr, "failed to encode prompt\n");
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    std::printf("prompt_token_ids=[");
    for (size_t idx = 0; idx < prompt_tokens.size(); ++idx) {
        if (idx > 0) {
            std::printf(",");
        }
        std::printf("%d", static_cast<int>(prompt_tokens[idx]));
    }
    std::printf("]\n");

    std::vector<int32_t> generated_token_ids;
    for (int n_pos = 0; n_pos + batch.n_tokens < static_cast<int>(prompt_tokens.size()) + n_predict;) {
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "failed to decode\n");
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        n_pos += batch.n_tokens;
        llama_token next_token_id = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, next_token_id)) {
            break;
        }

        generated_token_ids.push_back(next_token_id);
        batch = llama_batch_get_one(&next_token_id, 1);
    }

    std::printf("generated_token_ids=[");
    for (size_t idx = 0; idx < generated_token_ids.size(); ++idx) {
        if (idx > 0) {
            std::printf(",");
        }
        std::printf("%d", generated_token_ids[idx]);
    }
    std::printf("]\n");

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
