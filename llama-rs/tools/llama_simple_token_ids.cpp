#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    std::fprintf(stderr, "usage: %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello";
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
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt =
        -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to size prompt tokenization\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
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

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = true;
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
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
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
