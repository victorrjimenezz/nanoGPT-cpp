#pragma once

#include <torch/torch.h>
#include <vector>

struct GPTConfig {
    int block_size = 256;
    int vocab_size = 65;
    int n_layer = 6;
    int n_head = 6;
    int n_embd = 384;
    double dropout = 0.2;
    bool bias = false;
};

struct LayerNormImpl final : torch::nn::Module {
    LayerNormImpl(int ndim, bool bias);

    torch::Tensor forward(const torch::Tensor &input) const;

    torch::Tensor weight;
    torch::Tensor bias;
};

TORCH_MODULE(LayerNorm);

struct CausalSelfAttentionImpl final : torch::nn::Module {
    explicit CausalSelfAttentionImpl(const GPTConfig &config);

    torch::Tensor forward(const torch::Tensor &x);

    torch::nn::Linear c_attn{nullptr}, c_proj{nullptr};
    torch::nn::Dropout attn_dropout{nullptr}, resid_dropout{nullptr};
    int n_head, n_embd;
    double dropout;
    bool flash;
    torch::Tensor bias;
};

TORCH_MODULE(CausalSelfAttention);

struct MLPImpl final : torch::nn::Module {
    explicit MLPImpl(const GPTConfig &config);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear c_fc{nullptr}, c_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

TORCH_MODULE(MLP);

struct BlockImpl final : torch::nn::Module {
    explicit BlockImpl(const GPTConfig &config);

    torch::Tensor forward(torch::Tensor x);

    LayerNorm ln_1{nullptr}, ln_2{nullptr};
    CausalSelfAttention attn{nullptr};
    MLP mlp{nullptr};
};

TORCH_MODULE(Block);

struct GPTImpl final : torch::nn::Module {
    explicit GPTImpl(const GPTConfig &config);

    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor &idx,
                                                    const torch::Tensor &targets = torch::Tensor());

    void _init_weights(torch::nn::Module &module) const;

    torch::Tensor generate(torch::Tensor idx, int max_new_tokens, double temperature = 1.0, int top_k = -1);

    GPTConfig config;
    torch::nn::Embedding wte{nullptr}, wpe{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::ModuleList h{nullptr};
    LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};
};

TORCH_MODULE(GPT);
