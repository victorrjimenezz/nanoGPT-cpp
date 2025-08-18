#include "model.h"
#include <cmath>
#include <algorithm>

LayerNormImpl::LayerNormImpl(const int ndim, bool const bias_enabled) {
    weight = register_parameter("weight", torch::ones(ndim));
    if (bias_enabled) {
        bias = register_parameter("bias", torch::zeros(ndim));
    }
}

torch::Tensor LayerNormImpl::forward(const torch::Tensor &input) const {
    return torch::nn::functional::layer_norm(
        input, torch::nn::functional::LayerNormFuncOptions({input.size(-1)}).weight(weight).bias(bias).eps(1e-5));
}

CausalSelfAttentionImpl::CausalSelfAttentionImpl(const GPTConfig &config)
    : c_attn(torch::nn::LinearOptions(config.n_embd, 3 * config.n_embd).bias(config.bias)),
      c_proj(torch::nn::LinearOptions(config.n_embd, config.n_embd).bias(config.bias)),
      attn_dropout(torch::nn::DropoutOptions(config.dropout)),
      resid_dropout(torch::nn::DropoutOptions(config.dropout)),
      n_head(config.n_head),
      n_embd(config.n_embd),
      dropout(config.dropout) {
    register_module("c_attn", c_attn);
    register_module("c_proj", c_proj);
    register_module("attn_dropout", attn_dropout);
    register_module("resid_dropout", resid_dropout);

    flash = false;

    if (!flash) {
        std::cout << "WARNING: Using slow attention implementation." << std::endl;
        this->bias = torch::tril(torch::ones({config.block_size, config.block_size}))
                .view({1, 1, config.block_size, config.block_size});
        register_buffer("bias", this->bias);
    }
}

torch::Tensor CausalSelfAttentionImpl::forward(const torch::Tensor &x) {
    auto B = x.size(0);
    auto T = x.size(1);
    auto C = x.size(2);

    const auto qkv = c_attn->forward(x);
    const auto qkv_s = qkv.split(n_embd, 2);
    auto q = qkv_s[0];
    auto k = qkv_s[1];
    auto v = qkv_s[2];

    q = q.view({B, T, n_head, C / n_head}).transpose(1, 2);
    k = k.view({B, T, n_head, C / n_head}).transpose(1, 2);
    v = v.view({B, T, n_head, C / n_head}).transpose(1, 2);

    torch::Tensor y;
    if (flash) {
        // Disabled for compatibility
    } else {
        auto att = (q.matmul(k.transpose(-2, -1))) * (1.0 / std::sqrt(k.size(-1)));
        att = att.masked_fill(bias.slice(2, 0, T).slice(3, 0, T) == 0, -std::numeric_limits<float>::infinity());
        att = torch::nn::functional::softmax(att, -1);
        att = attn_dropout->forward(att);
        y = att.matmul(v);
    }

    y = y.transpose(1, 2).contiguous().view({B, T, C});
    y = resid_dropout->forward(c_proj->forward(y));
    return y;
}

MLPImpl::MLPImpl(const GPTConfig &config)
    : c_fc(torch::nn::LinearOptions(config.n_embd, 4 * config.n_embd).bias(config.bias)),
      c_proj(torch::nn::LinearOptions(4 * config.n_embd, config.n_embd).bias(config.bias)),
      dropout(torch::nn::DropoutOptions(config.dropout)) {
    register_module("c_fc", c_fc);
    register_module("c_proj", c_proj);
    register_module("dropout", dropout);
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    x = c_fc->forward(x);
    // VERIFIED FIX: Construct the options object and set the approximate property using the setter method.
    x = torch::nn::functional::gelu(x, torch::nn::functional::GELUFuncOptions().approximate("tanh"));
    x = c_proj->forward(x);
    x = dropout->forward(x);
    return x;
}

BlockImpl::BlockImpl(const GPTConfig &config)
    : ln_1(config.n_embd, config.bias),
      ln_2(config.n_embd, config.bias),
      attn(config),
      mlp(config) {
    register_module("ln_1", ln_1);
    register_module("attn", attn);
    register_module("ln_2", ln_2);
    register_module("mlp", mlp);
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    x = x + attn->forward(ln_1->forward(x));
    x = x + mlp->forward(ln_2->forward(x));
    return x;
}

GPTImpl::GPTImpl(const GPTConfig &config)
    : config(config),
      wte(torch::nn::EmbeddingOptions(config.vocab_size, config.n_embd)),
      wpe(torch::nn::EmbeddingOptions(config.block_size, config.n_embd)),
      drop(torch::nn::DropoutOptions(config.dropout)),
      h(torch::nn::ModuleList()),
      ln_f(config.n_embd, config.bias),
      lm_head(torch::nn::LinearOptions(config.n_embd, config.vocab_size).bias(false)) {
    register_module("wte", wte);
    register_module("wpe", wpe);
    register_module("drop", drop);
    for (int i = 0; i < config.n_layer; ++i) {
        h->push_back(Block(config));
    }
    register_module("h", h);
    register_module("ln_f", ln_f);
    register_module("lm_head", lm_head);

    // Weight tying
    lm_head->weight = wte->weight;

    // Weight initialization
    this->apply([this](torch::nn::Module &module) { this->_init_weights(module); });
}

void GPTImpl::_init_weights(torch::nn::Module &module) const {
    if (const auto *linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::normal_(linear->weight, 0.0, 0.02);
        if (linear->bias.defined()) {
            torch::nn::init::zeros_(linear->bias);
        }
    } else if (const auto *embedding = module.as<torch::nn::Embedding>()) {
        torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
    }

    for (auto &p: this->named_parameters()) {
        if (p.key().find("c_proj.weight") != std::string::npos) {
            torch::nn::init::normal_(p.value(), 0.0, 0.02 / std::sqrt(2.0 * config.n_layer));
        }
    }
}

std::pair<torch::Tensor, torch::Tensor> GPTImpl::forward(const torch::Tensor &idx, const torch::Tensor &targets) {
    auto device = idx.device();
    const auto t = idx.size(1);

    TORCH_CHECK(t <= config.block_size, "Cannot forward sequence of length ", t, ", block size is only ",
                config.block_size);

    const auto pos = torch::arange(0, t, torch::TensorOptions().dtype(torch::kLong).device(device));
    const auto tok_emb = wte->forward(idx);
    const auto pos_emb = wpe->forward(pos);
    auto x = drop->forward(tok_emb + pos_emb);

    for (const auto &block_module: *h) {
        x = block_module->as<BlockImpl>()->forward(x);
    }
    x = ln_f->forward(x);

    torch::Tensor logits, loss;
    if (targets.defined()) {
        logits = lm_head->forward(x);
        loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(-1)}), targets.view(-1),
                                                    torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-1));
    } else {
        // inference-time mini-optimization
        logits = lm_head->forward(
            x.index({torch::indexing::Slice(), torch::indexing::Slice(-1, torch::indexing::None)}));
        loss = torch::Tensor();
    }
    return {logits, loss};
}

torch::Tensor GPTImpl::generate(torch::Tensor idx, const int max_new_tokens, double const temperature,
                                int const top_k) {
    for (int i = 0; i < max_new_tokens; ++i) {
        auto idx_cond = idx.size(1) <= config.block_size ? idx : idx.slice(1, idx.size(1) - config.block_size);
        auto outputs = this->forward(idx_cond);
        auto logits = std::get<0>(outputs);
        logits = logits.slice(1, -1).squeeze(1) / temperature;

        if (top_k > 0) {
            auto topk_res = torch::topk(logits, std::min(static_cast<int64_t>(top_k), logits.size(-1)));
            auto topk_vals = std::get<0>(topk_res);
            logits.masked_fill_(logits < topk_vals.slice(1, -1), -std::numeric_limits<float>::infinity());
        }

        auto probs = torch::nn::functional::softmax(logits, -1);
        auto idx_next = torch::multinomial(probs, 1);
        idx = torch::cat({idx, idx_next}, 1);
    }
    return idx;
}
