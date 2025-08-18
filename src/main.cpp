#include "model.h"
#include "dataloader.h"
#include "cxxopts.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

double get_lr(const int iter, const int warmup_iters, const int lr_decay_iters, double const learning_rate,
              double const min_lr) {
    if (iter < warmup_iters) {
        return learning_rate * (iter + 1.0) / (warmup_iters + 1.0);
    }
    if (iter > lr_decay_iters) {
        return min_lr;
    }
    double const decay_ratio = static_cast<double>(iter - warmup_iters) / static_cast<double>(
                                   lr_decay_iters - warmup_iters);
    double const coeff = 0.5 * (1.0 + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (learning_rate - min_lr);
}

torch::Tensor estimate_loss(GPT &model, DataLoader &loader, torch::Device device, int eval_iters) {
    model->eval();
    torch::NoGradGuard no_grad;
    auto losses = torch::zeros({eval_iters});
    for (int k = 0; k < eval_iters; ++k) {
        auto batch = loader.get_batch("val");
        auto X = batch.first.to(device);
        auto Y = batch.second.to(device);
        auto [logits, loss] = model->forward(X, Y);
        losses[k] = loss.item<float>();
    }
    model->train();
    return losses.mean();
}


int main(int argc, char **argv) {
    cxxopts::Options options("nanogpt_cpp", "A C++ implementation of nanoGPT");

    options.add_options()
            ("mode", "Mode to run: 'train' or 'sample'", cxxopts::value<std::string>()->default_value("train"))
            // Training options
            ("device", "Device to use: 'cpu', 'cuda', 'mps'", cxxopts::value<std::string>()->default_value("cpu"))
            ("batch_size", "Batch size", cxxopts::value<int>()->default_value("12"))
            ("block_size", "Context block size", cxxopts::value<int>()->default_value("64"))
            ("max_iters", "Max training iterations", cxxopts::value<int>()->default_value("2000"))
            ("lr_decay_iters", "Learning rate decay iterations", cxxopts::value<int>()->default_value("2000"))
            ("eval_interval", "Evaluation interval", cxxopts::value<int>()->default_value("250"))
            ("eval_iters", "Evaluation iterations", cxxopts::value<int>()->default_value("20"))
            ("learning_rate", "Learning rate", cxxopts::value<double>()->default_value("1e-3"))
            ("warmup_iters", "Warmup iterations", cxxopts::value<int>()->default_value("100"))
            ("min_lr", "Minimum learning rate", cxxopts::value<double>()->default_value("1e-4"))
            ("dropout", "Dropout rate", cxxopts::value<double>()->default_value("0.0"))
            ("n_layer", "Number of layers", cxxopts::value<int>()->default_value("4"))
            ("n_head", "Number of heads", cxxopts::value<int>()->default_value("4"))
            ("n_embd", "Embedding dimension", cxxopts::value<int>()->default_value("128"))
            ("out_dir", "Output directory for checkpoints",
             cxxopts::value<std::string>()->default_value("out-shakespeare-char-cpp"))
            // Sampling options
            ("start_prompt", "Starting prompt for sampling", cxxopts::value<std::string>()->default_value("\n"))
            ("num_samples", "Number of samples to generate", cxxopts::value<int>()->default_value("5"))
            ("max_new_tokens", "Max new tokens to generate", cxxopts::value<int>()->default_value("100"))
            ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // --- Configuration from parsed arguments ---
    auto mode = result["mode"].as<std::string>();
    auto device_str = result["device"].as<std::string>();
    int batch_size = result["batch_size"].as<int>();
    int block_size = result["block_size"].as<int>();
    int max_iters = result["max_iters"].as<int>();
    int lr_decay_iters = result["lr_decay_iters"].as<int>();
    int eval_interval = result["eval_interval"].as<int>();
    int eval_iters = result["eval_iters"].as<int>();
    double learning_rate = result["learning_rate"].as<double>();
    int warmup_iters = result["warmup_iters"].as<int>();
    double min_lr = result["min_lr"].as<double>();
    double dropout = result["dropout"].as<double>();
    int n_layer = result["n_layer"].as<int>();
    int n_head = result["n_head"].as<int>();
    int n_embd = result["n_embd"].as<int>();
    auto out_dir = result["out_dir"].as<std::string>();
    auto start_prompt = result["start_prompt"].as<std::string>();
    int num_samples = result["num_samples"].as<int>();
    int max_new_tokens = result["max_new_tokens"].as<int>();

    // --- Device Setup ---
    torch::Device device(torch::kCPU);
    if (device_str == "cuda" && torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using CUDA." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else if (device_str == "mps" && torch::mps::is_available()) {
        std::cout << "MPS is available! Using Metal Performance Shaders." << std::endl;
        device = torch::Device(torch::kMPS);
    } else {
        std::cout << "Using CPU." << std::endl;
    }

    // --- Model Config ---
    GPTConfig config;
    config.block_size = block_size;
    config.vocab_size = 65; // For shakespeare_char
    config.n_layer = n_layer;
    config.n_head = n_head;
    config.n_embd = n_embd;
    config.dropout = dropout;
    config.bias = false;

    // --- Main Logic ---
    if (mode == "train") {
        std::cout << "Starting in Training Mode" << std::endl;
        if (!out_dir.empty()) {
            std::filesystem::create_directory(out_dir);
        }

        GPT model(config);
        model->to(device);

        torch::optim::AdamW optimizer(model->parameters(),
                                      torch::optim::AdamWOptions(learning_rate).betas({0.9, 0.99}));

        DataLoader loader("data/shakespeare_char", batch_size, block_size);

        for (int iter = 0; iter <= max_iters; ++iter) {
            double lr = get_lr(iter, warmup_iters, lr_decay_iters, learning_rate, min_lr);
            for (auto &param_group: optimizer.param_groups()) {
                static_cast<torch::optim::AdamWOptions &>(param_group.options()).lr(lr);
            }

            auto [fst, snd] = loader.get_batch("train");
            auto X = fst.to(device);
            auto Y = snd.to(device);

            auto [logits, loss] = model->forward(X, Y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if (iter > 0 && iter % eval_interval == 0) {
                auto val_loss = estimate_loss(model, loader, device, eval_iters);
                std::cout << "Step " << iter << ": Val loss " << val_loss.item<float>() << " | LR: " << lr << std::endl;

                if (!out_dir.empty()) {
                    torch::save(model, out_dir + "/ckpt.pt");
                }
            }
        }
    } else if (mode == "sample") {
        std::cout << "Starting in Sampling Mode" << std::endl;

        // 1. First, create an instance of the model with the correct config.
        //    This is more robust than loading into a nullptr.
        GPT model(config);
        std::string checkpoint_path = out_dir + "/ckpt.pt";

        try {
            std::cout << "Attempting to load checkpoint from: " << checkpoint_path << std::endl;
            // 2. Use a specific device mapping when loading. This is good practice.
            torch::load(model, checkpoint_path, device);
        } catch (const c10::Error &e) {
            // 3. Print the DETAILED error message.
            std::cerr << "\n--------------------------------------------------" << std::endl;
            std::cerr << "FATAL: Error loading model checkpoint." << std::endl;
            std::cerr << "  File path: " << checkpoint_path << std::endl;
            std::cerr << "  Error message: " << e.what() << std::endl;
            std::cerr << "  Potential causes:" << std::endl;
            std::cerr << "  1. The checkpoint file is corrupted or incomplete." << std::endl;
            std::cerr << "  2. The model architecture in the checkpoint does not match the current configuration." <<
                    std::endl;
            std::cerr << "     (Check n_layer, n_head, n_embd, etc.)" << std::endl;
            std::cerr << "--------------------------------------------------" << std::endl;
            return 1;
        }
        std::cout << "Checkpoint loaded successfully." << std::endl;

        model->to(device);
        model->eval();

        std::string chars = " \n!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        std::map<char, int> stoi;
        std::map<int, char> itos;
        for (size_t i = 0; i < chars.length(); ++i) {
            stoi[chars[i]] = i;
            itos[i] = chars[i];
        }

        std::vector<int64_t> start_ids;
        for (char c: start_prompt) {
            start_ids.push_back(stoi[c]);
        }
        auto x = torch::tensor(start_ids, torch::kLong).view({1, -1}).to(device);

        torch::NoGradGuard no_grad;
        for (int k = 0; k < num_samples; ++k) {
            auto y = model->generate(x, max_new_tokens);
            std::cout << "---------------" << std::endl;
            for (int i = 0; i < y.size(1); ++i) {
                std::cout << itos[y[0][i].item<int64_t>()];
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Invalid mode. Use 'train' or 'sample'." << std::endl;
        std::cout << options.help() << std::endl;
    }

    return 0;
}
