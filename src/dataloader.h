#pragma once
#include <torch/torch.h>
#include <string>

class DataLoader {
public:
    DataLoader(const std::string &data_dir, int batch_size, int block_size);

    std::pair<torch::Tensor, torch::Tensor> get_batch(const std::string &split);

private:
    static void load_tokens(const std::string &path, torch::Tensor &tensor);

    int batch_size;
    int block_size;
    torch::Tensor train_data;
    torch::Tensor val_data;
};
