#include "dataloader.h"
#include <fstream>
#include <iostream>
#include <filesystem>

void DataLoader::load_tokens(const std::string &path, torch::Tensor &tensor) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ios::end);
    const size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint16_t> tokens(file_size / sizeof(uint16_t));
    file.read(reinterpret_cast<char *>(tokens.data()), file_size);

    // Create tensor from blob and then clone it to take ownership of the memory
    tensor = torch::from_blob(tokens.data(), {static_cast<long>(tokens.size())}, torch::kUInt16).to(torch::kLong).
            clone();
}

DataLoader::DataLoader(const std::string &data_dir, const int batch_size, const int block_size)
    : batch_size(batch_size), block_size(block_size) {
    load_tokens(data_dir + "/train.bin", train_data);
    load_tokens(data_dir + "/val.bin", val_data);
    std::cout << "Train data loaded, " << train_data.size(0) << " tokens." << std::endl;
    std::cout << "Val data loaded, " << val_data.size(0) << " tokens." << std::endl;
}

std::pair<torch::Tensor, torch::Tensor> DataLoader::get_batch(const std::string &split) {
    const auto &data = (split == "train") ? train_data : val_data;
    const auto ix = torch::randint(data.size(0) - block_size, {batch_size});

    std::vector<torch::Tensor> x_list, y_list;
    for (int i = 0; i < batch_size; ++i) {
        int start_idx = ix[i].item<int>();
        x_list.push_back(data.slice(0, start_idx, start_idx + block_size));
        y_list.push_back(data.slice(0, start_idx + 1, start_idx + 1 + block_size));
    }

    auto x = torch::stack(x_list, 0);
    auto y = torch::stack(y_list, 0);
    return {x, y};
}
