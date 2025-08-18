# nanoGPT-C++

![nanoGPT](https://raw.githubusercontent.com/karpathy/nanoGPT/master/assets/nanogpt.jpg)

This repository is a C++ port of Andrej Karpathy's phenomenal [nanoGPT](https://github.com/karpathy/nanoGPT). It aims to replicate the core functionality—model definition, training, and sampling—using C++ and the LibTorch library.

The goal is to maintain the simplicity and hackability of the original while leveraging the performance and deployment advantages of a compiled C++ environment. Where `nanoGPT` is the simple, fast Python speedboat, this is its C++ counterpart, built for efficiency and integration into C++ applications.

This port currently implements:
*   The complete GPT-2 model architecture in `torch::nn::Module`.
*   A lightweight binary data loader for tokenized datasets.
*   A flexible training and sampling executable with command-line argument parsing.
*   Support for CPU, CUDA, and Apple Silicon (MPS) backends.

## Prerequisites

Before you begin, you will need a C++ development environment and the LibTorch library.

1.  **C++ Compiler**: A modern C++17 compatible compiler (e.g., g++, Clang). For macOS, Apple's Command Line Tools are sufficient.
2.  **CMake**: Version 3.16 or newer.
3.  **LibTorch**: The C++ distribution of PyTorch.
    *   Download it from the [PyTorch website](https://pytorch.org/get-started/locally/).
    *   **Crucially, select the `LibTorch` compute platform.**
    *   For macOS on Apple Silicon (M1/M2/M3/M4), download the **Arm64** version.
    *   For Linux, download the **Pre-cxx11 ABI** version for best compatibility.
    *   Unzip the downloaded file. You will get a directory named `libtorch`.

## How to Build

The build process is managed by CMake and a simple shell script. It requires one environment variable to be set.

### Step 1: Set Environment Variables

You must tell the build script where to find your unzipped LibTorch directory by setting the `LIBTORCH_PATH` environment variable.

```bash
# Example: replace with the actual path on your system
export LIBTORCH_PATH=/path/to/your/libtorch
```

**(Optional) Using a Custom Compiler:**
If you built LibTorch from source with a custom compiler, you must also set the `CC` and `CXX` environment variables to point to your C and C++ compilers.
```bash
export CC=/path/to/your/custom/clang
export CXX=/path/to/your/custom/clang++
```

### Step 2: Prepare the Dataset (Requires Python)

The C++ code expects the training data to be pre-tokenized into binary files (`train.bin`, `val.bin`). You must use the scripts from the original `nanoGPT` repository to do this.

```bash
# First, clone the original nanoGPT repository if you haven't already
git clone https://github.com/karpathy/nanoGPT.git

# Install its Python dependencies
cd nanoGPT
pip install -r requirements.txt

# Run the preparation script for the Shakespeare character-level dataset
python data/shakespeare_char/prepare.py

# Now, copy the generated `data` directory into your `nanogpt-cpp` project root.
# Your project structure should look like:
# nanogpt-cpp/
# ├── data/
# │   └── shakespeare_char/
# │       ├── train.bin
# │       └── val.bin
# └── src/
#     └── ...
```

### Step 3: Download Command-Line Parser

This project uses `cxxopts` for flexible command-line argument parsing. It's a single header file.

```bash
# Navigate into your nanogpt-cpp/src directory
cd src

# Download the header file
wget https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp
# Or, if you don't have wget:
# curl -O https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp

cd .. # Go back to the project root
```

### Step 4: Compile the Code

A helper script `build.sh` is provided.

1.  **Make the script executable:**
    ```bash
    chmod +x build.sh
    ```
2.  **Run the build script:**
    ```bash
    ./build.sh
    ```
    The script will use your environment variables to configure and build the project. The final executable, `nanogpt_cpp`, will be in the `build/` directory.

## How to Run

All commands should be run from within the `build` directory.

### Reproducing the "MacBook CPU" Training

This run reproduces the quick, small-scale training from the original README to get a feel for the model.

**1. Train the Model (~3 minutes on CPU)**

```bash
./build/nanogpt_cpp --mode=train \
  --device=cpu \
  --eval_iters=20 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0
```

The validation loss should drop to around **~1.88**. A checkpoint will be saved to `out-shakespeare-char-cpp/ckpt.pt`.

**2. Sample from the Model**

```bash
./build/nanogpt_cpp --mode=sample --device=cpu
```

This will load the checkpoint and generate text with the "character gestalt" of Shakespearean text:
```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

### A Better 10-Minute Training Run on Apple Silicon (MPS)

For more coherent results, we can scale up the model and train for longer on the GPU. On an M-series Mac, this should take about 10 minutes.

**1. Train the Larger Model**

```bash
./build/nanogpt_cpp --mode=train \
  --device=mps \
  --out_dir="out-shakespeare-char-cpp-10min" \
  --max_iters=1250 \
  --lr_decay_iters=1250 \
  --eval_interval=250 \
  --eval_iters=200 \
  --block_size=256 \
  --batch_size=64 \
  --n_layer=6 \
  --n_head=6 \
  --n_embd=384 \
  --dropout=0.2
```

The validation loss should now go below **1.5**. The checkpoint is saved in `out-shakespeare-char-cpp-10min/`.

**2. Sample from the Improved Model**

*Note: When sampling, you must provide the same model architecture flags used during training.*
```bash
./build/nanogpt_cpp --mode=sample \
  --device=mps \
  --out_dir="out-shakespeare-char-cpp-10min" \
  --n_layer=6 \
  --n_head=6 \
  --n_embd=384
```
The output should now contain correctly spelled words and more coherent sentences:
```
KING HENRY IV:
Therefore, my lord, I'll see a dozen of his subjects,
And that hath highly drawn him to his sword.

GLOUCESTER:
Why, then, is it not a subject of the world,
That have I not?
```

## Command-Line Arguments

The executable supports various flags. Run with `--help` to see all options and their defaults.
```bash
./build/nanogpt_cpp --help
```

## Acknowledgements

This project is entirely a C++ port and would not exist without the clean, brilliant, and educational foundation of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). All credit for the architecture, training recipes, and philosophy belongs to the original work.