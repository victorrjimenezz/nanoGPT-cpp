#!/bin/bash

# This script builds the nanoGPT-cpp project.
# It requires the LIBTORCH_PATH environment variable.
# It optionally uses CC and CXX for custom compilers.

set -euxo pipefail

# --- Environment Variable Check ---
if [ -z "${LIBTORCH_PATH:-}" ]; then
  echo "Error: LIBTORCH_PATH environment variable is not set."
  exit 1
fi

# --- Build Steps ---

echo "Step 1: Cleaning up previous build directory..."
rm -rf build
mkdir build
cd build

echo "Step 2: Configuring the project with CMake..."

# Start with the base CMake command
CMAKE_CMD="cmake .. -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH}"

# If a custom compiler is used, we add flags to tell the linker where
# to find its corresponding standard libraries.
if [ -n "${CXX:-}" ]; then
  echo "Using custom compilers specified by CC and CXX environment variables."
  CMAKE_CMD+=" -DCMAKE_C_COMPILER=${CC}"
  CMAKE_CMD+=" -DCMAKE_CXX_COMPILER=${CXX}"
  
  # Derive the library path from the compiler path
  # e.g., /path/to/bin/clang++ -> /path/to/lib
  CUSTOM_COMPILER_DIR=$(dirname "${CXX}")
  CUSTOM_LIB_DIR="${CUSTOM_COMPILER_DIR}/../lib"

  # Add linker flags:
  # -L tells the linker where to search for libraries at build time.
  # -Wl,-rpath tells the linker to embed a runtime search path into the executable.
  LINKER_FLAGS="-L${CUSTOM_LIB_DIR} -Wl,-rpath,${CUSTOM_LIB_DIR}"

  # Pass these flags to CMake
  CMAKE_CMD+=" -DCMAKE_EXE_LINKER_FLAGS='${LINKER_FLAGS}'"
else
  echo "Using default system compilers."
fi

# Execute the constructed CMake command
eval "$CMAKE_CMD"

echo "Step 3: Building the project with make..."
# Use VERBOSE=1 for detailed output
CORES=$(uname -s | grep -q Darwin && sysctl -n hw.ncpu || nproc)
make VERBOSE=1 -j"${CORES}"

echo "Build finished successfully! The executable is in the build/ directory."
