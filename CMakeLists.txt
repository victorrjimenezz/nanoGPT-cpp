cmake_minimum_required(VERSION 3.16)
project(NanoGPT_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set the architecture for Apple Silicon if on macOS.
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Find the LibTorch package, including the component for distributed training.
find_package(Torch REQUIRED COMPONENTS c10d)

# Add the executable target.
add_executable(nanogpt_cpp
    src/main.cpp
    src/model.cpp
    src/dataloader.cpp
)

# Add the 'src' directory to the include path for finding cxxopts.hpp.
target_include_directories(nanogpt_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link against the Torch libraries.
# The necessary linker search paths will be provided by the build script.
target_link_libraries(nanogpt_cpp PRIVATE "${TORCH_LIBRARIES}")

# The cxx11-abi flag is for Linux with the GNU standard library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(nanogpt_cpp PRIVATE "_GLIBCXX_USE_CXX11_ABI=0")
endif()
