cmake_minimum_required(VERSION 3.15)
project(LlamaTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)

add_subdirectory(llama.cpp)

add_executable(MyLlamaApp main.cpp)

# Убрали common из линковки
target_link_libraries(MyLlamaApp PRIVATE llama)