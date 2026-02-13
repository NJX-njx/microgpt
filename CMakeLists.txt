cmake_minimum_required(VERSION 3.24)
project(microgpt_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(microgpt_cuda microgpt_cuda.cu)
target_compile_options(microgpt_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
