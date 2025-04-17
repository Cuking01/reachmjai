cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mjai_experiment)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")

