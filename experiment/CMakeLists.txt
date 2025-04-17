cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mjai_experiment)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")

add_executable(exp_network_clone exp_network_clone.cpp)
target_link_libraries(exp_network_clone "${TORCH_LIBRARIES}")
set_property(TARGET exp_network_clone PROPERTY CXX_STANDARD 23)
