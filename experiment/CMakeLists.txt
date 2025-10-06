cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(mjai_experiment)


find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(exp_network_clone exp_network_clone.cpp)
target_link_libraries(exp_network_clone "${TORCH_LIBRARIES}")
set_property(TARGET exp_network_clone PROPERTY CXX_STANDARD 20)

add_executable(fit_ev fit_ev.cpp)
target_link_libraries(fit_ev "${TORCH_LIBRARIES}")
set_property(TARGET fit_ev PROPERTY CXX_STANDARD 20)

