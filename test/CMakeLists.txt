cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mjai_test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")


add_executable(test_libtorch_and_cuda test_libtorch_and_cuda.cpp)
target_link_libraries(test_libtorch_and_cuda "${TORCH_LIBRARIES}")
set_property(TARGET test_libtorch_and_cuda PROPERTY CXX_STANDARD 23)
