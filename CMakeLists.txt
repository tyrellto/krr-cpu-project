cmake_minimum_required(VERSION 3.10)
project(my_hpc_project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)

add_executable(krr_example src/main.cpp)

target_link_libraries(krr_example PRIVATE ${BLAS_LIBRARIES})
# target_link_libraries(krr_example PRIVATE /lib/x86_64-linux-gnu/libopenblas.so)