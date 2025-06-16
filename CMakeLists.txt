cmake_minimum_required(VERSION 3.30)
project(pao-inference-mlp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(Threads)
find_package(OpenMP REQUIRED)
add_executable(pao-inference-mlp main.cpp)
target_link_libraries(pao-inference-mlp ${CMAKE_THREAD_LIBS_INIT} gomp)
target_link_libraries(pao-inference-mlp PRIVATE OpenMP::OpenMP_CXX)