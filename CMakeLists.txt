cmake_minimum_required(VERSION 3.30)
project(pao-inference-mlp)

set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "-o0")
add_executable(pao-inference-mlp main.cpp)
