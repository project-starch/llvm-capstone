# Builds slotins.c as a DynamoRIO client: cmake -S <this dir> -B <build> -DDynamoRIO_DIR=<DynamoRIO>/cmake
cmake_minimum_required(VERSION 3.7)
project(slotins C CXX)   # DynamoRIO's CMake config reads the C++ pointer size
set(CMAKE_C_STANDARD 17)   # GCC 15 defaults to C23, where bool is a keyword DynamoRIO 11.3 typedefs
find_package(DynamoRIO REQUIRED)
add_library(slotins SHARED slotins.c)
configure_DynamoRIO_client(slotins)
use_DynamoRIO_extension(slotins drmgr)
use_DynamoRIO_extension(slotins drreg)
use_DynamoRIO_extension(slotins drwrap)
