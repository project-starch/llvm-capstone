cmake_minimum_required(VERSION 3.25)
project(capstone_application_contract LANGUAGES C ASM)
set(CMAKE_C_STANDARD 11)
add_subdirectory(../.. runtime)
add_executable(application-contract contract.c)
capstone_configure_application(application-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144)
add_executable(delegate-bench delegate-bench.c)
capstone_configure_application(delegate-bench
  DATA_BYTES 262144 STACK_BYTES 65536 ARENA_BYTES 65536 EXCHANGE_BYTES 65536)
add_executable(delegate-contract delegate-contract.c)
capstone_configure_application(delegate-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144)
