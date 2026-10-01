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
add_executable(signal-contract signal-contract.c)
capstone_configure_application(signal-contract
  DATA_BYTES 4194304 STACK_BYTES 262144 ARENA_BYTES 262144)
add_executable(context-probe context-probe.c context-probe-asm.S)
capstone_configure_application(context-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 CONTEXT_BYTES 1048576)
# a 16 KiB exchange region on purpose: the datagram rule is exercised below
# the kernel's own limits
add_executable(socket-contract socket-contract.c)
capstone_configure_application(socket-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 EXCHANGE_BYTES 16384)
