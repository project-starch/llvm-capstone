cmake_minimum_required(VERSION 3.25)
project(capstone_application_contract LANGUAGES C ASM)
set(CMAKE_C_STANDARD 11)
add_subdirectory(../.. runtime)
add_executable(application-contract contract.c)
capstone_configure_application(application-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144)
# Heap qualification needs an explicitly unprotected control even though
# ordinary level0 applications now receive per-object bounds by default.
add_executable(application-contract-unbounded contract.c)
capstone_configure_application(application-contract-unbounded
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144)
target_compile_definitions(application-contract-unbounded PRIVATE CAPSTONE_LEVEL0_OBJECT_BOUNDS=0)
add_executable(application-contract-sublet contract.c)
capstone_configure_application(application-contract-sublet
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 HEAP sublet HEAP_LOG 20 GRANT_BYTES 2097152)
foreach(target application-contract-unbounded application-contract-sublet)
  target_link_options(${target} PRIVATE "-Map=$<TARGET_FILE:${target}>.map")
endforeach()
add_executable(delegate-bench delegate-bench.c)
capstone_configure_application(delegate-bench
  DATA_BYTES 262144 STACK_BYTES 65536 ARENA_BYTES 65536 EXCHANGE_BYTES 65536)
add_executable(delegate-contract delegate-contract.c bounded-stack.S)
capstone_configure_application(delegate-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144)
add_executable(signal-contract signal-contract.c)
capstone_configure_application(signal-contract
  DATA_BYTES 4194304 STACK_BYTES 262144 ARENA_BYTES 262144)
add_executable(context-probe context-probe.c context-probe-asm.S)
capstone_configure_application(context-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 CONTEXT_BYTES 1048576 CONTEXTS 7)
# Each context takes 45 KiB of heap for its signal list (B8): eight of them
# and 1600 mappings need more than the contract images' 256 KiB.
add_executable(thread-probe thread-probe.c thread-probe-asm.S)
capstone_configure_application(thread-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 1048576 CONTEXT_BYTES 1048576 CONTEXTS 7)
# musl's threads (T4): each thread's stack and TLS come from the heap (musl's
# default 128 KiB stack), its area from the context arena; the most contexts
# an application may have (T5), with 45 KiB of signal list each (B8).
add_executable(pthread-probe pthread-probe.c pthread-kill-probe.c)
add_executable(level0-realloc-probe level0-realloc-probe.c)
capstone_configure_application(level0-realloc-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 1048576)
target_compile_definitions(level0-realloc-probe PRIVATE CAPSTONE_LEVEL0_STATS=1)
capstone_configure_application(pthread-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 8388608 CONTEXT_BYTES 131072 CONTEXTS 15)
# The same probe on the Sublet heap, for its lock (heap-stress).
add_executable(thread-probe-sublet thread-probe.c thread-probe-asm.S)
capstone_configure_application(thread-probe-sublet
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 CONTEXT_BYTES 1048576 CONTEXTS 7
  HEAP sublet HEAP_LOG 22)
# a 16 KiB exchange region on purpose: the datagram rule is exercised below
# the kernel's own limits
add_executable(socket-contract socket-contract.c)
capstone_configure_application(socket-contract
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 EXCHANGE_BYTES 16384)
