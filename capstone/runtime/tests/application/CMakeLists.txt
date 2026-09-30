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
capstone_configure_application(pthread-probe
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 8388608 CONTEXT_BYTES 131072 CONTEXTS 15)
# The same probe on the Sublet heap, for its lock (heap-stress).
add_executable(thread-probe-sublet thread-probe.c thread-probe-asm.S)
capstone_configure_application(thread-probe-sublet
  DATA_BYTES 1048576 STACK_BYTES 262144 ARENA_BYTES 262144 CONTEXT_BYTES 1048576 CONTEXTS 7
  HEAP sublet HEAP_LOG 22)
