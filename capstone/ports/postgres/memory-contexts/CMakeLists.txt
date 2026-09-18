cmake_minimum_required(VERSION 3.25)
include("${CMAKE_CURRENT_LIST_DIR}/../../common/cmake/Port.cmake")
project(postgres_memory_contexts LANGUAGES C)
port_check_platform()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
include(cmake/Workspace.cmake)
include(CTest)
find_package(Python3 3.11.4 REQUIRED COMPONENTS Interpreter)
# A shared definition controls region sizes in both ABIs; the runner checks
# their generated manifests before it stages either executable.
set(PG_PAYLOAD_BYTES 65536 CACHE STRING "Domain report region")
set(PG_ARENA_BYTES 67108864 CACHE STRING "Allocator arena")
set(PG_TRACE_BYTES 67108864 CACHE STRING "Trace input region")
set(PG_SCRATCH_BYTES 50331648 CACHE STRING "Replay identity tables")
set(PG_DOMAIN_STACK_BYTES 262144 CACHE STRING "Domain stack and data requirement")
file(WRITE "${CMAKE_BINARY_DIR}/regions.json"
  "{\"payload\": ${PG_PAYLOAD_BYTES}, \"arena\": ${PG_ARENA_BYTES}, \"trace\": ${PG_TRACE_BYTES}, \"scratch\": ${PG_SCRATCH_BYTES}}\n")
add_library(region-options INTERFACE)
target_compile_definitions(region-options INTERFACE
  PG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD_BYTES}UL
  PG_REPLAY_ARENA_SIZE=${PG_ARENA_BYTES}UL
  PG_REPLAY_TRACE_SIZE=${PG_TRACE_BYTES}UL
  PG_REPLAY_SCRATCH_SIZE=${PG_SCRATCH_BYTES}UL)
if(PORT_PLATFORM STREQUAL "linux-guest")
  add_subdirectory(src/linux-guest)
else()
  include(cmake/PostgreSQLSource.cmake)
  include(cmake/Replay.cmake)
  if(BUILD_TESTING)
    include(cmake/Tests.cmake)
  endif()
endif()
