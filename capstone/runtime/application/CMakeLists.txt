cmake_minimum_required(VERSION 3.25)
# The current capability compiler's application runtime is verified at -O1.
# Keep an explicit caller override possible for compiler regression work.
set(CMAKE_C_FLAGS_RELEASE "-O1 -DNDEBUG" CACHE STRING "Release C flags")
project(capstone_application LANGUAGES C ASM)
set(CMAKE_C_STANDARD 11)
set(CAPSTONE_APPLICATION_INPUTS "" CACHE STRING "Sources, objects and archives containing main")
set(CAPSTONE_APPLICATION_NAME application CACHE STRING "Output basename")
set(CAPSTONE_APPLICATION_DATA_BYTES 4194304 CACHE STRING "Globals and stack budget")
set(CAPSTONE_APPLICATION_STACK_BYTES 1048576 CACHE STRING "Stack budget")
set(CAPSTONE_APPLICATION_ARENA_BYTES 4194304 CACHE STRING "Application heap size")
set(CAPSTONE_APPLICATION_HEAP level0 CACHE STRING "level0 or sublet")
set(CAPSTONE_APPLICATION_HEAP_LOG 22 CACHE STRING "Sublet pool size as log2(bytes)")
option(CAPSTONE_APPLICATION_SDK "Build a reusable compiler SDK instead of an application" OFF)
if(NOT CAPSTONE_APPLICATION_INPUTS AND NOT CAPSTONE_APPLICATION_SDK)
  message(FATAL_ERROR "Set CAPSTONE_APPLICATION_INPUTS to sources, objects or archives")
endif()
add_subdirectory(.. runtime)
if(CAPSTONE_APPLICATION_SDK)
  set(CAPSTONE_APPLICATION_NAME application-runtime)
  add_library(${CAPSTONE_APPLICATION_NAME} STATIC)
else()
  add_executable(${CAPSTONE_APPLICATION_NAME})
endif()
foreach(input IN LISTS CAPSTONE_APPLICATION_INPUTS)
  if(input MATCHES "\\.a$")
    target_link_libraries(${CAPSTONE_APPLICATION_NAME} PRIVATE "${input}")
  else()
    target_sources(${CAPSTONE_APPLICATION_NAME} PRIVATE "${input}")
  endif()
endforeach()
capstone_configure_application(${CAPSTONE_APPLICATION_NAME}
  DATA_BYTES ${CAPSTONE_APPLICATION_DATA_BYTES}
  STACK_BYTES ${CAPSTONE_APPLICATION_STACK_BYTES}
  ARENA_BYTES ${CAPSTONE_APPLICATION_ARENA_BYTES}
  HEAP ${CAPSTONE_APPLICATION_HEAP} HEAP_LOG ${CAPSTONE_APPLICATION_HEAP_LOG})
if(CAPSTONE_APPLICATION_SDK)
  get_filename_component(capstone "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sdk.json" CONTENT
"{
  \"version\": 1,
  \"cc\": \"${CMAKE_C_COMPILER}\",
  \"ld\": \"${CMAKE_LINKER}\",
  \"musl\": \"${PORT_MUSL_ROOT}\",
  \"script\": \"${capstone}/my_first_domain/link.ld\",
  \"runtime\": \"$<TARGET_FILE:application-runtime>\",
  \"libc\": \"${CAPSTONE_MUSL_ARCHIVE}\",
  \"builtins\": \"$<TARGET_FILE:capstone-application-builtins>\"
}
")
  configure_file(../host/capstone_vm/compiler.py "${CMAKE_CURRENT_BINARY_DIR}/capstone-cc" COPYONLY)
  file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/capstone-cc"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()
