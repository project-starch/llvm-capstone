cmake_minimum_required(VERSION 3.25)
# The current capability compiler's application runtime is verified at -O1.
# Keep an explicit caller override possible for compiler regression work.
set(CMAKE_C_FLAGS_RELEASE "-O1 -DNDEBUG" CACHE STRING "Release C flags")
project(capstone_application LANGUAGES C ASM)
set(CMAKE_C_STANDARD 11)
option(CAPSTONE_APPLICATION_VIRTUAL "Use the virtual C runtime adapter (OFF: the physical profile)" ON)
set(profile_codegen)
set(sdk_profile physical)
if(CAPSTONE_APPLICATION_VIRTUAL)
  set(sdk_profile virtual)
  set(sdk_allocator mallocng)
  include("${CMAKE_CURRENT_LIST_DIR}/../cmake/VirtualCodegen.cmake")
  set(profile_codegen ${CAPSTONE_VIRTUAL_CODEGEN})
  list(JOIN profile_codegen " " virtual_codegen)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:SHELL:${virtual_codegen}>")
  list(JOIN profile_codegen "\", \"" sdk_codegen)
  set(sdk_codegen "[\"${sdk_codegen}\"]")
else()
  set(sdk_profile physical)
  set(sdk_allocator level0)
  set(sdk_codegen "[]")
endif()
# The virtual heap's measurement switches (runtime/virtual/heap.c). Both ON is the profile as
# qualified; OFF builds the arms of the spatial-versus-temporal comparison: LIFETIMES=OFF gives
# every object exact bounds but no lifetime of its own, SCRUB_FREE=OFF leaves a freed payload's
# bytes and tags in place. The physical profile's level0 has its own switches.
option(CAPSTONE_HEAP_LIFETIMES "Virtual heap: a lifetime per object, CDERIVE on malloc and CREVOKE on free" ON)
option(CAPSTONE_HEAP_SCRUB_FREE "Virtual heap: zero a payload on free" ON)
if(NOT CAPSTONE_APPLICATION_VIRTUAL AND NOT (CAPSTONE_HEAP_LIFETIMES AND CAPSTONE_HEAP_SCRUB_FREE))
  message(FATAL_ERROR "CAPSTONE_HEAP_LIFETIMES and CAPSTONE_HEAP_SCRUB_FREE are switches of the virtual heap")
endif()
foreach(switch IN ITEMS CAPSTONE_HEAP_LIFETIMES CAPSTONE_HEAP_SCRUB_FREE)
  if(${switch})
    set(sdk_${switch} true)
  else()
    set(sdk_${switch} false)
  endif()
endforeach()
option(CAPSTONE_CHECK_COMPILER_ABI "Qualify Capstone direct-call codegen" ON)
if(CAPSTONE_CHECK_COMPILER_ABI)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../host/capstone_vm/compiler.py"
            --check-compiler "${CMAKE_C_COMPILER}" ${profile_codegen}
    RESULT_VARIABLE compiler_probe_status
    ERROR_VARIABLE compiler_probe_error)
  if(NOT compiler_probe_status EQUAL 0)
    message(FATAL_ERROR "Capstone compiler ABI check failed: ${compiler_probe_error}")
  endif()
endif()
# The heap is level0 on the physical profile and musl mallocng on the virtual one; the
# Sublet heap and its HEAP/HEAP_LOG selection were removed on 2026-10-11. A recipe that
# still passes them would otherwise build level0 under its old label.
if(DEFINED CAPSTONE_APPLICATION_HEAP OR DEFINED CAPSTONE_APPLICATION_HEAP_LOG)
  message(FATAL_ERROR "CAPSTONE_APPLICATION_HEAP and CAPSTONE_APPLICATION_HEAP_LOG were removed with the Sublet heap; drop them")
endif()
set(CAPSTONE_APPLICATION_INPUTS "" CACHE STRING "Sources, objects and archives containing main")
set(CAPSTONE_APPLICATION_NAME application CACHE STRING "Output basename")
set(CAPSTONE_APPLICATION_DATA_BYTES 4194304 CACHE STRING "Globals and stack budget")
set(CAPSTONE_APPLICATION_STACK_BYTES 1048576 CACHE STRING "Stack budget")
set(CAPSTONE_APPLICATION_ARENA_BYTES 4194304 CACHE STRING "Application heap size")
set(CAPSTONE_APPLICATION_GRANT_BYTES 0 CACHE STRING "Shared backing for the heap and inner allocators")
set(CAPSTONE_APPLICATION_EXCHANGE_BYTES 262144 CACHE STRING "Exchange region for delegated syscalls (ABI v2)")
# Threads (docs/plans/delegation-threads.md): how many run at once besides the
# first, each with a transport of its own, and the context arena their areas
# come from (a musl thread's area is 1280 bytes; its stack and TLS are heap).
set(CAPSTONE_APPLICATION_CONTEXTS 15 CACHE STRING "Threads at once besides the first (0 to 15)")
set(CAPSTONE_APPLICATION_CONTEXT_BYTES 65536 CACHE STRING "Context arena for thread areas")
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
  EXCHANGE_BYTES ${CAPSTONE_APPLICATION_EXCHANGE_BYTES}
  GRANT_BYTES ${CAPSTONE_APPLICATION_GRANT_BYTES}
  CONTEXTS ${CAPSTONE_APPLICATION_CONTEXTS}
  CONTEXT_BYTES ${CAPSTONE_APPLICATION_CONTEXT_BYTES})
if(CAPSTONE_APPLICATION_SDK)
  get_filename_component(capstone "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sdk.json" CONTENT
"{
  \"version\": 1,
  \"application_abi\": 2,
  \"profile\": \"${sdk_profile}\",
  \"allocator\": \"${sdk_allocator}\",
  \"heap_lifetimes\": ${sdk_CAPSTONE_HEAP_LIFETIMES},
  \"heap_scrub_free\": ${sdk_CAPSTONE_HEAP_SCRUB_FREE},
  \"codegen\": ${sdk_codegen},
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
