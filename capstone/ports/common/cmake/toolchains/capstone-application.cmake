# A port component built as a Capstone APPLICATION: capability pointers and a
# Capstone libc, driven by an SDK's capstone-cc rather than by bare clang.
#
# WHY THIS EXISTS BESIDE capstone-domain.cmake. That toolchain builds a
# FREESTANDING domain -- my_first_domain/link.ld, start.S, its own entry, no
# libc -- which is what the physical arms of this tree's allocator corpora run.
# It cannot build a case whose driver calls printf and malloc, and that is the
# shape every corpus in bug-corpora/ uses for `main()`. The virtual address
# space runs applications, not bare-metal domains, so its arms need the hosted
# sources and a libc, with capability pointers underneath. Hence: HOSTED, like
# native and cheribsd, and cross-compiled, like capstone-domain.
#
# The SDK decides the libc, the runtime, the link script and the codegen flags;
# this file does not restate any of them. Point CAPSTONE_SDK at the SDK
# directory (its `sdk.json` names everything else) and the component's own
# CMake is unchanged.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR capstone64)
# capstone-cc compiles and links in one step and has no shared-object mode, so
# CMake's ABI probe must stop at the archive rather than link a test program.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(PORT_PLATFORM capstone-application CACHE STRING "Execution platform" FORCE)
include("${CMAKE_CURRENT_LIST_DIR}/../Workspace.cmake")
port_path(CAPSTONE_SDK "${tmp_root}/virtual/sdk" "A built Capstone application SDK")
if(NOT EXISTS "${CAPSTONE_SDK}/capstone-cc" OR NOT EXISTS "${CAPSTONE_SDK}/sdk.json")
  message(FATAL_ERROR "CAPSTONE_SDK must name a built SDK directory holding "
                      "capstone-cc and sdk.json; got '${CAPSTONE_SDK}'.")
endif()
set(CMAKE_C_COMPILER "${CAPSTONE_SDK}/capstone-cc")
set(CMAKE_C_COMPILER_ID Clang)
# The compiler's OWN headers. capstone-cc compiles with -nostdinc and then adds
# only the four musl directories, so the freestanding headers clang ships --
# stdatomic.h, stddef.h, float.h, limits.h -- are not on the path; musl carries
# a stdarg.h of its own, which is why most sources build anyway and FFmpeg's
# buffer.c does not. capstone-domain.cmake adds the resource directory for the
# same reason; this does the same, reading the compiler out of the SDK's own
# sdk.json so the two cannot drift apart.
file(READ "${CAPSTONE_SDK}/sdk.json" capstone_sdk_json)
string(JSON capstone_sdk_cc GET "${capstone_sdk_json}" cc)
execute_process(COMMAND "${capstone_sdk_cc}" -print-resource-dir
  OUTPUT_VARIABLE capstone_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(CMAKE_C_FLAGS_INIT "-isystem \"${capstone_resource_dir}/include\"")
# Object names: see capstone-application-rules.cmake. It has to be a rules
# override rather than a plain set() here, because CMake assigns the extension
# after the toolchain file has run.
set(CMAKE_USER_MAKE_RULES_OVERRIDE_C
    "${CMAKE_CURRENT_LIST_DIR}/capstone-application-rules.cmake")
set(CMAKE_AR "${CAPSTONE_LLVM_BUILD_DIR}/bin/llvm-ar")
set(CMAKE_RANLIB "${CAPSTONE_LLVM_BUILD_DIR}/bin/llvm-ranlib")
set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_AR> s <TARGET>")
# The SDK supplies the target triple, the ISA features, the headers and the
# link script. Adding any of them here would be a second source of truth.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_SDK CAPSTONE_LLVM_BUILD_DIR)
