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
set(CMAKE_USER_MAKE_RULES_OVERRIDE_ASM
    "${CMAKE_CURRENT_LIST_DIR}/capstone-application-rules.cmake")
set(CMAKE_AR "${CAPSTONE_LLVM_BUILD_DIR}/bin/llvm-ar")
set(CMAKE_RANLIB "${CAPSTONE_LLVM_BUILD_DIR}/bin/llvm-ranlib")
set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_AR> s <TARGET>")
# The SDK supplies the target triple, the ISA features, the headers and the
# link script. Adding any of them here would be a second source of truth.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_SDK CAPSTONE_LLVM_BUILD_DIR)

# THE SDK IS AN INPUT TO EVERY OBJECT AND EVERY LINK, AND CMAKE CANNOT SEE IT.
# capstone-cc compiles against the musl headers named in sdk.json and links
# --whole-archive over the SDK's application runtime, then its libc, its
# builtins and its link script. None of those paths appear on a command line
# CMake generates, so CMake has no file-level dependency on any of them: after
# the SDK is rebuilt, `cmake --build` on a port's existing build tree reports
# nothing to do and leaves every object and every executable built against the
# PREVIOUS runtime. The result still runs, which is what makes it dangerous --
# a qualification run can attribute an old runtime's behaviour to a new one.
#
# Two dependencies are needed, because neither covers the other:
#
#   * recompilation. The SDK decides the headers and the codegen flags, so a
#     new SDK invalidates the objects, not just the link. CMake reruns a
#     compile when the command line changes, so the SDK's identity goes INTO
#     the command line as a define (capstone-application-sdk.cmake). It cannot
#     be added here: CMAKE_C_FLAGS_INIT only seeds the cache on the first
#     configure, so a digest placed there would freeze at its first value.
#
#   * the link. A target that links only prebuilt archives has no source of
#     its own to recompile, so nothing above would retrigger its link; its
#     LINK_DEPENDS names the SDK files the driver actually reads.
#
# Both need a reconfigure to notice a rebuilt SDK, which is the directory's
# CMAKE_CONFIGURE_DEPENDS, also registered in that file.
set(CAPSTONE_SDK_ARTIFACTS "${CAPSTONE_SDK}/sdk.json" "${CAPSTONE_SDK}/capstone-cc")
string(JSON capstone_sdk_ld GET "${capstone_sdk_json}" ld)
foreach(artifact runtime libc builtins script)
  string(JSON capstone_sdk_${artifact} GET "${capstone_sdk_json}" ${artifact})
  if(NOT EXISTS "${capstone_sdk_${artifact}}")
    message(FATAL_ERROR "SDK ${CAPSTONE_SDK} names a ${artifact} that does not exist: "
                        "'${capstone_sdk_${artifact}}'. Rebuild the SDK.")
  endif()
  list(APPEND CAPSTONE_SDK_ARTIFACTS "${capstone_sdk_${artifact}}")
endforeach()
# What the digest covers: the contents of the six files above -- sdk.json
# carries the compiler path, the linker path and the codegen flags, so a
# changed flag list is a changed digest. The compiler and linker BINARIES are
# identified by size and timestamp rather than content: hashing a debug clang
# at every configure costs more than the whole configure, and a rebuilt clang
# moves both. A build that needs the compiler's content identified has it in
# the image manifest's compiler_sha256 (ports/common/application/build.py).
set(capstone_sdk_identity "")
foreach(artifact IN LISTS CAPSTONE_SDK_ARTIFACTS)
  file(SHA256 "${artifact}" artifact_sha)
  string(APPEND capstone_sdk_identity "${artifact_sha}")
endforeach()
foreach(binary "${capstone_sdk_cc}" "${capstone_sdk_ld}")
  if(EXISTS "${binary}")
    file(SIZE "${binary}" binary_size)
    file(TIMESTAMP "${binary}" binary_stamp "%Y%m%d%H%M%S" UTC)
    string(APPEND capstone_sdk_identity "${binary_size}:${binary_stamp}")
  endif()
endforeach()
string(SHA256 capstone_sdk_digest "${capstone_sdk_identity}")
string(SUBSTRING "${capstone_sdk_digest}" 0 16 CAPSTONE_SDK_IDENTITY)
set(CMAKE_PROJECT_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/capstone-application-sdk.cmake")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_SDK_ARTIFACTS CAPSTONE_SDK_IDENTITY)
