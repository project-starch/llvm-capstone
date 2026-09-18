add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/capstone-runtime")
add_library(replay-options INTERFACE)
target_link_libraries(replay-options INTERFACE Capstone::Runtime)
target_include_directories(replay-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared" "${PYMALLOC_SOURCE}/Include/internal")
target_compile_definitions(replay-options INTERFACE PYMALLOC_PORT Py_BUILD_CORE WITH_PYMALLOC SIZEOF_SIZE_T=8 SIZEOF_INT=4)
target_compile_options(replay-options INTERFACE -ffunction-sections -fdata-sections -Wall -Wextra)
add_library(pymalloc OBJECT "${PYMALLOC_SOURCE}/Objects/obmalloc.c" src/shared/backing.c)
target_link_libraries(pymalloc PUBLIC replay-options)
add_dependencies(pymalloc cpython-source)
if(PORT_PLATFORM STREQUAL "native")
  set(entry src/native/main.c)
  target_compile_definitions(replay-options INTERFACE SIZEOF_VOID_P=8)
else()
  enable_language(ASM)
  set(entry src/capstone-domain/entry.c)
  target_compile_definitions(replay-options INTERFACE SIZEOF_VOID_P=16 PYMALLOC_DOMAIN)
  target_sources(pymalloc PRIVATE src/allocators/sublet/block-lifetimes.c)
endif()
add_executable(replay ${entry} src/shared/replay.c)
target_link_libraries(replay PRIVATE pymalloc)
add_dependencies(replay cpython-source)
if(PORT_PLATFORM STREQUAL "native")
  target_link_options(replay PRIVATE LINKER:--gc-sections)
  set(reference_source "${CMAKE_BINARY_DIR}/reference/Python-${UPSTREAM_version}")
  add_custom_command(OUTPUT "${reference_source}/prepared.stamp"
    BYPRODUCTS "${reference_source}/Objects/obmalloc.c" "${reference_source}/Include/internal/pycore_obmalloc.h"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
      "${PYMALLOC_ARCHIVE}" "${UPSTREAM_sha256}" "${UPSTREAM_version}"
      "${reference_source}" --patch-tool "${PORT_PATCH_TOOL}" --variant reference
    DEPENDS "${PYMALLOC_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    VERBATIM)
  add_custom_target(cpython-reference-source DEPENDS "${reference_source}/prepared.stamp")
  add_executable(replay-reference src/native/main.c src/shared/replay.c src/shared/backing.c
    "${reference_source}/Objects/obmalloc.c")
  set_source_files_properties("${reference_source}/Objects/obmalloc.c" PROPERTIES GENERATED TRUE)
  target_include_directories(replay-reference BEFORE PRIVATE "${reference_source}/Include/internal")
  target_compile_definitions(replay-reference PRIVATE PYMALLOC_REFERENCE)
  target_link_libraries(replay-reference PRIVATE replay-options)
  target_link_options(replay-reference PRIVATE LINKER:--gc-sections)
  add_dependencies(replay-reference cpython-reference-source)
else()
  target_sources(replay PRIVATE
    "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
    "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
    "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
  set(link_script "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/link.ld")
  target_link_options(replay PRIVATE --gc-sections -T "${link_script}")
  set_target_properties(replay PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
  add_executable(pool-security src/capstone-domain/entry.c security-tests/shared/lifetimes.c
    "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
    "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
    "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
  target_link_libraries(pool-security PRIVATE pymalloc)
  add_dependencies(pool-security cpython-source)
  target_link_options(pool-security PRIVATE --gc-sections -T "${link_script}")
  set_target_properties(pool-security PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
endif()
