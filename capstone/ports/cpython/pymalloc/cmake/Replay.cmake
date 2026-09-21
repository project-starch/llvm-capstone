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
# The seam the bug corpus builds through. It supplies its own pym_replay, the
# way security-tests/shared/lifetimes.c does, so the corpus stays outside the
# port and the port keeps one way in. Both the Capstone domain build and a
# hosted build read it; the corpus source picks its own probes and markers.
set(PY_CORPUS_SRC "" CACHE FILEPATH "Corpus-supplied defect program")
option(PYMALLOC_POISONCAP "Use the experimental PoisonCap pymalloc lifetime adapter" OFF)
if(PYMALLOC_POISONCAP)
  if(NOT PORT_PLATFORM STREQUAL "cheribsd")
    message(FATAL_ERROR "PYMALLOC_POISONCAP requires the CheriBSD purecap toolchain")
  endif()
  target_compile_definitions(replay-options INTERFACE PYMALLOC_POISONCAP)
  target_sources(pymalloc PRIVATE src/cheribsd/poisoncap-lifetimes.c)
  add_executable(poisoncap-probe
    "${PROJECT_SOURCE_DIR}/../../ffmpeg/buffer-pool/security-tests/cheribsd/poisoncap-probe.c")
  target_link_libraries(poisoncap-probe PRIVATE replay-options)
  set_target_properties(poisoncap-probe PROPERTIES C_EXTENSIONS ON)
  # The published revoke.h uses GNU asm.
  set_target_properties(pymalloc PROPERTIES C_EXTENSIONS ON)
  add_executable(allocator-checks security-tests/cheribsd/allocator-checks.c)
  target_link_libraries(allocator-checks PRIVATE pymalloc)
  add_dependencies(allocator-checks cpython-source)
  add_executable(pool-security src/native/main.c security-tests/shared/lifetimes.c)
  target_link_libraries(pool-security PRIVATE pymalloc)
  add_dependencies(pool-security cpython-source)
endif()
if(PORT_HOSTED)
  set(entry src/native/main.c)
  target_compile_definitions(replay-options INTERFACE SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P})
else()
  enable_language(ASM)
  set(entry src/capstone-domain/entry.c)
  target_compile_definitions(replay-options INTERFACE SIZEOF_VOID_P=16 PYMALLOC_DOMAIN)
  target_sources(pymalloc PRIVATE src/allocators/sublet/block-lifetimes.c)
endif()
add_executable(replay ${entry} src/shared/replay.c)
target_link_libraries(replay PRIVATE pymalloc)
add_dependencies(replay cpython-source)
if(PORT_HOSTED)
  target_link_options(replay PRIVATE LINKER:--gc-sections)
  add_library(pymalloc-library STATIC $<TARGET_OBJECTS:pymalloc>)
  set_target_properties(pymalloc-library PROPERTIES OUTPUT_NAME pymalloc)
  target_link_libraries(pymalloc-library PUBLIC replay-options)
  add_library(CPython::Pymalloc ALIAS pymalloc-library)
  add_executable(allocator-example examples/pymalloc.c)
  target_link_libraries(allocator-example PRIVATE CPython::Pymalloc)
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(CPython::Pymalloc)
  if(PY_CORPUS_SRC)
    add_executable(defects src/native/main.c "${PY_CORPUS_SRC}")
    target_link_libraries(defects PRIVATE pymalloc)
    add_dependencies(defects cpython-source)
    # The corpus probes use GNU inline asm, as the published revoke.h does.
    set_target_properties(defects PROPERTIES C_EXTENSIONS ON)
  endif()
  if(PORT_PLATFORM STREQUAL "native")
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
  endif()
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
  if(PY_CORPUS_SRC)
    add_executable(defects src/capstone-domain/entry.c "${PY_CORPUS_SRC}"
      "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
      "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
      "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
    target_link_libraries(defects PRIVATE pymalloc)
    add_dependencies(defects cpython-source)
    target_link_options(defects PRIVATE --gc-sections -T "${link_script}")
    set_target_properties(defects PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
  endif()
endif()
