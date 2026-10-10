add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/capstone-runtime")
add_library(replay-options INTERFACE)
target_link_libraries(replay-options INTERFACE Capstone::Runtime)
target_include_directories(replay-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared" "${PYMALLOC_SOURCE}/Include/internal")
target_compile_definitions(replay-options INTERFACE PYMALLOC_PORT Py_BUILD_CORE WITH_PYMALLOC SIZEOF_SIZE_T=8 SIZEOF_INT=4
  SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P})
target_compile_options(replay-options INTERFACE -ffunction-sections -fdata-sections -Wall -Wextra)
add_library(pymalloc OBJECT "${PYMALLOC_SOURCE}/Objects/obmalloc.c" src/shared/backing.c)
target_link_libraries(pymalloc PUBLIC replay-options)
add_dependencies(pymalloc cpython-source)
if(NOT PORT_HOSTED)
  message(FATAL_ERROR "the pymalloc port builds hosted programs only: capstone-application, cheribsd or native")
endif()
# The seam the bug corpus builds through: a corpus source supplies its own
# pym_replay, so the corpus stays outside the port and the port keeps one way in.
set(PY_CORPUS_SRC "" CACHE FILEPATH "Corpus-supplied defect program")
add_executable(replay src/native/main.c src/shared/replay.c)
target_link_libraries(replay PRIVATE pymalloc)
add_dependencies(replay cpython-source)
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
  # The corpus probes use GNU inline asm.
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
