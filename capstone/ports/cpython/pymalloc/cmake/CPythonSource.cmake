# The Sublet protection (patch 0003): every pymalloc block a child lifetime of
# its arena (CDERIVE), revoked on free (CREVOKE). Only the Capstone application
# target executes those instructions.
option(PYMALLOC_SUBLET "Apply patch 0003, pymalloc's blocks as Sublet child lifetimes" OFF)
if(PYMALLOC_SUBLET AND NOT PORT_PLATFORM STREQUAL "capstone-application")
  message(FATAL_ERROR "PYMALLOC_SUBLET needs the capstone-application toolchain")
endif()
if(PYMALLOC_SUBLET)
  set(PYMALLOC_VARIANT protected)
else()
  set(PYMALLOC_VARIANT ported)
endif()
port_read_upstream()
port_path(PYMALLOC_ARCHIVE "${PYMALLOC_WORK}/download/Python-${UPSTREAM_version}.tgz" "Verified CPython release archive")
port_download("${PYMALLOC_ARCHIVE}")
find_program(PORT_PATCH_TOOL NAMES patch REQUIRED)
set(PYMALLOC_SOURCE "${CMAKE_BINARY_DIR}/source/Python-${UPSTREAM_version}")
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch")
add_custom_command(OUTPUT "${PYMALLOC_SOURCE}/prepared.stamp"
  BYPRODUCTS "${PYMALLOC_SOURCE}/Objects/obmalloc.c" "${PYMALLOC_SOURCE}/Include/internal/pycore_obmalloc.h"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    "${PYMALLOC_ARCHIVE}" "${UPSTREAM_sha256}" "${UPSTREAM_version}"
    "${PYMALLOC_SOURCE}" --patch-tool "${PORT_PATCH_TOOL}" --variant ${PYMALLOC_VARIANT}
  DEPENDS "${PYMALLOC_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
  VERBATIM)
add_custom_target(cpython-source DEPENDS "${PYMALLOC_SOURCE}/prepared.stamp")
set_source_files_properties("${PYMALLOC_SOURCE}/Objects/obmalloc.c" PROPERTIES GENERATED TRUE)
