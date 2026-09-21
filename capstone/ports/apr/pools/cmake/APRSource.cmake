port_read_upstream()
# The census's fetch-apr.sh caches the archive here too, so one download serves both.
port_path(APRP_ARCHIVE "${tmp_root}/dl/apr-${UPSTREAM_version}.tar.gz" "Verified APR release archive")
port_download("${APRP_ARCHIVE}")
find_program(PORT_PATCH_TOOL NAMES patch REQUIRED)
set(APRP_SOURCE "${CMAKE_BINARY_DIR}/source/apr-${UPSTREAM_version}")
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch")
add_custom_command(OUTPUT "${APRP_SOURCE}/prepared.stamp"
  BYPRODUCTS "${APRP_SOURCE}/memory/unix/apr_pools.c"
    "${APRP_SOURCE}/include/apr_pools.h" "${APRP_SOURCE}/include/apr_allocator.h"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    "${APRP_ARCHIVE}" "${UPSTREAM_sha256}" "${UPSTREAM_version}"
    "${APRP_SOURCE}" --patch-tool "${PORT_PATCH_TOOL}"
  DEPENDS "${APRP_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
  VERBATIM)
add_custom_target(apr-source DEPENDS "${APRP_SOURCE}/prepared.stamp")
set_source_files_properties("${APRP_SOURCE}/memory/unix/apr_pools.c" PROPERTIES GENERATED TRUE)
