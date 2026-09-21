port_read_upstream()
# The census's fetch-memcached.sh caches the archive here too, so one download serves both.
port_path(MC_ARCHIVE "${tmp_root}/dl/memcached-${UPSTREAM_version}.tar.gz" "Verified memcached release archive")
port_download("${MC_ARCHIVE}")
find_program(PORT_PATCH_TOOL NAMES patch REQUIRED)
set(MC_SOURCE "${CMAKE_BINARY_DIR}/source/memcached-${UPSTREAM_version}")
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch")
add_custom_command(OUTPUT "${MC_SOURCE}/prepared.stamp"
  BYPRODUCTS "${MC_SOURCE}/slabs.c" "${MC_SOURCE}/slabs.h"
    "${MC_SOURCE}/cache.c" "${MC_SOURCE}/cache.h" "${MC_SOURCE}/queue.h"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    "${MC_ARCHIVE}" "${UPSTREAM_sha256}" "${UPSTREAM_version}"
    "${MC_SOURCE}" --patch-tool "${PORT_PATCH_TOOL}"
  DEPENDS "${MC_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
  VERBATIM)
add_custom_target(memcached-source DEPENDS "${MC_SOURCE}/prepared.stamp")
set_source_files_properties("${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c" PROPERTIES GENERATED TRUE)
