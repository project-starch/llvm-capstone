# The Sublet protection: the application's patch 0006, every slab chunk and cache.c object a
# child lifetime (CDERIVE) revoked when the allocator takes it back (CREVOKE). Only the Capstone
# application target executes those instructions. It replaces the hooks (0002) and the ledger.
option(MCP_SUBLET "Apply the application's patch 0006: slab chunks and cache objects as Sublet child lifetimes" OFF)
if(MCP_SUBLET AND NOT PORT_PLATFORM STREQUAL "capstone-application")
  message(FATAL_ERROR "MCP_SUBLET needs the capstone-application toolchain")
endif()
if(MCP_SUBLET)
  set(MC_VARIANT protected)
else()
  set(MC_VARIANT ported)
endif()
port_read_upstream()
# The census's fetch-memcached.sh caches the archive here too, so one download serves both.
port_path(MC_ARCHIVE "${tmp_root}/dl/memcached-${UPSTREAM_version}.tar.gz" "Verified memcached release archive")
port_download("${MC_ARCHIVE}")
find_program(PORT_PATCH_TOOL NAMES patch REQUIRED)
set(MC_SOURCE "${CMAKE_BINARY_DIR}/source/memcached-${UPSTREAM_version}")
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch"
  "${PROJECT_SOURCE_DIR}/../app/patches/*-0006-*.patch")
add_custom_command(OUTPUT "${MC_SOURCE}/prepared.stamp"
  BYPRODUCTS "${MC_SOURCE}/slabs.c" "${MC_SOURCE}/slabs.h"
    "${MC_SOURCE}/cache.c" "${MC_SOURCE}/cache.h" "${MC_SOURCE}/queue.h"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    "${MC_ARCHIVE}" "${UPSTREAM_sha256}" "${UPSTREAM_version}"
    "${MC_SOURCE}" --patch-tool "${PORT_PATCH_TOOL}" --variant ${MC_VARIANT}
  DEPENDS "${MC_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
  VERBATIM)
add_custom_target(memcached-source DEPENDS "${MC_SOURCE}/prepared.stamp")
set_source_files_properties("${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c" PROPERTIES GENERATED TRUE)
