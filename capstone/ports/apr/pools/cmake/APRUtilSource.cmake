# apr-util's bucket allocator, prepared beside APR: its own archive, its own
# patch series under patches/apr-util/, the same script. The census's
# fetch-apr-util.sh caches the archive at the same path.
file(READ "${PROJECT_SOURCE_DIR}/upstream-apr-util.json" apu_json)
foreach(key name version url sha256)
  string(JSON APU_${key} GET "${apu_json}" ${key})
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/upstream-apr-util.json")
port_path(APU_ARCHIVE "${tmp_root}/dl/apr-util-${APU_version}.tar.gz" "Verified apr-util release archive")
# port_download reads the UPSTREAM_* variables; lend it apr-util's for the call.
set(saved_url "${UPSTREAM_url}")
set(saved_sha "${UPSTREAM_sha256}")
set(UPSTREAM_url "${APU_url}")
set(UPSTREAM_sha256 "${APU_sha256}")
port_download("${APU_ARCHIVE}")
set(UPSTREAM_url "${saved_url}")
set(UPSTREAM_sha256 "${saved_sha}")
set(APU_SOURCE "${CMAKE_BINARY_DIR}/source/apr-util-${APU_version}")
file(GLOB apu_patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/apr-util/*.patch")
add_custom_command(OUTPUT "${APU_SOURCE}/prepared.stamp"
  BYPRODUCTS "${APU_SOURCE}/buckets/apr_buckets_alloc.c"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    "${APU_ARCHIVE}" "${APU_sha256}" "${APU_version}" "${APU_SOURCE}"
    --patch-tool "${PORT_PATCH_TOOL}" --name apr-util
    --patches "${PROJECT_SOURCE_DIR}/patches/apr-util"
  DEPENDS "${APU_ARCHIVE}" ${apu_patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
  VERBATIM)
add_custom_target(apr-util-source DEPENDS "${APU_SOURCE}/prepared.stamp")
set_source_files_properties("${APU_SOURCE}/buckets/apr_buckets_alloc.c" PROPERTIES GENERATED TRUE)
