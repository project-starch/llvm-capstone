port_read_upstream()
port_path(WG_ARCHIVE "${WG_WORK}/download/whisper.cpp-${UPSTREAM_version}.tar.gz" "Pinned release archive")
port_download("${WG_ARCHIVE}")
find_program(PATCH_TOOL patch REQUIRED)
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch")
function(wg_source variant)
  set(source "${CMAKE_BINARY_DIR}/source-${variant}")
  add_custom_command(OUTPUT "${source}/prepared.stamp"
    BYPRODUCTS "${source}/ggml/src/ggml.c" "${source}/ggml/include/ggml.h"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
      "${WG_ARCHIVE}" "${UPSTREAM_sha256}" "${source}" --variant "${variant}"
    DEPENDS "${WG_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    VERBATIM)
  add_custom_target(source-${variant} DEPENDS "${source}/prepared.stamp")
  set_source_files_properties("${source}/ggml/src/ggml.c" PROPERTIES GENERATED TRUE)
endfunction()
