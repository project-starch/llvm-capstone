port_read_upstream()
port_path(WM_ARCHIVE "${WM_WORK}/download/wireshark-${UPSTREAM_version}.tar.xz" "Pinned release archive")
port_download("${WM_ARCHIVE}")
find_program(PATCH_TOOL patch REQUIRED)
file(GLOB patch_inputs CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/*.patch")
# The six translation units that make up wmem's core and its four allocators.
set(WM_UPSTREAM_UNITS wmem_core.c wmem_user_cb.c wmem_allocator_simple.c
  wmem_allocator_block.c wmem_allocator_block_fast.c wmem_allocator_strict.c)
function(wm_source variant)
  set(source "${CMAKE_BINARY_DIR}/source-${variant}")
  set(products)
  foreach(unit ${WM_UPSTREAM_UNITS})
    list(APPEND products "${source}/wsutil/wmem/${unit}")
  endforeach()
  add_custom_command(OUTPUT "${source}/prepared.stamp" BYPRODUCTS ${products}
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
      "${WM_ARCHIVE}" "${UPSTREAM_sha256}" "${source}" --variant "${variant}"
    DEPENDS "${WM_ARCHIVE}" ${patch_inputs} "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
    VERBATIM)
  add_custom_target(source-${variant} DEPENDS "${source}/prepared.stamp")
  set_source_files_properties(${products} PROPERTIES GENERATED TRUE)
endfunction()
function(wm_upstream_units out variant)
  set(units)
  foreach(unit ${WM_UPSTREAM_UNITS})
    list(APPEND units "${CMAKE_BINARY_DIR}/source-${variant}/wsutil/wmem/${unit}")
  endforeach()
  set(${out} ${units} PARENT_SCOPE)
endfunction()
