include_guard(GLOBAL)

get_filename_component(CAPSTONE_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." REALPATH)
set(PORT_SUPPORT_ROOT "${CAPSTONE_REPO_ROOT}/capstone/ports/common")

# Environment variables supply defaults; nonempty CMake cache settings win.
function(port_path name fallback description)
  if(DEFINED ENV{${name}} AND NOT "$ENV{${name}}" STREQUAL "")
    set(fallback "$ENV{${name}}")
  endif()
  # An unset environment reference in a preset expands to an empty cache value.
  if(DEFINED ${name} AND "${${name}}" STREQUAL "")
    unset(${name} CACHE)
  endif()
  set(${name} "${fallback}" CACHE PATH "${description}")
endfunction()

set(tmp_root /tmp/capstone)
if(DEFINED ENV{CAPSTONE_TMP_ROOT} AND NOT "$ENV{CAPSTONE_TMP_ROOT}" STREQUAL "")
  set(tmp_root "$ENV{CAPSTONE_TMP_ROOT}")
endif()

port_path(CAPSTONE_LLVM_BUILD_DIR "${CAPSTONE_REPO_ROOT}/llvm/cmake-build-debug" "Capstone compiler build")
port_path(CAPSTONE_BUILDROOT_DIR "${CAPSTONE_REPO_ROOT}/capstone/caplifive-buildroot" "Prepared guest Linux toolchain")
