include("${CMAKE_CURRENT_LIST_DIR}/Workspace.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/Upstream.cmake")

set(PORT_PLATFORM native CACHE STRING "native, capstone-domain or linux-guest")
set_property(CACHE PORT_PLATFORM PROPERTY STRINGS native capstone-domain linux-guest)
file(REAL_PATH "${CMAKE_BINARY_DIR}" build_path)
cmake_path(IS_PREFIX CAPSTONE_REPO_ROOT "${build_path}" NORMALIZE in_repository)
if(in_repository)
  message(FATAL_ERROR "Keep downloaded sources and generated builds outside the repository; use a preset or -B /tmp/port-build.")
endif()

function(port_check_platform)
  if(NOT PORT_PLATFORM MATCHES "^(native|capstone-domain|linux-guest)$")
    message(FATAL_ERROR "Unknown PORT_PLATFORM: ${PORT_PLATFORM}")
  endif()
  if(NOT PORT_PLATFORM STREQUAL "native" AND NOT CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Select the ${PORT_PLATFORM} toolchain through its preset.")
  endif()
endfunction()

function(port_add_support_tests)
  add_test(NAME port-support COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PORT_SUPPORT_ROOT}/tests" -p "test_*.py")
  set_tests_properties(port-support PROPERTIES LABELS native TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()
