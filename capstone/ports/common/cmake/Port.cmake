include("${CMAKE_CURRENT_LIST_DIR}/Workspace.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/Upstream.cmake")

set(PORT_PLATFORM native CACHE STRING
    "native, cheribsd, capstone-application, capstone-domain or linux-guest")
set_property(CACHE PORT_PLATFORM PROPERTY STRINGS
             native cheribsd capstone-application capstone-domain linux-guest)
file(REAL_PATH "${CMAKE_BINARY_DIR}" build_path)
cmake_path(IS_PREFIX CAPSTONE_REPO_ROOT "${build_path}" NORMALIZE in_repository)
if(in_repository)
  message(FATAL_ERROR "Keep downloaded sources and generated builds outside the repository; use a preset or -B /tmp/port-build.")
endif()

function(port_check_platform)
  if(NOT PORT_PLATFORM MATCHES "^(native|cheribsd|capstone-application|capstone-domain|linux-guest)$")
    message(FATAL_ERROR "Unknown PORT_PLATFORM: ${PORT_PLATFORM}")
  endif()
  if(NOT PORT_PLATFORM STREQUAL "native" AND NOT CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Select the ${PORT_PLATFORM} toolchain through its preset.")
  endif()
  # capstone-application is HOSTED: it has a libc and an ordinary main(), so a
  # component takes the same sources as native and cheribsd. What differs is the
  # pointer representation underneath, which is the toolchain's business and not
  # the component's. capstone-domain is the freestanding one and stays apart.
  if(PORT_PLATFORM MATCHES "^(native|cheribsd|capstone-application)$")
    set(PORT_HOSTED ON PARENT_SCOPE)
  else()
    set(PORT_HOSTED OFF PARENT_SCOPE)
  endif()
endfunction()

function(port_add_support_tests)
  add_test(NAME port-support COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PORT_SUPPORT_ROOT}/tests" -p "test_*.py")
  set_tests_properties(port-support PROPERTIES LABELS native TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()
