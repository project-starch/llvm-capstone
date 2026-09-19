if(PORT_PLATFORM STREQUAL "native")
  port_add_support_tests()
  add_test(NAME native-measurement-controls
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/native/test-measurement.py")
  set_tests_properties(native-measurement-controls PROPERTIES LABELS native TIMEOUT 30)
  add_test(NAME native-measurement-resume-controls
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/native/test-measurement-resume.py")
  set_tests_properties(native-measurement-resume-controls PROPERTIES LABELS native TIMEOUT 30)
  if(FFPOOL_BUILD_DECODERS)
    add_test(NAME native-workload-and-controls
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/native/test-workload.py"
        "${CMAKE_BINARY_DIR}")
    set_tests_properties(native-workload-and-controls PROPERTIES
      LABELS native TIMEOUT 180)
  endif()
else()
  # Build/test native first to produce the decoder-checked recording, then build
  # linux-guest. These tests load the current build's domains into guest Linux.
  set(test_environment
    "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}"
    "FFPOOL_WORK=${FFPOOL_WORK}"
    "FFPOOL_DOMAIN_BUILD_DIR=${CMAKE_BINARY_DIR}"
    "FFPOOL_LINUX_BUILD_DIR=${FFPOOL_LINUX_BUILD_DIR}"
    "PYTHON=${Python3_EXECUTABLE}")
  foreach(mode spatial backing sublet)
    add_test(NAME qemu-replay-${mode}
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/qemu/test-replay.py"
        "${FFPOOL_RECORDING_DIR}" "${CMAKE_BINARY_DIR}/test-results" "${mode}")
    set_tests_properties(qemu-replay-${mode} PROPERTIES
      ENVIRONMENT "${test_environment}" LABELS qemu-replay
      RUN_SERIAL TRUE TIMEOUT 600)
  endforeach()
  add_test(NAME qemu-pool-lifetimes
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/security-tests/qemu/test-suite.py"
      "${CMAKE_BINARY_DIR}/test-results")
  set_tests_properties(qemu-pool-lifetimes PROPERTIES
    ENVIRONMENT "${test_environment}" LABELS qemu-security
    RUN_SERIAL TRUE TIMEOUT 7200)
endif()
