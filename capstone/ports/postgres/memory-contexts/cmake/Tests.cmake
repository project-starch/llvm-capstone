set(fixture "${CMAKE_BINARY_DIR}/tests/fixture.a11")
add_custom_command(OUTPUT "${fixture}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/tests"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/make-fixture-trace.py" "${fixture}"
  DEPENDS tests/make-fixture-trace.py src/shared/a11trace.h VERBATIM)
add_custom_target(replay-fixture ALL DEPENDS "${fixture}")
set(context_fixture "${CMAKE_BINARY_DIR}/tests/contexts.a11")
add_custom_command(OUTPUT "${context_fixture}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/tests"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/make-contexts-trace.py" "${context_fixture}"
  DEPENDS tests/make-contexts-trace.py tests/make-fixture-trace.py src/shared/a11trace.h VERBATIM)
add_custom_target(context-replay-fixture ALL DEPENDS "${context_fixture}")
if(PORT_PLATFORM STREQUAL "native")
  port_add_support_tests()
  add_test(NAME fault-isolation-verdict-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-fault-isolation.py")
  set_tests_properties(fault-isolation-verdict-controls PROPERTIES LABELS native TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME context-verdict-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-context-verdicts.py")
  set_tests_properties(context-verdict-controls PROPERTIES LABELS native TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME upstream-pin COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-pin.py")
  set_tests_properties(upstream-pin PROPERTIES LABELS native TIMEOUT 30)
  add_test(NAME memory-profile-accounting COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/tests/memory" -p "test_*.py")
  set_tests_properties(memory-profile-accounting PROPERTIES LABELS native TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME native-replay-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-native.py" "$<TARGET_FILE:replay>" "${fixture}")
  add_test(NAME native-context-replay-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-native.py" "$<TARGET_FILE:replay>" "${context_fixture}")
  set_tests_properties(native-context-replay-controls PROPERTIES LABELS native TIMEOUT 60)
  add_test(NAME build-guards COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-build-guards.py" "${PROJECT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}" "${PG_ARCHIVE}" "${PG_VERSION}" "${PG_SHA256}"
    "${PG_HOST_CC}" "${PG_MAKE}")
  set_tests_properties(native-replay-controls build-guards PROPERTIES LABELS native TIMEOUT 60)
else()
  if(PG_DOMAIN_FAULT_RECOVERY)
    add_test(NAME qemu-fault-isolation COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/security-tests/run-fault-isolation.py"
      --domain-build "${CMAKE_BINARY_DIR}" --linux-build "${PG_LINUX_BUILD_DIR}"
      --output "${CMAKE_BINARY_DIR}/fault-results")
    set_tests_properties(qemu-fault-isolation PROPERTIES LABELS "qemu-fault-isolation"
      RUN_SERIAL TRUE TIMEOUT 600 ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
  endif()
  foreach(mode spatial sublet)
    add_test(NAME qemu-context-replay-${mode} COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/host/run-qemu.py" "${context_fixture}" "${CMAKE_BINARY_DIR}/test-results"
      --domain-build "${CMAKE_BINARY_DIR}" --linux-build "${PG_LINUX_BUILD_DIR}"
      --protection "${mode}")
    set_tests_properties(qemu-context-replay-${mode} PROPERTIES LABELS "qemu-replay"
      RUN_SERIAL TRUE TIMEOUT 600 ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
  endforeach()
  add_test(NAME qemu-context-allocators COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/security-tests/run-contexts.py"
    "${CMAKE_BINARY_DIR}/context-results"
    --domain-build "${CMAKE_BINARY_DIR}" --linux-build "${PG_LINUX_BUILD_DIR}")
  set_tests_properties(qemu-context-allocators PROPERTIES LABELS "qemu-security"
    RUN_SERIAL TRUE TIMEOUT 1200 ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
  foreach(program replay-spatial replay-sublet subpool-lifetimes context-hierarchy)
    if(program STREQUAL "replay-spatial")
      set(run_options --protection spatial)
    elseif(program STREQUAL "replay-sublet")
      set(run_options --protection sublet)
    else()
      set(run_options --protection sublet --program "${program}")
    endif()
    add_test(NAME qemu-${program} COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/host/run-qemu.py" "${fixture}" "${CMAKE_BINARY_DIR}/test-results"
      --domain-build "${CMAKE_BINARY_DIR}" --linux-build "${PG_LINUX_BUILD_DIR}" ${run_options})
    if(program MATCHES "^replay-")
      set(label qemu-replay)
    else()
      set(label qemu-security)
    endif()
    set_tests_properties(qemu-${program} PROPERTIES LABELS "${label}" RUN_SERIAL TRUE TIMEOUT 600
      ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
  endforeach()
endif()
