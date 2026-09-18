set(fixture "${CMAKE_BINARY_DIR}/tests/fixture.a11")
add_custom_command(OUTPUT "${fixture}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/tests"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/make-fixture-trace.py" "${fixture}"
  DEPENDS tests/make-fixture-trace.py shared/a11trace.h VERBATIM)
add_custom_target(replay-fixture ALL DEPENDS "${fixture}")
if(PG_PLATFORM STREQUAL "native")
  add_test(NAME memory-profile-accounting COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/memory-profile/tests" -p "test_*.py")
  set_tests_properties(memory-profile-accounting PROPERTIES LABELS native TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME native-replay-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-native.py" "$<TARGET_FILE:replay>" "${fixture}")
  add_test(NAME build-guards COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-build-guards.py" "${PROJECT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}" "${PG_ARCHIVE}" "${PG_VERSION}" "${PG_SHA256}"
    "${PG_HOST_CC}" "${PG_MAKE}")
  set_tests_properties(native-replay-controls build-guards PROPERTIES LABELS native TIMEOUT 60)
else()
  foreach(program replay-spatial replay-sublet subpool-lifetimes context-hierarchy)
    add_test(NAME qemu-${program} COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/capstone/run-qemu.py" "${CMAKE_BINARY_DIR}" "${PG_LINUX_BUILD_DIR}"
      "${program}" "${fixture}")
    if(program MATCHES "^replay-")
      set(label qemu-replay)
    else()
      set(label qemu-security)
    endif()
    set_tests_properties(qemu-${program} PROPERTIES LABELS "${label}" RUN_SERIAL TRUE TIMEOUT 600
      ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
  endforeach()
endif()
