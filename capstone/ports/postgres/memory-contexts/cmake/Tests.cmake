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
  add_test(NAME upstream-pin COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-pin.py")
  set_tests_properties(upstream-pin PROPERTIES LABELS native TIMEOUT 30)
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
endif()
