set(PYMALLOC_RECORD_PYTHON "" CACHE FILEPATH "Optional CPython 3.13.7 executable for workload recording")
if(PYMALLOC_RECORD_PYTHON)
  execute_process(COMMAND "${PYMALLOC_RECORD_PYTHON}" -c
    "import json,sys,sysconfig; assert sys.version_info[:3] == (3,13,7); assert not sysconfig.get_config_var('Py_GIL_DISABLED'); print(json.dumps([sysconfig.get_path('include'),sysconfig.get_config_var('EXT_SUFFIX')]))"
    OUTPUT_VARIABLE python_config OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(JSON python_include GET "${python_config}" 0)
  string(JSON python_suffix GET "${python_config}" 1)
  add_library(pymrecord MODULE src/native/record.c)
  target_include_directories(pymrecord PRIVATE "${python_include}" "${PROJECT_SOURCE_DIR}/src/shared")
  target_compile_options(pymrecord PRIVATE -Wall -Wextra)
  set_target_properties(pymrecord PROPERTIES PREFIX "" OUTPUT_NAME _pymrecord
    SUFFIX "${python_suffix}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
  if(BUILD_TESTING)
    add_test(NAME recorded-workload COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/native/test-recording.py"
      "${PYMALLOC_RECORD_PYTHON}" "${CMAKE_BINARY_DIR}/python"
      "$<TARGET_FILE:replay>" "$<TARGET_FILE:replay-reference>" "${CMAKE_BINARY_DIR}/test-results")
  endif()
endif()
