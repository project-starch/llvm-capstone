option(WG_BUILD_RECORDER "Build stock and instrumented native Whisper executables" OFF)
if(NOT WG_BUILD_RECORDER)
  return()
endif()
include(ExternalProject)
wg_source(recorder)
foreach(arm stock recorded)
  set(variant reference)
  set(extra)
  if(arm STREQUAL "recorded")
    set(variant recorder)
    list(APPEND extra "-DCMAKE_PROJECT_whisper.cpp_INCLUDE=${PROJECT_SOURCE_DIR}/cmake/RecorderInject.cmake"
      "-DWG_PORT_ROOT=${PROJECT_SOURCE_DIR}")
  endif()
  ExternalProject_Add(whisper-${arm}
    SOURCE_DIR "${CMAKE_BINARY_DIR}/source-${variant}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/whisper-${arm}"
    DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
    DEPENDS source-${variant}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
      -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=OFF
      -DWHISPER_CURL=OFF -DWHISPER_SDL2=OFF ${extra}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target whisper-cli --parallel 4
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/whisper-${arm}/bin/whisper-cli"
      "${CMAKE_BINARY_DIR}/whisper-${arm}/ggml/src/libggml-base.a")
endforeach()
add_executable(replay-full-reference src/native/main.c src/native/full-reference.c
  src/native/regions.c src/shared/backing.c src/shared/replay.c)
add_dependencies(replay-full-reference whisper-stock)
target_include_directories(replay-full-reference PRIVATE src/shared "${CMAKE_BINARY_DIR}/source-reference/ggml/include")
target_link_libraries(replay-full-reference PRIVATE "${CMAKE_BINARY_DIR}/whisper-stock/ggml/src/libggml-base.a" stdc++ m pthread)
if(BUILD_TESTING)
  add_test(NAME full-library-reference COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/native/test-replay.py" "$<TARGET_FILE:replay>"
    "$<TARGET_FILE:replay-full-reference>" "${CMAKE_BINARY_DIR}/tests-full-reference")
endif()
