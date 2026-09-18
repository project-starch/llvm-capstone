port_read_upstream()
set(FFMPEG_VERSION "${UPSTREAM_version}")
set(FFMPEG_SHA256 "${UPSTREAM_sha256}")
set(FFPOOL_ARCHIVE "${FFPOOL_WORK}/download/ffmpeg-${FFMPEG_VERSION}.tar.xz"
  CACHE FILEPATH "Pinned FFmpeg archive; prepopulate for offline builds")
port_download("${FFPOOL_ARCHIVE}")

function(ffpool_source variant)
  set(source_dir "${CMAKE_BINARY_DIR}/sources/ffmpeg-${variant}")
  set(stamp "${source_dir}/prepared.stamp")
  set(dependencies "${FFPOOL_ARCHIVE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py")
  if(NOT variant STREQUAL "stock")
    list(APPEND dependencies
      "${PROJECT_SOURCE_DIR}/host/instrument-pools.py"
      "${PROJECT_SOURCE_DIR}/src/native/ffmpeg/record-pool-events.c"
      "${PROJECT_SOURCE_DIR}/src/shared/observe-pool-events.c"
      "${PROJECT_SOURCE_DIR}/src/shared/trace.h")
  endif()
  if(variant STREQUAL "ported")
    list(APPEND dependencies "${PROJECT_SOURCE_DIR}/patches/apply-pool-port.py")
  endif()
  add_custom_command(OUTPUT "${stamp}"
    BYPRODUCTS "${source_dir}/libavutil/buffer.c" "${source_dir}/libavutil/refstruct.c"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py"
      "${FFPOOL_ARCHIVE}" "${FFMPEG_SHA256}" "${FFMPEG_VERSION}" "${source_dir}" "${variant}"
    DEPENDS ${dependencies}
    COMMENT "Preparing SHA-verified FFmpeg ${variant} sources"
    VERBATIM)
  add_custom_target(ffmpeg-${variant}-source DEPENDS "${stamp}")
  set(FFMPEG_${variant}_SOURCE "${source_dir}" PARENT_SCOPE)
  set(FFMPEG_${variant}_STAMP "${stamp}" PARENT_SCOPE)
endfunction()
