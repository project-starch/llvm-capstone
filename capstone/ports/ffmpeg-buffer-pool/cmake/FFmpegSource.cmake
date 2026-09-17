# This is the only release pin. Source copies belong to individual build trees.
set(FFMPEG_VERSION 9.0.1)
set(FFMPEG_SHA256 cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635)
set(FFPOOL_ARCHIVE "${FFPOOL_WORK}/download/ffmpeg-${FFMPEG_VERSION}.tar.xz"
  CACHE FILEPATH "Pinned FFmpeg archive; prepopulate for offline builds")

function(ffpool_download)
  get_filename_component(download_dir "${FFPOOL_ARCHIVE}" DIRECTORY)
  file(MAKE_DIRECTORY "${download_dir}")
  # Independent native/domain configure processes may share the archive cache.
  file(LOCK "${FFPOOL_ARCHIVE}.lock" GUARD FUNCTION TIMEOUT 120)
  if(NOT EXISTS "${FFPOOL_ARCHIVE}")
    file(DOWNLOAD "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
      "${FFPOOL_ARCHIVE}.part" EXPECTED_HASH "SHA256=${FFMPEG_SHA256}"
      TLS_VERIFY ON SHOW_PROGRESS STATUS download_status)
    list(GET download_status 0 download_code)
    if(NOT download_code EQUAL 0)
      file(REMOVE "${FFPOOL_ARCHIVE}.part")
      message(FATAL_ERROR "FFmpeg download failed: ${download_status}")
    endif()
    file(RENAME "${FFPOOL_ARCHIVE}.part" "${FFPOOL_ARCHIVE}")
  endif()
  file(SHA256 "${FFPOOL_ARCHIVE}" actual_sha)
  if(NOT actual_sha STREQUAL FFMPEG_SHA256)
    message(FATAL_ERROR "FFmpeg archive SHA256 mismatch: ${FFPOOL_ARCHIVE}")
  endif()
endfunction()
ffpool_download()

function(ffpool_source variant)
  set(source_dir "${CMAKE_BINARY_DIR}/sources/ffmpeg-${variant}")
  set(stamp "${source_dir}/prepared.stamp")
  set(dependencies "${FFPOOL_ARCHIVE}" "${PROJECT_SOURCE_DIR}/cmake/prepare-source.py")
  if(NOT variant STREQUAL "stock")
    list(APPEND dependencies
      "${PROJECT_SOURCE_DIR}/native/ffmpeg/instrument-pools.py"
      "${PROJECT_SOURCE_DIR}/native/ffmpeg/record-pool-events.c"
      "${PROJECT_SOURCE_DIR}/shared/observe-pool-events.c"
      "${PROJECT_SOURCE_DIR}/shared/trace.h")
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
