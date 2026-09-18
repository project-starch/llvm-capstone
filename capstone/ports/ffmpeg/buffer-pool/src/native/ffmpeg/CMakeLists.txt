# Full decoders retain FFmpeg's configure/Make build. CMake tracks the pinned
# inputs and patch dependencies and keeps their artifacts outside the repo.
include(ExternalProject)
find_program(FFPOOL_MAKE NAMES gmake make REQUIRED)
set(FFPOOL_FFMPEG_JOBS 4 CACHE STRING "Parallel jobs per upstream FFmpeg build")
if(CMAKE_BINARY_DIR MATCHES "[ \t]")
  message(FATAL_ERROR "Upstream FFmpeg configure does not support source/build paths containing whitespace.")
endif()

foreach(variant stock traced)
  ffpool_source(${variant})
  set(decoder_build "${CMAKE_BINARY_DIR}/ffmpeg/${variant}")
  ExternalProject_Add(ffmpeg-${variant}
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${variant}"
    SOURCE_DIR "${FFMPEG_${variant}_SOURCE}"
    BINARY_DIR "${decoder_build}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND "<SOURCE_DIR>/configure" "--cc=${CMAKE_C_COMPILER}"
      --disable-everything --disable-autodetect
      --disable-doc --disable-network --disable-x86asm --disable-debug
      --disable-ffplay --disable-ffprobe --enable-ffmpeg
      --enable-protocol=file,pipe --enable-indev=lavfi
      --enable-filter=testsrc2,scale,hflip,format,null,anull
      --enable-encoder=mpeg4,rawvideo --enable-decoder=mpeg4,wrapped_avframe
      --enable-parser=mpeg4video --enable-muxer=matroska,framemd5,md5,null
      --enable-demuxer=matroska --enable-static --disable-shared
    BUILD_COMMAND "${FFPOOL_MAKE}" "-j${FFPOOL_FFMPEG_JOBS}" ffmpeg
    BUILD_BYPRODUCTS "${decoder_build}/ffmpeg"
    INSTALL_COMMAND ""
    LOG_CONFIGURE ON LOG_BUILD ON LOG_OUTPUT_ON_FAILURE ON)
  ExternalProject_Add_StepDependencies(ffmpeg-${variant} configure
    ffmpeg-${variant}-source "${FFMPEG_${variant}_STAMP}")
endforeach()
