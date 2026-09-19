# Compatibility entry point for existing FFmpeg campaign build commands.
include("${CMAKE_CURRENT_LIST_DIR}/../../../../common/cmake/toolchains/cheribsd.cmake")
set(FFPOOL_CHERI ON CACHE BOOL "CHERI spatial arena backend")
set(FFPOOL_BUILD_DECODERS OFF CACHE BOOL "Decode recordings on the host")
