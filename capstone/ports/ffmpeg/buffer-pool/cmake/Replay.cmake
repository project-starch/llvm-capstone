option(FFPOOL_SUBLET "AVBufferPool and AVRefStructPool entries as Sublet lifetimes (patch 0003)" OFF)
if(FFPOOL_SUBLET AND NOT PORT_PLATFORM STREQUAL "capstone-application")
  message(FATAL_ERROR "FFPOOL_SUBLET needs the capstone-application toolchain")
endif()
if(PORT_PLATFORM STREQUAL "capstone-application")
  # FFmpeg's pools as a library for a Capstone process. Payloads and metadata come from the
  # process's own malloc (src/hosted/system-memory.c), virtual mallocng on the virtual profile,
  # as they do in FFmpeg linked against libc. FFPOOL_SUBLET adds patch 0003.
  if(FFPOOL_SUBLET)
    set(ffpool_variant protected)
  else()
    set(ffpool_variant traced)
  endif()
  ffpool_source(${ffpool_variant})
  add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime"
    "${CMAKE_BINARY_DIR}/capstone-runtime")
  add_library(ffmpeg-pool STATIC
    "${FFMPEG_${ffpool_variant}_SOURCE}/libavutil/buffer.c"
    "${FFMPEG_${ffpool_variant}_SOURCE}/libavutil/refstruct.c"
    src/shared/observe-pool-events.c
    src/hosted/system-memory.c)
  set_source_files_properties(
    "${FFMPEG_${ffpool_variant}_SOURCE}/libavutil/buffer.c"
    "${FFMPEG_${ffpool_variant}_SOURCE}/libavutil/refstruct.c" PROPERTIES GENERATED TRUE)
  target_include_directories(ffmpeg-pool PUBLIC
    "${PROJECT_SOURCE_DIR}/src/shared"
    "${FFMPEG_${ffpool_variant}_SOURCE}"
    "${PROJECT_SOURCE_DIR}/cmake/replay-config")
  target_link_libraries(ffmpeg-pool PUBLIC Capstone::Runtime)
  add_dependencies(ffmpeg-pool ffmpeg-${ffpool_variant}-source)
  add_library(FFmpeg::BufferPool ALIAS ffmpeg-pool)
  return()
endif()

if(FFPOOL_CHERI AND (NOT PORT_HOSTED OR
                      NOT CMAKE_SYSTEM_NAME STREQUAL "FreeBSD"))
  message(FATAL_ERROR "FFPOOL_CHERI requires the CheriBSD toolchain and the hosted replay entry")
endif()

ffpool_source(ported)

add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime"
  "${CMAKE_BINARY_DIR}/capstone-runtime")

add_library(replay-options INTERFACE)
target_link_libraries(replay-options INTERFACE Capstone::Runtime)
target_include_directories(replay-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared"
  "${FFMPEG_ported_SOURCE}"
  "${PROJECT_SOURCE_DIR}/cmake/replay-config")
target_compile_options(replay-options INTERFACE
  "$<$<COMPILE_LANGUAGE:C>:-ffunction-sections;-fdata-sections>")

add_library(pool-core OBJECT
  "${FFMPEG_ported_SOURCE}/libavutil/buffer.c"
  "${FFMPEG_ported_SOURCE}/libavutil/refstruct.c"
  src/shared/observe-pool-events.c
  src/shared/pool-allocator.c
  src/shared/metadata-allocator.c)
target_link_libraries(pool-core PUBLIC replay-options)
add_dependencies(pool-core ffmpeg-ported-source)

if(PORT_PLATFORM STREQUAL "capstone-domain")
  enable_language(ASM)
  target_compile_definitions(replay-options INTERFACE FFPOOL_DOMAIN)
  set(entry src/capstone-domain/entry.c)
  target_sources(pool-core PRIVATE
    src/capstone-domain/payload-capabilities.c
    src/capstone-domain/node-snapshots.c
    src/allocators/sublet/pool-leases.c)
  set(suffix .dom)
  set(domain_runtime
    "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
    "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
    "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
  add_library(domain-runtime OBJECT ${domain_runtime})
  target_link_libraries(domain-runtime PRIVATE replay-options)
endif()
if(PORT_HOSTED)
  set(entry src/native/replay/main.c)
  if(FFPOOL_CHERI)
    target_compile_definitions(replay-options INTERFACE FFPOOL_CHERI)
    target_sources(pool-core PRIVATE src/cheribsd/payload-capabilities.c)
  else()
    target_sources(pool-core PRIVATE src/native/replay/payload-pointers.c)
  endif()
endif()

add_executable(replay ${entry} src/shared/replay-engine.c)
if(PORT_HOSTED)
  add_library(ffmpeg-pool STATIC $<TARGET_OBJECTS:pool-core>)
  target_link_libraries(ffmpeg-pool PUBLIC replay-options)
  add_library(FFmpeg::BufferPool ALIAS ffmpeg-pool)
  add_executable(allocator-example examples/pool.c)
  target_link_libraries(allocator-example PRIVATE FFmpeg::BufferPool)
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(FFmpeg::BufferPool)
endif()
add_executable(pool-security ${entry} src/shared/replay-engine.c
  security-tests/shared/pool-lifetime-probes.c)
target_compile_definitions(pool-security PRIVATE FF2_SECURITY)
if(PORT_PLATFORM STREQUAL "capstone-domain")
  target_sources(pool-security PRIVATE
    security-tests/capstone/alias-scatter-register.S)
endif()
foreach(program replay pool-security)
  target_link_libraries(${program} PRIVATE pool-core)
  # Headers in the prepared source tree must exist before compiling entry points.
  add_dependencies(${program} ffmpeg-ported-source)
  if(PORT_PLATFORM STREQUAL "capstone-domain")
    target_link_libraries(${program} PRIVATE domain-runtime)
    set(link_script "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/link.ld")
    set_target_properties(${program} PROPERTIES SUFFIX "${suffix}" LINK_DEPENDS "${link_script}")
    target_link_options(${program} PRIVATE --gc-sections -T "${link_script}")
  else()
    target_link_options(${program} PRIVATE LINKER:--gc-sections)
  endif()
endforeach()
