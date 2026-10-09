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
# The virtual application platform is HOSTED -- ordinary main(), a libc -- and
# carries capability pointers, so it takes the SAME payload backend and lease
# policy as the freestanding domain and differs only in where the payload comes
# from: a linear block borrowed from the system allocator rather than a grant.
# FFPOOL_DOMAIN is what the probes and the replay engine key the capability path
# on, so it is defined here too; it does not imply freestanding anywhere.
if(PORT_PLATFORM STREQUAL "capstone-application")
  target_compile_definitions(replay-options INTERFACE FFPOOL_DOMAIN FFPOOL_BORROW_LINEAR)
  target_sources(pool-core PRIVATE
    src/capstone-domain/payload-capabilities.c
    src/capstone-domain/node-snapshots.c
    src/allocators/sublet/pool-leases.c)
endif()
if(PORT_HOSTED)
  set(entry src/native/replay/main.c)
  if(PORT_PLATFORM STREQUAL "capstone-application")
    # the capability backend above is already selected
  elseif(FFPOOL_CHERI)
    target_compile_definitions(replay-options INTERFACE FFPOOL_CHERI)
    option(FFPOOL_POISONCAP "Use experimental PoisonCap with sweep before pool reuse" OFF)
    if(FFPOOL_POISONCAP)
      target_compile_definitions(replay-options INTERFACE FFPOOL_POISONCAP)
      target_sources(pool-core PRIVATE src/cheribsd/poisoncap-payload.c)
      add_executable(poisoncap-probe security-tests/cheribsd/poisoncap-probe.c)
      target_link_libraries(poisoncap-probe PRIVATE replay-options)
      # The published revoke.h contains a GNU 'asm' inline function.
      set_target_properties(pool-core poisoncap-probe PROPERTIES C_EXTENSIONS ON)
    else()
      target_sources(pool-core PRIVATE src/cheribsd/payload-capabilities.c)
    endif()
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
# The fixture that DEFINES ff2_alias_scatter_register_span and the two register
# probes is Capstone assembly with no domain scaffolding in it -- plain
# instructions over a0..a7 -- so it belongs to every Capstone platform, not to
# the freestanding one. Keyed on `capstone-domain` alone it was missing on
# capstone-application and pool-security could not link there at all.
if(PORT_PLATFORM MATCHES "^capstone-")
  # CMake SILENTLY DROPS a .S source when ASM is not an enabled language: no
  # warning, no build rule, just three undefined symbols at link. The domain
  # branch near the top of this file enables it, and the hosted Capstone
  # platform needs it for the very same fixture; calling it again where it is
  # already on is harmless.
  enable_language(ASM)
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
