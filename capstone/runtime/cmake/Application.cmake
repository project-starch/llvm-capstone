include_guard(GLOBAL)

# A normal main(argc, argv), linked against the shared musl application ABI.
# The caller selects the existing capstone-domain toolchain and musl headers.
function(capstone_configure_application target)
  cmake_parse_arguments(PARSE_ARGV 1 app "" "DATA_BYTES;STACK_BYTES;ARENA_BYTES" "")
  if(app_UNPARSED_ARGUMENTS OR app_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid capstone_configure_application arguments")
  endif()
  foreach(size DATA_BYTES STACK_BYTES ARENA_BYTES)
    if(NOT "${app_${size}}" MATCHES "^[0-9]+$")
      message(FATAL_ERROR "capstone_configure_application requires numeric ${size}")
    endif()
  endforeach()
  if(app_STACK_BYTES LESS 4096 OR app_DATA_BYTES LESS app_STACK_BYTES)
    message(FATAL_ERROR "Application data must cover its stack (at least 4096 bytes)")
  endif()
  if(NOT PORT_HEADER_PROVIDER STREQUAL "musl" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "capstone64")
    message(FATAL_ERROR "Applications require the capstone-domain toolchain with musl headers")
  endif()
  if(NOT EXISTS "${CAPSTONE_MUSL_ARCHIVE}")
    message(FATAL_ERROR "Set CAPSTONE_MUSL_ARCHIVE to the built Capstone musl archive")
  endif()
  get_target_property(configured ${target} CAPSTONE_DOMAIN_CONFIGURED)
  if(configured)
    message(FATAL_ERROR "Domain ${target} is already configured")
  endif()
  set_target_properties(${target} PROPERTIES CAPSTONE_DOMAIN_CONFIGURED TRUE)
  get_filename_component(capstone "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../.." ABSOLUTE)
  set(musl "${capstone}/ports/musl-capstone/runtime")
  if(NOT TARGET capstone-application-core)
    add_library(capstone-application-core OBJECT
      "${musl}/start-musl.S" "${musl}/set_thread_area.S" "${musl}/setjmp.S"
      "${musl}/hostcall.c" "${musl}/tls.c" "${capstone}/runtime/common/launch.c")
    file(STRINGS "${musl}/libc_overrides.list" overrides)
    foreach(source IN LISTS overrides)
      target_sources(capstone-application-core PRIVATE "${musl}/${source}.c")
    endforeach()
    target_include_directories(capstone-application-core PRIVATE
      "${PORT_MUSL_ROOT}/src/include" "${PORT_MUSL_ROOT}/src/internal"
      "${PORT_MUSL_ROOT}/obj/src/internal" "${PORT_MUSL_ROOT}/src/multibyte")
    target_compile_definitions(capstone-application-core PRIVATE
      _XOPEN_SOURCE=700 CAPSTONE_APPLICATION_RUNTIME=1 CAPSTONE_DOMAIN_FAULT_RECOVERY=1)
    target_compile_options(capstone-application-core PRIVATE
      -ffunction-sections -fdata-sections -fno-jump-tables
      "$<$<COMPILE_LANGUAGE:C>:-Wno-int-conversion>")
    target_link_libraries(capstone-application-core PUBLIC Capstone::Runtime)
    file(STRINGS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/softfloat.list" builtins)
    add_library(capstone-application-builtins STATIC)
    foreach(source IN LISTS builtins)
      target_sources(capstone-application-builtins PRIVATE
        "${capstone}/../compiler-rt/lib/builtins/${source}.c")
    endforeach()
    target_compile_options(capstone-application-builtins PRIVATE
      -ffunction-sections -fdata-sections -fno-jump-tables)
  endif()
  math(EXPR data_bytes "${app_DATA_BYTES} + 256")
  target_sources(${target} PRIVATE "${capstone}/runtime/domain/application.c"
    "${musl}/level0.c" "${capstone}/runtime/domain/domreq.S"
    "${capstone}/runtime/domain/gct-section-end.S")
  target_compile_definitions(${target} PRIVATE _XOPEN_SOURCE=700
    CAPSTONE_DOMREQ_DATA=${data_bytes} CAPSTONE_DOMREQ_STACK=${app_STACK_BYTES}
    CAPSTONE_LEVEL0_ARENA_BYTES=${app_ARENA_BYTES})
  target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections -fno-jump-tables)
  target_link_libraries(${target} PRIVATE capstone-application-core
    "${CAPSTONE_MUSL_ARCHIVE}" capstone-application-builtins)
  set(script "${capstone}/my_first_domain/link.ld")
  target_link_options(${target} PRIVATE --gc-sections -T "${script}")
  set_target_properties(${target} PROPERTIES SUFFIX .dom LINK_DEPENDS "${script}")
endfunction()
