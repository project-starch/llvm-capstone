include_guard(GLOBAL)

# A normal main(argc, argv), linked against the shared musl application ABI:
# the delegated runtime (ABI v2), the only one. The caller selects the existing
# capstone-domain toolchain and musl headers.

function(capstone_configure_application target)
  cmake_parse_arguments(PARSE_ARGV 1 app "" "DATA_BYTES;STACK_BYTES;ARENA_BYTES;HEAP;HEAP_LOG;EXCHANGE_BYTES;GRANT_BYTES;CONTEXT_BYTES;CONTEXTS" "")
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
  # Contexts of one application share lock words (docs/plans/delegation-threads.md):
  # the runtime and the application are built with the atomic ISA, as musl is.
  if(NOT PORT_C11_ATOMICS)
    message(FATAL_ERROR "Applications require PORT_C11_ATOMICS=ON (the A extension)")
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
      "${musl}/hostcall.c" "${musl}/tls.c" "${musl}/atomic_libcalls.c" "${musl}/context.c"
      "${musl}/lock.c"
      "${capstone}/runtime/common/launch.c")
    file(STRINGS "${musl}/libc_overrides.list" overrides)
    foreach(source IN LISTS overrides)
      target_sources(capstone-application-core PRIVATE "${musl}/${source}.c")
    endforeach()
    target_include_directories(capstone-application-core PRIVATE
      "${PORT_MUSL_ROOT}/src/include" "${PORT_MUSL_ROOT}/src/internal"
      "${PORT_MUSL_ROOT}/obj/src/internal" "${PORT_MUSL_ROOT}/src/multibyte")
    target_compile_definitions(capstone-application-core PRIVATE _XOPEN_SOURCE=700)
    target_sources(capstone-application-core PRIVATE
        "${musl}/delegate.c" "${musl}/posix_spawn_delegate.c" "${musl}/signals.c" "${musl}/altstack.S"
        "${capstone}/runtime/common/delegate.c" "${capstone}/runtime/common/spawn.c")
    set_source_files_properties("${musl}/posix_spawn_delegate.c" PROPERTIES
        INCLUDE_DIRECTORIES "${PORT_MUSL_ROOT}/src/process")
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
  if(NOT app_HEAP)
    set(app_HEAP level0)
  endif()
  set(heap_bytes 0)
  if(app_HEAP STREQUAL "sublet")
    if(NOT app_HEAP_LOG MATCHES "^[0-9]+$" OR app_HEAP_LOG LESS 12 OR app_HEAP_LOG GREATER 27)
      message(FATAL_ERROR "Sublet applications require HEAP_LOG between 12 and 27")
    endif()
    set(heap "${musl}/sublet_heap.c")
    math(EXPR heap_bytes "2 * (1 << ${app_HEAP_LOG})")
    target_compile_definitions(${target} PRIVATE CAPSTONE_SUBLET_HEAP_LOG=${app_HEAP_LOG})
    target_include_directories(${target} PRIVATE "${capstone}/sublet")
  elseif(app_HEAP STREQUAL "level0")
    set(heap "${musl}/level0.c")
  else()
    message(FATAL_ERROR "Unknown application heap: ${app_HEAP}")
  endif()
  if(app_GRANT_BYTES)
    if(NOT app_GRANT_BYTES MATCHES "^[0-9]+$" OR app_GRANT_BYTES LESS heap_bytes OR
       app_GRANT_BYTES GREATER 268435456)
      message(FATAL_ERROR "GRANT_BYTES must cover the heap and be at most 256 MiB")
    endif()
    set(heap_bytes ${app_GRANT_BYTES})
  endif()
  target_compile_definitions(${target} PRIVATE CAPSTONE_APPLICATION_HEAP_BYTES=${heap_bytes})
  if(NOT app_EXCHANGE_BYTES)
    set(app_EXCHANGE_BYTES 262144)
  endif()
  if(NOT app_EXCHANGE_BYTES MATCHES "^[0-9]+$" OR app_EXCHANGE_BYTES LESS 4096 OR
     app_EXCHANGE_BYTES GREATER 1073741824)
    message(FATAL_ERROR "EXCHANGE_BYTES must be between 4096 and 1073741824")
  endif()
  math(EXPR exchange_rest "${app_EXCHANGE_BYTES} % 4096")
  if(NOT exchange_rest EQUAL 0)
    message(FATAL_ERROR "EXCHANGE_BYTES must be a multiple of 4096")
  endif()
  # CONTEXTS: how many contexts besides the first may run at once with a
  # transport of their own (docs/plans/delegation-threads.md); the launcher
  # grants 1 + CONTEXTS entry blocks and exchange regions. At most 15: the
  # monitor lends each application 16 invocation descriptors.
  if(NOT app_CONTEXTS)
    set(app_CONTEXTS 0)
  endif()
  if(NOT app_CONTEXTS MATCHES "^([0-9]|1[0-5])$")
    message(FATAL_ERROR "CONTEXTS must be between 0 and 15")
  endif()
  math(EXPR exchange_total "(1 + ${app_CONTEXTS}) * ${app_EXCHANGE_BYTES}")
  if(exchange_total GREATER 1073741824)
    message(FATAL_ERROR "(1 + CONTEXTS) * EXCHANGE_BYTES must be at most 1073741824")
  endif()
  target_compile_definitions(${target} PRIVATE
    CAPSTONE_APPLICATION_EXCHANGE_BYTES=${app_EXCHANGE_BYTES}
    CAPSTONE_APPLICATION_CONTEXTS=${app_CONTEXTS})
  # CONTEXT_BYTES: the linear arena _start splits off the data region for
  # minted contexts (docs/plans/delegation-threads.md). 0 leaves the data
  # region as it was.
  if(NOT app_CONTEXT_BYTES)
    set(app_CONTEXT_BYTES 0)
  endif()
  if(NOT app_CONTEXT_BYTES MATCHES "^[0-9]+$")
    message(FATAL_ERROR "CONTEXT_BYTES must be numeric")
  endif()
  math(EXPR context_rest "${app_CONTEXT_BYTES} % 4096")
  if(NOT context_rest EQUAL 0)
    message(FATAL_ERROR "CONTEXT_BYTES must be a multiple of 4096")
  endif()
  math(EXPR data_bytes "${app_DATA_BYTES} + 256 + ${app_CONTEXT_BYTES}")
  target_sources(${target} PRIVATE "${capstone}/runtime/domain/application.c"
    "${heap}" "${capstone}/runtime/domain/domreq.S"
    "${capstone}/runtime/domain/gct-section-end.S")
  target_compile_definitions(${target} PRIVATE _XOPEN_SOURCE=700
    CAPSTONE_DOMREQ_DATA=${data_bytes} CAPSTONE_DOMREQ_STACK=${app_STACK_BYTES}
    CAPSTONE_CONTEXT_ARENA_BYTES=${app_CONTEXT_BYTES}
    CAPSTONE_LEVEL0_ARENA_BYTES=${app_ARENA_BYTES})
  target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections -fno-jump-tables)
  target_sources(${target} PRIVATE $<TARGET_OBJECTS:capstone-application-core>)
  target_link_libraries(${target} PRIVATE Capstone::Runtime
    "${CAPSTONE_MUSL_ARCHIVE}" capstone-application-builtins)
  set(script "${capstone}/my_first_domain/link.ld")
  target_link_options(${target} PRIVATE --gc-sections -T "${script}")
  get_target_property(kind ${target} TYPE)
  if(kind STREQUAL "EXECUTABLE")
    set_target_properties(${target} PROPERTIES SUFFIX .dom LINK_DEPENDS "${script}")
  endif()
endfunction()
