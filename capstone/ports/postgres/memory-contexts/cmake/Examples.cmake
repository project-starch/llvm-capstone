# The same client source is linked to each manager implementation. No Sublet
# conditionals belong in client code; only the domain wrapper changes backing.
foreach(client allocset generation slab bump)
  if(PORT_PLATFORM STREQUAL "native")
    set(target client-${client}-native)
    add_executable(${target} examples/${client}.c examples/support/native.c
      src/native/replay/printf.c)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/examples")
    target_compile_definitions(${target} PRIVATE PG_CLIENT_NAME="${client}")
    target_link_libraries(${target} PRIVATE manager-native)
    target_link_options(${target} PRIVATE LINKER:--gc-sections)
    if(BUILD_TESTING)
      add_test(NAME ${target} COMMAND ${target})
      set_tests_properties(${target} PROPERTIES LABELS "native;client-examples" TIMEOUT 30)
    endif()
  else()
    foreach(mode spatial sublet)
      set(target client-${client}-${mode})
      pg_domain(${target} examples/${client}.c examples/support/domain.c src/capstone-domain/string.c)
      target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/examples")
      target_compile_definitions(${target} PRIVATE PG_CLIENT_NAME="${client}"
        "PG_DOM_FAIL_MARKER=\"__CAPSTONE_PG_CLIENT_FAILED__\\n\"")
      target_link_libraries(${target} PRIVATE manager-${mode})
      if(mode STREQUAL "sublet")
        target_compile_definitions(${target} PRIVATE PG_CLIENT_SUBLET)
        target_sources(${target} PRIVATE src/allocators/sublet/context-pools.c
          src/allocators/sublet/unsupported-allocators.c)
      else()
        target_sources(${target} PRIVATE src/allocators/spatial/backing-allocator.c)
      endif()
    endforeach()
    if(BUILD_TESTING)
      add_test(NAME qemu-client-${client} COMMAND "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/examples/run-qemu.py" ${client}
        --output "${CMAKE_BINARY_DIR}/client-results"
        --domain-build "${CMAKE_BINARY_DIR}" --linux-build "${PG_LINUX_BUILD_DIR}")
      set_tests_properties(qemu-client-${client} PROPERTIES LABELS "qemu-examples;client-examples"
        RUN_SERIAL TRUE TIMEOUT 300 ENVIRONMENT "CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT}")
    endif()
  endif()
endforeach()
if(BUILD_TESTING AND PORT_PLATFORM STREQUAL "native")
  add_test(NAME client-runner-controls COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/check-client-runner.py")
  set_tests_properties(client-runner-controls PROPERTIES LABELS "native;client-examples"
    TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()
