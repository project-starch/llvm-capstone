# The same client source is linked to each manager implementation. No Sublet
# conditionals belong in client code.
foreach(client allocset generation slab bump)
  set(target client-${client}-${PORT_PLATFORM})
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
endforeach()
