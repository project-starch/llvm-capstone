# Host tests of the kept runtime code that the libc links: the launch header
# reader (common/launch.c) and the delegation wire (common/delegate.c, spawn.c,
# msghdr.c), plus two port helpers. Build natively and run ctest:
#   cmake -S capstone/runtime/tests -B build && cmake --build build && ctest --test-dir build
# They moved here from runtime/exec, which went with the physical launcher on
# 2026-11-14.
cmake_minimum_required(VERSION 3.25)
project(capstone_runtime_host_tests LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
include(CTest)
get_filename_component(runtime "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(capstone "${runtime}/.." ABSOLUTE)

add_library(launch STATIC "${runtime}/common/launch.c")
target_compile_definitions(launch PRIVATE _POSIX_C_SOURCE=200809L)
target_include_directories(launch PUBLIC "${runtime}/include")
add_library(delegate STATIC "${runtime}/common/delegate.c" "${runtime}/common/spawn.c"
  "${runtime}/common/msghdr.c")
target_include_directories(delegate PUBLIC "${runtime}/include")

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(launch-test delegation/launch-test.c)
  target_link_libraries(launch-test PRIVATE launch)
  add_test(NAME launch-wire-contract COMMAND launch-test)
  foreach(name delegate spawn msghdr)
    add_executable(${name}-test delegation/${name}-test.c)
    target_link_libraries(${name}-test PRIVATE delegate)
    add_test(NAME ${name}-wire-contract COMMAND ${name}-test)
  endforeach()
  add_executable(mruby-spawn-test "${capstone}/ports/mruby/app/spawn-shell.c"
    "${capstone}/ports/mruby/app/spawn-shell-test.c")
  add_test(NAME mruby-spawn-shell COMMAND mruby-spawn-test)
  add_test(NAME application-safety-classifier
    COMMAND "${Python3_EXECUTABLE}" "${capstone}/ports/common/application/check-safety-test.py")
  foreach(test launch-test delegate-test spawn-test msghdr-test mruby-spawn-test)
    target_compile_options(${test} PRIVATE -UNDEBUG)
  endforeach()
  set_tests_properties(launch-wire-contract delegate-wire-contract spawn-wire-contract
    msghdr-wire-contract mruby-spawn-shell PROPERTIES TIMEOUT 15)
endif()
