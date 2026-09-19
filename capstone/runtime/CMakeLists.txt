# Shared headers, domain build support, and an optional Linux fault policy.
foreach(header capstone/capability.h capstone/capability-slot.h sublet/sublet.h)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/${header}")
    message(FATAL_ERROR "Incomplete Capstone runtime: missing include/${header}")
  endif()
endforeach()
add_library(capstone-runtime INTERFACE)
add_library(Capstone::Runtime ALIAS capstone-runtime)
target_include_directories(capstone-runtime INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Domain.cmake")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(Threads REQUIRED)
  add_library(capstone-linux-fault-policy STATIC EXCLUDE_FROM_ALL linux/domain-fault.c)
  add_library(Capstone::LinuxFaultPolicy ALIAS capstone-linux-fault-policy)
  target_link_libraries(capstone-linux-fault-policy PUBLIC Capstone::Runtime Threads::Threads)
endif()
