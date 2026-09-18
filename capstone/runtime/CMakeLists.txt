# Header-only capability operations and Sublet lifetime helpers.
foreach(header capstone/capability.h capstone/capability-slot.h sublet/sublet.h)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/${header}")
    message(FATAL_ERROR "Incomplete Capstone runtime: missing include/${header}")
  endif()
endforeach()
add_library(capstone-runtime INTERFACE)
add_library(Capstone::Runtime ALIAS capstone-runtime)
target_include_directories(capstone-runtime INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
