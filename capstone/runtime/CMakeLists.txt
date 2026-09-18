# Header-only capability operations and Sublet lifetime helpers.
add_library(capstone-runtime INTERFACE)
add_library(Capstone::Runtime ALIAS capstone-runtime)
target_include_directories(capstone-runtime INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
