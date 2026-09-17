set(libcapstone "${CAPSTONE_BUILDROOT_DIR}/package/modcapstone/userspace/lib")
add_executable(domain-loader domain-loader.c "${libcapstone}/libcapstone.c")
target_include_directories(domain-loader PRIVATE
  "${PROJECT_SOURCE_DIR}/shared"
  "${libcapstone}"
  "${CAPSTONE_BUILDROOT_DIR}/package/modcapstone/include")
target_compile_options(domain-loader PRIVATE -Wall)
