set(libcapstone "${CAPSTONE_BUILDROOT_DIR}/package/modcapstone/userspace/lib")
add_executable(domain-loader domain-loader.c "${libcapstone}/libcapstone.c")
target_include_directories(domain-loader PRIVATE "${libcapstone}"
  "${CAPSTONE_REPO_ROOT}/capstone/runtime/include"
  "${CAPSTONE_BUILDROOT_DIR}/package/modcapstone/include")
target_link_libraries(domain-loader PRIVATE region-options Capstone::LinuxFaultPolicy)
target_compile_options(domain-loader PRIVATE -Wall)
if(BUILD_TESTING)
  add_executable(fault-supervisor ../../security-tests/linux-guest/fault-supervisor.c)
endif()
