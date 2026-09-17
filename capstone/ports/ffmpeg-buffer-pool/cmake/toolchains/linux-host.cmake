# Ordinary RISC-V Linux userspace, used only for the domain loader.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(FFPOOL_PLATFORM linux-host CACHE STRING "Execution platform" FORCE)

include("${CMAKE_CURRENT_LIST_DIR}/../Workspace.cmake")
get_filename_component(repo "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
ffpool_path(CAPSTONE_BUILDROOT_DIR "${repo}/capstone/caplifive-buildroot" "Prepared guest Linux toolchain and libcapstone")
set(CMAKE_C_COMPILER "${CAPSTONE_BUILDROOT_DIR}/build/host/bin/riscv64-buildroot-linux-gnu-gcc")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_BUILDROOT_DIR)
