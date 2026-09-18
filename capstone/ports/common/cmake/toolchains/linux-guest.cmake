# Ordinary RISC-V Linux userspace inside QEMU, used by application loaders.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(PORT_PLATFORM linux-guest CACHE STRING "Execution platform" FORCE)
include("${CMAKE_CURRENT_LIST_DIR}/../Workspace.cmake")
set(CMAKE_C_COMPILER "${CAPSTONE_BUILDROOT_DIR}/build/host/bin/riscv64-buildroot-linux-gnu-gcc")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_BUILDROOT_DIR)
