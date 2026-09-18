include("${CMAKE_CURRENT_LIST_DIR}/../../../common/cmake/Workspace.cmake")

port_path(FFPOOL_WORK "${tmp_root}/ffmpeg-buffer-pool" "Archive cache root and conventional runner paths; use -B to relocate a CMake build")
port_path(FFPOOL_NATIVE_BUILD_DIR "${FFPOOL_WORK}/build/native" "Native build used by recording and QEMU tests")
port_path(FFPOOL_DOMAIN_BUILD_DIR "${FFPOOL_WORK}/build/capstone-domain" "Capstone domain build used by QEMU runners")
port_path(FFPOOL_LINUX_BUILD_DIR "${FFPOOL_WORK}/build/linux-guest" "Guest Linux loader build used by QEMU runners")
port_path(FFPOOL_RECORDING_DIR "${FFPOOL_NATIVE_BUILD_DIR}/test-results/workload" "Native CTest recording used by QEMU replay tests")
