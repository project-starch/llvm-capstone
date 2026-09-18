include("${CMAKE_CURRENT_LIST_DIR}/../../../common/cmake/Workspace.cmake")

port_path(PG_WORK "${tmp_root}/postgres-memory-contexts" "Archive cache and conventional runner paths")
port_path(PG_LINUX_BUILD_DIR "${PG_WORK}/build/linux-guest" "Guest Linux loader build")
