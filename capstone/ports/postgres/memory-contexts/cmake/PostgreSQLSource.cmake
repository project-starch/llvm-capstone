port_read_upstream()
set(PG_VERSION "${UPSTREAM_version}")
set(PG_SHA256 "${UPSTREAM_sha256}")
port_path(PG_ARCHIVE "${PG_WORK}/download/postgresql-${PG_VERSION}.tar.bz2" "Pinned PostgreSQL archive")
port_download("${PG_ARCHIVE}")
set(PG_SOURCE "${CMAKE_BINARY_DIR}/source/postgresql-${PG_VERSION}")
find_program(PG_HOST_CC NAMES cc gcc REQUIRED)
find_program(PG_MAKE NAMES gmake make REQUIRED)
find_program(PORT_PATCH_TOOL NAMES patch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/prepare-source.py"
  --archive "${PG_ARCHIVE}" --version "${PG_VERSION}" --sha256 "${PG_SHA256}"
  --source "${PG_SOURCE}" --variants "${CMAKE_BINARY_DIR}/variants"
  --patches "${PROJECT_SOURCE_DIR}/patches" --cc "${PG_HOST_CC}" --make "${PG_MAKE}"
  --patch-tool "${PORT_PATCH_TOOL}"
  RESULT_VARIABLE prepare_status)
if(NOT prepare_status EQUAL 0)
  message(FATAL_ERROR "PostgreSQL source preparation failed; see source/prepare.log in this build.")
endif()
file(GLOB patch_inputs "${PROJECT_SOURCE_DIR}/patches/*.patch")
file(GLOB manager_inputs "${PG_SOURCE}/src/backend/utils/mmgr/*.c")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/prepare-source.py" "${PG_ARCHIVE}" ${patch_inputs}
  ${manager_inputs} "${PG_SOURCE}/src/include/utils/memutils_memorychunk.h")
