add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/capstone-runtime")
add_library(pools-options INTERFACE)
target_link_libraries(pools-options INTERFACE Capstone::Runtime)
# The shim stays where the census keeps it; the port includes it from there
# rather than carrying a second copy that could drift.
target_include_directories(pools-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared" "${APRP_SOURCE}/include" "${PROJECT_SOURCE_DIR}/../adapted")
target_compile_definitions(pools-options INTERFACE APRP_PORT APR_ALIGN_DEFAULT_BOUNDARY=16)
target_compile_options(pools-options INTERFACE -ffunction-sections -fdata-sections -Wall -Wextra)
add_library(apr-pools OBJECT "${APRP_SOURCE}/memory/unix/apr_pools.c"
  src/shared/metadata.c src/shared/services.c)
# Upstream's unused-parameter warnings are upstream's; ours stay on.
set_source_files_properties("${APRP_SOURCE}/memory/unix/apr_pools.c"
  PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
target_link_libraries(apr-pools PUBLIC pools-options)
add_dependencies(apr-pools apr-source)
# The seam the bug corpus builds through. The corpus source supplies
# aprp_replay, so the corpus stays outside the port and the port keeps one way
# in. Both the Capstone domain build and the hosted build read it.
set(APRP_CORPUS_SRC "" CACHE FILEPATH "Corpus-supplied program that defines aprp_replay")
if(PORT_HOSTED)
  target_sources(apr-pools PRIVATE src/native/node-pointers.c)
  add_library(apr-pools-library STATIC $<TARGET_OBJECTS:apr-pools>)
  set_target_properties(apr-pools-library PROPERTIES OUTPUT_NAME apr-pools)
  target_link_libraries(apr-pools-library PUBLIC pools-options)
  add_library(APR::Pools ALIAS apr-pools-library)
  add_executable(allocator-example examples/pools.c)
  target_link_libraries(allocator-example PRIVATE APR::Pools)
  add_dependencies(allocator-example apr-source)
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(APR::Pools)
  if(APRP_CORPUS_SRC)
    add_executable(defects src/native/main.c "${APRP_CORPUS_SRC}")
    target_link_libraries(defects PRIVATE apr-pools)
    add_dependencies(defects apr-source)
    target_link_options(defects PRIVATE LINKER:--gc-sections)
  endif()
else()
  enable_language(ASM)
  target_compile_definitions(pools-options INTERFACE APRP_DOMAIN)
  target_sources(apr-pools PRIVATE src/allocators/sublet/node-leases.c)
  if(APRP_CORPUS_SRC)
    set(link_script "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/link.ld")
    add_executable(defects src/capstone-domain/entry.c "${APRP_CORPUS_SRC}"
      "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
      "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
      "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
    target_link_libraries(defects PRIVATE apr-pools)
    add_dependencies(defects apr-source)
    target_link_options(defects PRIVATE --gc-sections -T "${link_script}")
    set_target_properties(defects PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
  endif()
endif()
