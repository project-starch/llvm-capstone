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
# apr-util's bucket allocator, a client of the pool allocator, carried by the
# same library when asked for: its blocks are pool nodes, its level below is
# this one, and a second component would only duplicate the seam.
option(APRP_BUCKETS "Build apr-util's bucket allocator on top of the pools" OFF)
if(APRP_BUCKETS)
  include(cmake/APRUtilSource.cmake)
  target_sources(apr-pools PRIVATE "${APU_SOURCE}/buckets/apr_buckets_alloc.c")
  set_source_files_properties("${APU_SOURCE}/buckets/apr_buckets_alloc.c"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
  target_compile_definitions(pools-options INTERFACE APRP_BUCKETS)
  add_dependencies(apr-pools apr-util-source)
endif()
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
  if(PORT_PLATFORM STREQUAL "cheribsd")
    # Stock CheriBSD: nodes from the platform's own malloc, so its revocation
    # is asked the question at the level where it lives. No payload region.
    option(APRP_POISONCAP "PoisonCap lifetimes on CheriBSD: mapped nodes, poison and sweep at release" OFF)
    if(APRP_POISONCAP)
      target_compile_definitions(pools-options INTERFACE APRP_POISONCAP)
      target_sources(apr-pools PRIVATE src/cheribsd/node-poison.c)
      if(APRP_BUCKETS)
        target_sources(apr-pools PRIVATE src/cheribsd/bucket-poison.c)
      endif()
    else()
      target_compile_definitions(pools-options INTERFACE APRP_NODES_FROM_MALLOC)
      target_sources(apr-pools PRIVATE src/cheribsd/node-malloc.c)
      if(APRP_BUCKETS)
        target_sources(apr-pools PRIVATE src/native/bucket-pointers.c)
      endif()
    endif()
    # The positive control: this guest's libc revocation, made to fire at the
    # corpus's own labelled load shape. Pure libc, no port library.
    add_executable(revocation-control security-tests/cheribsd/revocation-control.c)
    # The probes use GNU inline asm with a "C" operand, as cheric.h does.
    set_target_properties(revocation-control PROPERTIES C_EXTENSIONS ON)
  else()
    target_sources(apr-pools PRIVATE src/native/node-pointers.c)
    if(APRP_BUCKETS)
      target_sources(apr-pools PRIVATE src/native/bucket-pointers.c)
    endif()
  endif()
  add_library(apr-pools-library STATIC $<TARGET_OBJECTS:apr-pools>)
  set_target_properties(apr-pools-library PROPERTIES OUTPUT_NAME apr-pools)
  target_link_libraries(apr-pools-library PUBLIC pools-options)
  add_library(APR::Pools ALIAS apr-pools-library)
  add_executable(allocator-example examples/pools.c)
  target_link_libraries(allocator-example PRIVATE APR::Pools)
  add_dependencies(allocator-example apr-source)
  if(APRP_BUCKETS)
    add_executable(bucket-example examples/buckets.c)
    target_link_libraries(bucket-example PRIVATE APR::Pools)
    add_dependencies(bucket-example apr-util-source)
  endif()
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(APR::Pools)
  if(APRP_CORPUS_SRC)
    add_executable(defects src/native/main.c "${APRP_CORPUS_SRC}")
    target_link_libraries(defects PRIVATE apr-pools)
    add_dependencies(defects apr-source)
    target_link_options(defects PRIVATE LINKER:--gc-sections)
    if(PORT_PLATFORM STREQUAL "cheribsd")
      set_target_properties(defects PROPERTIES C_EXTENSIONS ON)
    endif()
  endif()
else()
  enable_language(ASM)
  target_compile_definitions(pools-options INTERFACE APRP_DOMAIN)
  target_sources(apr-pools PRIVATE src/allocators/sublet/node-leases.c)
  if(APRP_BUCKETS)
    target_sources(apr-pools PRIVATE src/allocators/sublet/bucket-leases.c)
  endif()
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
