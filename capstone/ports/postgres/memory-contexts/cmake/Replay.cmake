add_library(replay-options INTERFACE)
# The nested arm in a Capstone process: patch 0003 makes every chunk a child
# lifetime of its block (CDERIVE), revoked by pfree and repalloc (CREVOKE).
option(PG_SUBLET "Memory-context chunks as Sublet lifetimes (patch 0003)" OFF)
if(PG_SUBLET AND NOT PORT_PLATFORM STREQUAL "capstone-application")
  message(FATAL_ERROR "PG_SUBLET needs the capstone-application toolchain")
endif()
if(NOT PORT_HOSTED)
  message(FATAL_ERROR "the memory-context port builds hosted programs only: capstone-application, cheribsd or native")
endif()
set(PG_CORPUS_DIR "" CACHE PATH "Corpus root: bug-corpora/postgres/mmgr-repros")
# The corpus contract (bug-corpora/cpython/pymalloc-repros/SCHEMA.md) is one
# directory per case, NN_<upstream-fix>_<slug>, holding a case.c that is a
# complete translation unit. Both targets build from the same list, and the
# target stem is what the contract calls a run artifact -- 03-live-parts-stale-
# alias -- so an archived result tree stays readable away from the corpus.
function(pg_corpus_cases dirs_out stems_out)
  set(dirs "")
  set(stems "")
  if(PG_CORPUS_DIR)
    file(GLOB found CONFIGURE_DEPENDS "${PG_CORPUS_DIR}/[0-9][0-9]_*")
    foreach(dir ${found})
      if(IS_DIRECTORY "${dir}" AND EXISTS "${dir}/case.c")
        get_filename_component(name "${dir}" NAME)
        string(REGEX REPLACE "^([0-9][0-9])_[^_]+_(.*)$" "\\1;\\2" parts "${name}")
        list(GET parts 0 num)
        list(GET parts 1 slug)
        string(REPLACE "_" "-" slug "${slug}")
        list(APPEND dirs "${dir}")
        list(APPEND stems "${num}-${slug}")
      endif()
    endforeach()
  endif()
  set(${dirs_out} "${dirs}" PARENT_SCOPE)
  set(${stems_out} "${stems}" PARENT_SCOPE)
endfunction()
target_link_libraries(replay-options INTERFACE Capstone::Runtime)
target_include_directories(replay-options INTERFACE "${PROJECT_SOURCE_DIR}/src/shared")
target_compile_options(replay-options INTERFACE
  "$<$<COMPILE_LANGUAGE:C>:-ffunction-sections;-fdata-sections>")
option(PG_CHECK_DATA "Write/check payloads at free and realloc" ON)
if(PG_CHECK_DATA)
  target_compile_definitions(replay-options INTERFACE REPLAY_CHECK_DATA)
endif()

function(pg_manager name mode)
  if(mode STREQUAL "native")
    set(aset "${PG_SOURCE}/src/backend/utils/mmgr/aset.c")
  else()
    set(aset "${CMAKE_BINARY_DIR}/variants/${mode}/aset.c")
  endif()
  add_library(${name} OBJECT "${aset}" src/shared/postgres-compat.c)
  foreach(file mcxt generation slab bump alignedalloc memdebug)
    if(mode STREQUAL "sublet" AND file MATCHES "^(mcxt|generation|slab|bump)$")
      target_sources(${name} PRIVATE "${CMAKE_BINARY_DIR}/variants/sublet/${file}.c")
    else()
      target_sources(${name} PRIVATE "${PG_SOURCE}/src/backend/utils/mmgr/${file}.c")
    endif()
  endforeach()
  target_link_libraries(${name} PUBLIC replay-options)
  if(NOT mode STREQUAL "native")
    target_include_directories(${name} PUBLIC "${CMAKE_BINARY_DIR}/variants/${mode}/include")
  endif()
  target_include_directories(${name} PUBLIC "${PG_SOURCE}/src/include" "${PG_SOURCE}/src/backend")
endfunction()

# Both hosted CAPABILITY platforms, because the reason is the pointer and not
# the operating system: upstream's aset.c asserts that an AllocFreeListLink
# fits in the minimum chunk, and a 16-byte pointer does not. Building
# capstone-application against the plain native manager fails that assertion
# at compile time (measured 2026-10-07), which is the assertion doing its job.
if(PORT_PLATFORM MATCHES "^(cheribsd|capstone-application)$")
  # The capability-layout variant has 16-byte chunks and free links.
  if(PG_SUBLET)
    pg_manager(manager-native sublet)
    set(capability_variant sublet)
  else()
    pg_manager(manager-native spatial)
    set(capability_variant spatial)
  endif()
  file(READ "${CMAKE_BINARY_DIR}/variants/${capability_variant}/include/pg_config.h"
    capability_config)
  string(REPLACE "#define SIZEOF_VOID_P 8" "#define SIZEOF_VOID_P 16"
    capability_config "${capability_config}")
  file(WRITE "${CMAKE_BINARY_DIR}/capability-include/pg_config.h" "${capability_config}")
  target_include_directories(manager-native BEFORE PUBLIC
    "${CMAKE_BINARY_DIR}/capability-include")
else()
  pg_manager(manager-native native)
endif()
add_library(postgres-contexts STATIC src/native/replay/printf.c)
target_link_libraries(postgres-contexts PUBLIC manager-native)
add_library(PostgreSQL::MemoryContexts ALIAS postgres-contexts)
add_executable(allocator-example examples/contexts.c)
target_link_libraries(allocator-example PRIVATE PostgreSQL::MemoryContexts)
include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
port_add_client(PostgreSQL::MemoryContexts)
# src/native/replay/main.c INTERPOSES malloc/free/realloc to count the
# manager's blocks. That works where libc's allocator is the only one, and it
# does not work on a platform whose runtime supplies malloc itself: on
# capstone-application the SDK's libapplication-runtime.a defines malloc in
# heap.c, and linking the interposer beside it is a duplicate symbol. The
# non-interposing entry is the right one for both such platforms, and its own
# header already gives the reason -- on an ABI that is not the recording
# host's, recorded backing counts are reference observations rather than
# required outcomes, so counting them here buys nothing.
if(PORT_PLATFORM MATCHES "^(cheribsd|capstone-application)$")
  set(hosted_entry src/cheribsd/main.c)
else()
  set(hosted_entry src/native/replay/main.c)
endif()
add_executable(replay ${hosted_entry} src/native/replay/printf.c src/shared/replay-engine.c)
target_link_libraries(replay PRIVATE manager-native)
target_link_options(replay PRIVATE LINKER:--gc-sections)
add_executable(contexts-native tests/contexts-native.c src/native/replay/printf.c)
target_link_libraries(contexts-native PRIVATE manager-native)
target_link_options(contexts-native PRIVATE LINKER:--gc-sections)
add_test(NAME contexts-native COMMAND contexts-native)
set_tests_properties(contexts-native PROPERTIES LABELS native TIMEOUT 60)
# The corpus belongs to the PLATFORM, not to the protection mechanism: the
# same cases must build against the plain spatial CheriBSD manager as well,
# so that the arm running under the guest's OWN libc revocation can be
# measured rather than argued about.
# Built on every hosted platform that HAS a fault oracle: CheriBSD, where a
# supervisor reads signal and si_code from outside, and capstone-application,
# where the virtual launcher reports the fault and the runner resolves the
# probe from the image. Not on native, which has neither and uses `replay`.
if(PORT_PLATFORM MATCHES "^(cheribsd|capstone-application)$")
  # One program per case, per the corpus contract in
  # bug-corpora/cpython/pymalloc-repros/SCHEMA.md: a capability fault ends
  # the run, so a case that provokes one cannot report results beside it.
  # Programs are named as the contract names run artifacts -- 03-live-parts-
  # stale-alias, not defect-3 -- so an archived result tree stays readable
  # away from the corpus.
  pg_corpus_cases(pg_dirs pg_stems)
  list(LENGTH pg_dirs pg_case_count)
  if(PG_CORPUS_DIR AND pg_case_count EQUAL 0)
    message(FATAL_ERROR
      "PG_CORPUS_DIR=${PG_CORPUS_DIR} holds no NN_*/case.c; a corpus that "
      "builds nothing must not look like a corpus that passed")
  endif()
  math(EXPR pg_last "${pg_case_count} - 1")
  foreach(i RANGE 0 ${pg_last})
    if(pg_case_count GREATER 0)
      list(GET pg_dirs ${i} pg_dir)
      list(GET pg_stems ${i} pg_stem)
      add_executable("${pg_stem}" "${pg_dir}/case.c" "${PG_CORPUS_DIR}/shared/driver.c")
      target_include_directories("${pg_stem}" PRIVATE "${PG_CORPUS_DIR}/shared")
      target_compile_definitions("${pg_stem}" PRIVATE PG_CORPUS_HOSTED)
      target_link_libraries("${pg_stem}" PRIVATE PostgreSQL::MemoryContexts)
    endif()
  endforeach()
  if(pg_case_count GREATER 0)
    message(STATUS "PostgreSQL defect corpus: ${pg_case_count} hosted cases")
  endif()
  # The out-of-process observer is CheriBSD's; the virtual arm's observer is
  # the launcher, and this host program would not link against a Capstone libc.
  if(PORT_PLATFORM STREQUAL "cheribsd")
    add_executable(supervise
      "${CAPSTONE_REPO_ROOT}/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c")
    target_compile_definitions(supervise PRIVATE PROBE_SYMBOL="pg_defect_probe")
    target_link_libraries(supervise PRIVATE util)
  endif()
endif()
