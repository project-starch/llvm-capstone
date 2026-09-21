add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/capstone-runtime")
add_library(allocators-options INTERFACE)
target_link_libraries(allocators-options INTERFACE Capstone::Runtime)
# The shims live beside the census; the port includes them from there rather
# than carrying a second copy that could drift.
target_include_directories(allocators-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared" "${MC_SOURCE}" "${PROJECT_SOURCE_DIR}/../adapted")
# NDEBUG is memcached's own production build (Makefile.am:92). CHUNK_ALIGN_BYTES
# is 8 upstream; 16 here, because a chunk is a Sublet region and a region is
# whole capabilities. The shim defaults to 8 so a census keeps upstream's table.
target_compile_definitions(allocators-options INTERFACE NDEBUG CHUNK_ALIGN_BYTES=16)
target_compile_options(allocators-options INTERFACE -ffunction-sections -fdata-sections -Wall -Wextra)
# The lifetime ledger: one over a payload region for native and domain builds,
# one over the platform's malloc for stock CheriBSD.
if(PORT_PLATFORM STREQUAL "cheribsd")
  set(ledger src/cheribsd/malloc-leases.c)
else()
  set(ledger src/shared/leases.c)
endif()
add_library(mc-allocators OBJECT "${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c"
  src/shared/metadata.c src/shared/services.c ${ledger})
# Upstream's warnings are upstream's; ours stay on.
set_source_files_properties("${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c"
  PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter;-Wno-sign-compare")
target_link_libraries(mc-allocators PUBLIC allocators-options)
add_dependencies(mc-allocators memcached-source)
# The seam the bug corpus builds through. The corpus source supplies
# mcp_replay, so the corpus stays outside the port and the port keeps one way
# in. Both the Capstone domain build and the hosted build read it.
set(MCP_CORPUS_SRC "" CACHE FILEPATH "Corpus-supplied program that defines mcp_replay")
if(PORT_HOSTED)
  # The shim takes the host's <pthread.h> in a hosted build (see mc_pthread_shim.h).
  find_package(Threads REQUIRED)
  target_link_libraries(allocators-options INTERFACE Threads::Threads)
  if(PORT_PLATFORM STREQUAL "cheribsd")
    # Stock CheriBSD: pages and objects from the platform's own malloc, so its
    # revocation is asked the question at the level where it lives. No payload
    # region, no ledger over one: src/cheribsd/malloc-leases.c is the seam.
    target_compile_definitions(allocators-options INTERFACE MCP_UNITS_FROM_MALLOC)
    # The SDK ships ld.lld and no ld; without this clang falls back to the host
    # linker. The cheribsd preset says the same through CMAKE_EXE_LINKER_FLAGS,
    # but the shared host/cheribsd/build.py configures from the toolchain file
    # alone, so the port says it here too.
    add_link_options(-fuse-ld=lld)
    # The positive control: this guest's libc revocation, made to fire at the
    # corpus's own labelled load shape. Pure libc, no port library.
    add_executable(revocation-control security-tests/cheribsd/revocation-control.c)
    # The probes use GNU inline asm with a "C" operand, as cheric.h does.
    set_target_properties(revocation-control PROPERTIES C_EXTENSIONS ON)
  else()
    target_sources(mc-allocators PRIVATE src/native/authority.c)
  endif()
  add_library(mc-allocators-library STATIC $<TARGET_OBJECTS:mc-allocators>)
  set_target_properties(mc-allocators-library PROPERTIES OUTPUT_NAME memcached-allocators)
  target_link_libraries(mc-allocators-library PUBLIC allocators-options)
  add_library(Memcached::Allocators ALIAS mc-allocators-library)
  add_executable(allocator-example examples/allocators.c)
  target_link_libraries(allocator-example PRIVATE Memcached::Allocators)
  add_dependencies(allocator-example memcached-source)
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(Memcached::Allocators)
  if(MCP_CORPUS_SRC)
    add_executable(defects src/native/main.c "${MCP_CORPUS_SRC}")
    target_link_libraries(defects PRIVATE mc-allocators)
    add_dependencies(defects memcached-source)
    target_link_options(defects PRIVATE LINKER:--gc-sections)
    if(PORT_PLATFORM STREQUAL "cheribsd")
      set_target_properties(defects PROPERTIES C_EXTENSIONS ON)
    endif()
  endif()
else()
  enable_language(ASM)
  target_compile_definitions(allocators-options INTERFACE MCP_DOMAIN)
  # A pointer is 16 bytes here and uintptr_t 8: the cast the ledger keys its
  # tables by IS the address-of-a-capability operation, so the warning that
  # names it carries no information.
  target_compile_options(allocators-options INTERFACE
    -Wno-pointer-to-int-cast -Wno-void-pointer-to-int-cast)
  target_sources(mc-allocators PRIVATE src/allocators/sublet/authority.c)
  # slabs_init grows the class table by a double factor (size *= 1.25). The
  # freestanding target has no FP ABI and no compiler-rt of its own, so the
  # double builtins are compiled in from the tree, as the BEEBS FP benchmarks
  # do (benchmarks/beebs/build-beebs-softfloat-common.sh). Nine files; the
  # linker names any the allocator needs beyond them.
  set(builtins "${CAPSTONE_REPO_ROOT}/compiler-rt/lib/builtins")
  foreach(builtin adddf3 subdf3 muldf3 divdf3 fixdfsi floatsidf comparedf2 floatunsidf fixunsdfsi)
    list(APPEND softfloat "${builtins}/${builtin}.c")
  endforeach()
  target_sources(mc-allocators PRIVATE ${softfloat})
  set_source_files_properties(${softfloat} PROPERTIES
    COMPILE_OPTIONS "-w" INCLUDE_DIRECTORIES "${builtins}")
  if(MCP_CORPUS_SRC)
    set(link_script "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/link.ld")
    add_executable(defects src/capstone-domain/entry.c "${MCP_CORPUS_SRC}"
      "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
      "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
      "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S")
    target_link_libraries(defects PRIVATE mc-allocators)
    add_dependencies(defects memcached-source)
    target_link_options(defects PRIVATE --gc-sections -T "${link_script}")
    set_target_properties(defects PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
  endif()
endif()
