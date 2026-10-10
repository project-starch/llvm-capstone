add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/capstone-runtime")
add_library(allocators-options INTERFACE)
target_link_libraries(allocators-options INTERFACE Capstone::Runtime)
# The shims live beside the census; the port includes them from there rather
# than carrying a second copy that could drift.
target_include_directories(allocators-options INTERFACE
  "${PROJECT_SOURCE_DIR}/src/shared" "${MC_SOURCE}" "${PROJECT_SOURCE_DIR}/../adapted")
# NDEBUG is memcached's own production build (Makefile.am:92). CHUNK_ALIGN_BYTES
# is 8 upstream; 16 here, because an item begins with pointers, which are
# 16-byte capabilities (the application's patch 0001). The shim defaults to 8 so
# a census keeps upstream's table.
target_compile_definitions(allocators-options INTERFACE NDEBUG CHUNK_ALIGN_BYTES=16)
target_compile_options(allocators-options INTERFACE -ffunction-sections -fdata-sections -Wall -Wextra)
option(MCP_POISONCAP "Use the experimental PoisonCap adapter on the CheriBSD build" OFF)
if(MCP_POISONCAP AND NOT PORT_PLATFORM STREQUAL "cheribsd")
  message(FATAL_ERROR "MCP_POISONCAP requires the CheriBSD purecap toolchain")
endif()
# The lifetime ledger, for the arms that go through the hooks (patch 0002). Stock
# CheriBSD has its own, because the platform's malloc places every unit and there
# is no region to carve. Native and PoisonCap run the SAME ledger over one region
# and only the authority layer beneath it differs. The Sublet arm (MCP_SUBLET,
# cmake/MemcachedSource.cmake) has no hooks and no ledger.
# MCP_STOCK_MALLOC: the hosted build on the same stock ledger, so a native tool
# (AddressSanitizer) sees what upstream memcached gives it -- each slab page and
# each cache.c object its own malloc -- rather than one arena with no redzones
# inside it. Off, the native build keeps the shared ledger over one region.
option(MCP_STOCK_MALLOC "Hosted native build: pages and objects from the host malloc, as upstream" OFF)
if(MCP_STOCK_MALLOC AND (NOT PORT_HOSTED OR PORT_PLATFORM STREQUAL "cheribsd"))
  message(FATAL_ERROR "MCP_STOCK_MALLOC is for the hosted native build; stock CheriBSD already uses that ledger")
endif()
if(MCP_SUBLET AND MCP_STOCK_MALLOC)
  message(FATAL_ERROR "MCP_SUBLET has no ledger and MCP_STOCK_MALLOC is one; choose one")
endif()
if(MCP_SUBLET)
  # slabs.c and cache.c take their pages and objects from the system allocator, as
  # upstream does, and derive and revoke every chunk and object themselves.
  set(ledger)
elseif((PORT_PLATFORM STREQUAL "cheribsd" AND NOT MCP_POISONCAP) OR MCP_STOCK_MALLOC)
  set(ledger src/cheribsd/malloc-leases.c src/shared/metadata.c)
else()
  set(ledger src/shared/leases.c src/shared/metadata.c)
endif()
add_library(mc-allocators OBJECT "${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c"
  src/shared/services.c ${ledger})
# Upstream's warnings are upstream's; ours stay on.
set_source_files_properties("${MC_SOURCE}/slabs.c" "${MC_SOURCE}/cache.c"
  PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter;-Wno-sign-compare")
target_link_libraries(mc-allocators PUBLIC allocators-options)
add_dependencies(mc-allocators memcached-source)
# The seam the bug corpus builds through. The corpus source supplies
# mcp_replay, so the corpus stays outside the port and the port keeps one way
# in. The hosted builds read it.
set(MCP_CORPUS_SRC "" CACHE FILEPATH "Corpus-supplied program that defines mcp_replay")
if(NOT PORT_HOSTED)
  message(FATAL_ERROR "the allocators port builds hosted only: native, cheribsd or capstone-application")
endif()

# The shim takes the host's <pthread.h> in a hosted build (see mc_pthread_shim.h).
find_package(Threads REQUIRED)
target_link_libraries(allocators-options INTERFACE Threads::Threads)
if(PORT_PLATFORM STREQUAL "cheribsd")
  # Either CheriBSD adapter owns its own backing, so the entry passes none.
  target_compile_definitions(allocators-options INTERFACE MCP_ADAPTER_BACKING)
  # The SDK ships ld.lld and no ld; without this clang falls back to the host
  # linker. The cheribsd preset says the same through CMAKE_EXE_LINKER_FLAGS,
  # but the shared host/cheribsd/build.py configures from the toolchain file
  # alone, so the port says it here too.
  add_link_options(-fuse-ld=lld)
  if(MCP_POISONCAP)
    # PoisonCap: one mmap'd arena with poison authority under the shared
    # ledger. Mode 0 is bounded leases; mode 1 poisons and sweeps a unit as
    # it is released.
    target_compile_definitions(allocators-options INTERFACE MCP_POISONCAP)
    target_include_directories(allocators-options INTERFACE
      "${PROJECT_SOURCE_DIR}/src/cheribsd")
    target_sources(mc-allocators PRIVATE src/cheribsd/poisoncap-authority.c)
    # The platform's own instruction control, referenced where it already
    # lives rather than copied: live access, poison, sweep, reuse, and a
    # retained old alias that must be refused.
    add_executable(poisoncap-probe
      "${PROJECT_SOURCE_DIR}/../../ffmpeg/buffer-pool/security-tests/cheribsd/poisoncap-probe.c")
    target_link_libraries(poisoncap-probe PRIVATE allocators-options)
    # The poison primitives and the published revocation header are GNU C.
    set_target_properties(poisoncap-probe mc-allocators PROPERTIES C_EXTENSIONS ON)
  else()
    # Stock CheriBSD: pages and objects from the platform's own malloc, so its
    # revocation is asked the question at the level where it lives. No payload
    # region, no ledger over one: src/cheribsd/malloc-leases.c is the seam.
    # The positive control: this guest's libc revocation, made to fire at the
    # corpus's own labelled load shape. Pure libc, no port library.
    add_executable(revocation-control security-tests/cheribsd/revocation-control.c)
    # The probes use GNU inline asm with a "C" operand, as cheric.h does.
    set_target_properties(revocation-control PROPERTIES C_EXTENSIONS ON)
  endif()
elseif(MCP_STOCK_MALLOC)
  # The ledger owns its backing, as on stock CheriBSD; the entry passes none.
  target_compile_definitions(allocators-options INTERFACE MCP_ADAPTER_BACKING)
elseif(MCP_SUBLET)
  target_compile_definitions(allocators-options INTERFACE MC_CAPSTONE_SUBLET)
else()
  target_sources(mc-allocators PRIVATE src/native/authority.c)
endif()
add_library(mc-allocators-library STATIC $<TARGET_OBJECTS:mc-allocators>)
set_target_properties(mc-allocators-library PROPERTIES OUTPUT_NAME memcached-allocators)
target_link_libraries(mc-allocators-library PUBLIC allocators-options)
add_library(Memcached::Allocators ALIAS mc-allocators-library)
include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
port_add_client(Memcached::Allocators)
# The example and the replay entry speak the ledger's interface; the Sublet
# build has none, and the corpus links its cases with its own driver.
if(NOT MCP_SUBLET)
  add_executable(allocator-example examples/allocators.c)
  target_link_libraries(allocator-example PRIVATE Memcached::Allocators)
  add_dependencies(allocator-example memcached-source)
endif()
if(MCP_CORPUS_SRC AND NOT MCP_SUBLET)
  add_executable(defects src/native/main.c "${MCP_CORPUS_SRC}")
  target_link_libraries(defects PRIVATE mc-allocators)
  add_dependencies(defects memcached-source)
  target_link_options(defects PRIVATE LINKER:--gc-sections)
  if(PORT_PLATFORM STREQUAL "cheribsd")
    set_target_properties(defects PROPERTIES C_EXTENSIONS ON)
  endif()
endif()
