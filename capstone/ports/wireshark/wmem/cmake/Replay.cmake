add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" "${CMAKE_BINARY_DIR}/runtime")
set(WM_SHARED src/shared/scopes.c src/shared/freestanding.c)
# The backing policy is the one part that differs per platform: Capstone
# regions with Sublet handles, a native bump, or PoisonCap-mapped regions.
set(WM_BACKING src/shared/backing.c)
if(PORT_PLATFORM STREQUAL "cheribsd")
  option(WM_POISONCAP "PoisonCap lifetimes on CheriBSD" OFF)
  if(WM_POISONCAP)
    set(WM_BACKING src/cheribsd/poisoncap.c)
  endif()
endif()
# The chunk port (patches/...-0002): every chunk of the block allocator is a
# region of its own, so a chunk free is a revoke. It needs the Sublet backing,
# so the PoisonCap arm keeps the region-granular hooks and its own per-chunk
# poisoning.
set(WM_CHUNKS_HOSTED)
if(NOT WM_POISONCAP)
  set(WM_CHUNKS_HOSTED src/native/chunks.c)
endif()
# WM_CHUNKS=OFF builds the region-granular hooks alone from the same tree: every hunk of 0002 is
# guarded by WMEM_PORT_CHUNKS, so the patch is applied and inert. With WM_P2_CONTROL the fixtures'
# check that a stale unprotected read sees the old byte is compiled in anyway -- the positive
# control for that check, which must FAIL when the headers are still in the chunk.
option(WM_CHUNKS "the chunk port (patches/...-0002)" ON)
option(WM_P2_CONTROL "compile the old-byte check without the chunk port" OFF)
# WM_P1_ABLATE keeps the port and stubs out the ONE give a chunk free performs (chunks.c,
# wm_chunk_retire): the matched arm that attributes a chunk-free fault to that revoke alone. Valid
# only for fixtures that never reissue the freed chunk (4 and 13); anything that does would take a
# slot still holding a lent handle.
option(WM_P1_ABLATE "the port with a chunk free's revoke stubbed out" OFF)
# WM_VARIANT=reference builds the defect corpus's domain programs against wmem AS RELEASED: none
# of the port's patches, so no hook fires, no reset ends an epoch and no chunk is a region. What is
# left is Sublet as the SYSTEM allocator -- every g_malloc a region, every g_free a revoke -- under
# a stock wmem: the corpus's `sublet-malloc` arm. The replay and security programs stay ported.
set(WM_VARIANT "ported" CACHE STRING "wmem source the corpus's domain programs build against")
set_property(CACHE WM_VARIANT PROPERTY STRINGS ported reference)
if(NOT WM_VARIANT MATCHES "^(ported|reference)$")
  message(FATAL_ERROR "WM_VARIANT must be ported or reference, not ${WM_VARIANT}")
endif()
# The nested arm in the VIRTUAL address space. Until this existed the only
# capability build was the freestanding domain, so one macro carried two
# meanings: WM_DOMAIN said both "capabilities" and "no libc". They are now
# WM_CAPABILITY and WM_DOMAIN, and this option asks for the first without the
# second -- the Sublet region and chunk layers in a Capstone PROCESS, with the
# payload LENT by the system allocator as one linear capability.
option(WM_SUBLET "Nested lifetimes in a Capstone process: the Sublet region and chunk layers, with the payload lent linear" OFF)
if(WM_SUBLET AND NOT PORT_PLATFORM STREQUAL "capstone-application")
  message(FATAL_ERROR "WM_SUBLET needs the capstone-application toolchain")
endif()
# One allocation on the virtual profile is at most 256 MiB (runtime/virtual/vm.h), below the 384 MiB
# payload every other platform takes; see the end of src/shared/port.h.
set(WM_VIRTUAL_PAYLOAD_MIB 128 CACHE STRING "capstone-application: the payload in MiB (the virtual heap caps one allocation at 256)")
if(PORT_PLATFORM STREQUAL "capstone-application")
  add_compile_definitions("WM_VIRTUAL_PAYLOAD_BYTES=(${WM_VIRTUAL_PAYLOAD_MIB}UL << 20)")
endif()
function(wm_executable name variant)
  set(workload ${ARGN})
  set(source "${CMAKE_BINARY_DIR}/source-${variant}")
  wm_upstream_units(upstream ${variant})
  if(PORT_HOSTED AND WM_SUBLET)
    set(platform src/native/main.c src/allocators/sublet/regions.c
      src/allocators/sublet/chunks.c ${WM_BACKING})
  elseif(PORT_HOSTED)
    set(platform src/native/main.c src/native/regions.c ${WM_BACKING} ${WM_CHUNKS_HOSTED})
  else()
    set(platform src/capstone-domain/entry.c src/allocators/sublet/regions.c
      src/allocators/sublet/chunks.c
      "${CAPSTONE_REPO_ROOT}/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
      "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/start.S"
      "${CAPSTONE_REPO_ROOT}/capstone/tests/runtime-qemu/gct-section-end.S"
      src/shared/backing.c)
  endif()
  add_executable(${name} ${platform} ${workload} ${WM_SHARED} ${upstream})
  add_dependencies(${name} source-${variant})
  # The shim directory shadows glib.h and the ws_* headers the upstream units include.
  target_include_directories(${name} PRIVATE src/shared/shim src/shared "${source}/wsutil/wmem")
  target_compile_definitions(${name} PRIVATE WMEM_PORT_HOOKS)
  if(WM_POISONCAP)
    target_compile_definitions(${name} PRIVATE WM_POISONCAP)
    target_include_directories(${name} PRIVATE src/cheribsd)
  elseif(WM_CHUNKS)
    target_compile_definitions(${name} PRIVATE WMEM_PORT_CHUNKS)
  endif()
  if(WM_P2_CONTROL)
    target_compile_definitions(${name} PRIVATE WM_P2_CONTROL)
  endif()
  if(WM_P1_ABLATE)
    target_compile_definitions(${name} PRIVATE WM_ABLATE_RETIRE_GIVE)
  endif()
  target_compile_options(${name} PRIVATE -Wall -Wextra -ffunction-sections -fdata-sections)
  target_link_libraries(${name} PRIVATE Capstone::Runtime)
  if(WM_SUBLET)
    target_compile_definitions(${name} PRIVATE WM_CAPABILITY WM_BORROW_LINEAR)
  endif()
  if(NOT PORT_HOSTED)
    # Both: the domain is freestanding AND it is a capability build.
    target_compile_definitions(${name} PRIVATE WM_DOMAIN WM_CAPABILITY)
    set(link_script "${CAPSTONE_REPO_ROOT}/capstone/my_first_domain/link.ld")
    target_link_options(${name} PRIVATE --gc-sections -T "${link_script}")
    set_target_properties(${name} PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
  endif()
endfunction()
if(NOT PORT_HOSTED)
  enable_language(ASM)
endif()
wm_source(ported)
wm_executable(replay ported src/shared/replay.c)
if(PORT_HOSTED)
  wm_upstream_units(upstream_ported ported)
  # The corpus's hosted targets link this library, so the nested arm has to be
  # IN IT: the region and chunk layers, and the two definitions, are PUBLIC so
  # that a corpus case compiled against it takes the capability path too.
  if(WM_SUBLET)
    add_library(wireshark-wmem STATIC src/allocators/sublet/regions.c
      src/allocators/sublet/chunks.c ${WM_BACKING} ${WM_SHARED} ${upstream_ported})
    target_compile_definitions(wireshark-wmem PUBLIC WM_CAPABILITY WM_BORROW_LINEAR)
  else()
    add_library(wireshark-wmem STATIC src/native/regions.c ${WM_BACKING} ${WM_CHUNKS_HOSTED} ${WM_SHARED} ${upstream_ported})
  endif()
  add_dependencies(wireshark-wmem source-ported)
  target_include_directories(wireshark-wmem PUBLIC src/shared/shim src/shared
    "${CMAKE_BINARY_DIR}/source-ported/wsutil/wmem")
  target_compile_definitions(wireshark-wmem PUBLIC WMEM_PORT_HOOKS)
  # WM_LIBC_SYSTEM: g_malloc/g_free are the host's malloc/free (src/shared/backing.c), so the
  # unported arms -- ASan natively, libc revocation on stock CheriBSD -- see wmem's system
  # requests as a stock build makes them. Hosted only, and not with the chunk port or PoisonCap,
  # both of which own the backing.
  option(WM_LIBC_SYSTEM "hosted: g_malloc and g_free are the host's malloc and free" OFF)
  if(WM_LIBC_SYSTEM AND (WM_POISONCAP OR WM_CHUNKS OR WM_SUBLET))
    message(FATAL_ERROR "WM_LIBC_SYSTEM needs WM_CHUNKS=OFF, no PoisonCap and no WM_SUBLET: those own the backing")
  endif()
  if(WM_LIBC_SYSTEM)
    target_compile_definitions(wireshark-wmem PUBLIC WM_LIBC_SYSTEM)
  endif()
  if(WM_POISONCAP)
    target_compile_definitions(wireshark-wmem PUBLIC WM_POISONCAP)
    target_include_directories(wireshark-wmem PUBLIC src/cheribsd)
  elseif(WM_CHUNKS)
    target_compile_definitions(wireshark-wmem PUBLIC WMEM_PORT_CHUNKS)
  endif()
  if(PORT_PLATFORM STREQUAL "cheribsd")
    # The corpus arms run under a supervisor that reports the child's fault
    # from outside it and resolves the labelled probe from the child's map.
    add_executable(supervise
      "${CAPSTONE_REPO_ROOT}/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c")
    target_compile_definitions(supervise PRIVATE PROBE_SYMBOL="wm_defect_probe")
    target_link_libraries(supervise PRIVATE util)
    # A case whose defective access is a WRITE faults at wm_defect_write, which the supervisor above
    # cannot resolve; without its own supervisor every such fault read "NOT at probe" (cases 16, 17,
    # 18, 20 and 21, corrected by hand at the 2026-10-10 audit).
    add_executable(supervise-wm_defect_write
      "${CAPSTONE_REPO_ROOT}/capstone/bug-corpora/cpython/pymalloc-repros/observe/supervise.c")
    target_compile_definitions(supervise-wm_defect_write PRIVATE PROBE_SYMBOL="wm_defect_write")
    target_link_libraries(supervise-wm_defect_write PRIVATE util)
  endif()
  target_link_libraries(wireshark-wmem PUBLIC Capstone::Runtime)
  add_library(Wireshark::Wmem ALIAS wireshark-wmem)
  add_executable(allocator-example examples/wmem.c)
  target_link_libraries(allocator-example PRIVATE Wireshark::Wmem)
  include("${PORT_SUPPORT_ROOT}/cmake/Client.cmake")
  port_add_client(Wireshark::Wmem)
  wm_source(reference)
  wm_executable(replay-reference reference src/shared/replay.c)
else()
  wm_executable(wmem-security ported security-tests/shared/lifetimes.c)
  if(WM_VARIANT STREQUAL "reference")
    wm_source(reference)
  endif()
endif()
# The defect corpus: one program per NN_*/case.c, named as the contract names
# run artifacts. Case material lives in bug-corpora, not inside the port.
set(WM_CORPUS_DIR "" CACHE PATH "Defect corpus root holding NN_*/case.c programs")
function(wm_corpus_cases dirs_out stems_out)
  set(dirs "")
  set(stems "")
  if(WM_CORPUS_DIR)
    file(GLOB found CONFIGURE_DEPENDS "${WM_CORPUS_DIR}/[0-9][0-9]_*")
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
wm_corpus_cases(wm_dirs wm_stems)
list(LENGTH wm_dirs wm_case_count)
if(WM_CORPUS_DIR AND wm_case_count EQUAL 0)
  message(FATAL_ERROR "WM_CORPUS_DIR=${WM_CORPUS_DIR} holds no NN_*/case.c; a corpus that builds nothing must not look like a corpus that passed")
endif()
if(wm_case_count GREATER 0)
  math(EXPR wm_last "${wm_case_count} - 1")
  foreach(i RANGE 0 ${wm_last})
    list(GET wm_dirs ${i} wm_dir)
    list(GET wm_stems ${i} wm_stem)
    if(PORT_HOSTED)
      # WM_SUBLET replaces only the hosted main() (shared/driver-virtual.c includes driver.c
      # unchanged), so every other build compiles driver.c exactly as before.
      set(wm_driver "${WM_CORPUS_DIR}/shared/driver.c")
      if(WM_SUBLET)
        set(wm_driver "${WM_CORPUS_DIR}/shared/driver-virtual.c")
      endif()
      add_executable("${wm_stem}" "${wm_dir}/case.c" "${wm_driver}")
      target_include_directories("${wm_stem}" PRIVATE "${WM_CORPUS_DIR}/shared")
      target_compile_definitions("${wm_stem}" PRIVATE WM_CORPUS_HOSTED)
      target_link_libraries("${wm_stem}" PRIVATE Wireshark::Wmem)
    else()
      wm_executable("${wm_stem}" ${WM_VARIANT} "${wm_dir}/case.c" "${WM_CORPUS_DIR}/shared/driver.c")
      target_include_directories("${wm_stem}" PRIVATE "${WM_CORPUS_DIR}/shared")
    endif()
  endforeach()
  message(STATUS "wmem defect corpus: ${wm_case_count} cases")
endif()
