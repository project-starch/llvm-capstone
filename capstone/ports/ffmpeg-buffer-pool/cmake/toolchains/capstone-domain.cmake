# Freestanding Capstone ABI: MUSL provides headers, but no libc is linked.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR capstone64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(FFPOOL_PLATFORM capstone-domain CACHE STRING "Execution platform" FORCE)

include("${CMAKE_CURRENT_LIST_DIR}/../Workspace.cmake")
get_filename_component(repo "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
ffpool_path(CAPSTONE_LLVM_BUILD_DIR "${repo}/llvm/cmake-build-debug" "Compiler build with the AS200 atomic fix")
ffpool_path(FFPOOL_MUSL "${tmp_root}/musl-src/musl-1.2.5" "Prepared Capstone MUSL headers")

set(CMAKE_C_COMPILER "${CAPSTONE_LLVM_BUILD_DIR}/bin/clang")
set(CMAKE_ASM_COMPILER "${CMAKE_C_COMPILER}")
set(CMAKE_C_COMPILER_TARGET capstone64-unknown-elf)
set(CMAKE_ASM_COMPILER_TARGET capstone64-unknown-elf)
set(CMAKE_AR "${CAPSTONE_LLVM_BUILD_DIR}/bin/llvm-ar")
set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_AR> s <TARGET>")
set(CMAKE_LINKER "${CAPSTONE_LLVM_BUILD_DIR}/bin/ld.lld")

if(NOT EXISTS "${FFPOOL_MUSL}/obj/include/bits/alltypes.h")
  message(FATAL_ERROR "Set FFPOOL_MUSL to a prepared MUSL tree with obj/include/bits/alltypes.h.")
endif()
execute_process(COMMAND "${CMAKE_C_COMPILER}" -print-resource-dir
  OUTPUT_VARIABLE resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

# Preserve both target-feature pairs; CMake's option deduplication must not
# merge the repeated -Xclang arguments. C11 atomics require +a and the AS200 fix.
set(CMAKE_C_FLAGS_INIT "-Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -nostdinc")
foreach(headers arch/capstone64 arch/generic obj/include include)
  string(APPEND CMAKE_C_FLAGS_INIT " -isystem \"${FFPOOL_MUSL}/${headers}\"")
endforeach()
string(APPEND CMAKE_C_FLAGS_INIT " -isystem \"${resource_dir}/include\"")
set(CMAKE_ASM_FLAGS_INIT "--target=capstone64-unknown-elf -ffreestanding")

# The custom clang driver has no hosted link environment for this ABI.
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CAPSTONE_LLVM_BUILD_DIR FFPOOL_MUSL)
