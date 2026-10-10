# Included through CMAKE_USER_MAKE_RULES_OVERRIDE_C, which runs AFTER
# CMakeCInformation has chosen the object-file extension. It chooses `.obj`
# whenever UNIX is false, and UNIX is false for CMAKE_SYSTEM_NAME Generic, so
# the toolchain file cannot win that assignment -- this hook is where CMake
# documents the override. capstone-cc takes `.o`, `.a` and `.lo` as linker
# inputs and refuses anything else rather than guessing, so the names have to
# agree here instead of the driver being widened to accept whatever arrives.
set(CMAKE_C_OUTPUT_EXTENSION .o)
# ASM has its OWN extension variable and the same default, so a port that
# assembles a fixture -- ffmpeg's buffer-pool is the one in tree -- got a
# `.S.obj` that capstone-cc rejected at the link with `unsupported compiler
# driver argument`, after the C side had already been fixed. One file sets both,
# and the toolchain registers it for both languages.
set(CMAKE_ASM_OUTPUT_EXTENSION .o)
