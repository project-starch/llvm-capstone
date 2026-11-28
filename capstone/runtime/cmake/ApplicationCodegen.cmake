# Code generation for application compiles only: capstone-cc reads it from sdk.json's codegen, after the profile's
# (VirtualCodegen.cmake). The SDK's runtime, builtins and libc are built without it, as the measured SDKs were.
#
# CAPSTONE_APPLICATION_REDUNDANT_BOUNDS (default ON, programme item C5): drop a global's SHRINKTO / SHRINK
# narrowing where every access through it is in bounds at a constant offset, also for a global the unit only
# declares. The declared-size case records its assumption in .capstone.size_assume, and capstone-cc's link checks
# every record against the image's definitions (compiler.py check_size_assumptions). An object linked by ld.lld
# directly skips that check, so a declared size that disagrees with the definition would go unnoticed there.
#
# CAPSTONE_APPLICATION_MEASURED_CODEGEN (default OFF): the rest of the paper's Full arm (FRCDKMSa3) application
# flags, for SPEC and kit builds. With it, a compile with no codegen flags of its own is the Full arm's compile.
# Making it every port's default is the lead's decision (ports/common/application/build-sdk.sh).
option(CAPSTONE_APPLICATION_REDUNDANT_BOUNDS "Applications drop provably redundant global narrowing (C5)" ON)
option(CAPSTONE_APPLICATION_MEASURED_CODEGEN "Applications compile with the Full arm's flags" OFF)
set(CAPSTONE_APPLICATION_CODEGEN)
if(CAPSTONE_APPLICATION_MEASURED_CODEGEN)
  list(APPEND CAPSTONE_APPLICATION_CODEGEN
    -Xclang -target-feature -Xclang +relax -Xclang -target-feature -Xclang +c
    -Xclang -target-feature -Xclang +cap-base-rvc
    -mllvm -capstone-shrinkto-globals -mllvm -capstone-shrinkto-end-adjust=1 -mllvm -capstone-direct-calls
    -fomit-frame-pointer -fjump-tables
    -mllvm -capstone-fold-global-lo -mllvm -capstone-disable-cost-per-use -mllvm -capstone-memcpy-align-dispatch)
endif()
if(CAPSTONE_APPLICATION_REDUNDANT_BOUNDS)
  list(APPEND CAPSTONE_APPLICATION_CODEGEN
    -mllvm -capstone-drop-redundant-shrink -mllvm -capstone-drop-redundant-shrink-declarations)
endif()
