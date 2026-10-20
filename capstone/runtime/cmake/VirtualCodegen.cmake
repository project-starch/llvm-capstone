# Code generation for virtual C: every application, its runtime and builtins.
# capstone-cc reads the same list from sdk.json, and musl's build
# (ports/musl-capstone/survey-musl-capstone.py --virtual) passes the same flags.
# Virtual C runs the F and D extensions in hardware; QEMU's supervisor keeps
# each context's f registers and fcsr. The calling convention stays lp64:
# floating-point arguments and results travel in integer registers, as with
# soft float. This target does not accept -mabi=, hence -target-abi.
# +movc-keeps-integer-source: capstone-qemu's MOVC keeps an untagged source
# (Q-04, decided 2026-10-14), so the C-32 STC+LDC rewrite of live-source copies
# is off. The physical profile keeps it until the bitstream has Q-04 (b).
# CAPSTONE_VIRTUAL_MOVC_KEEP (cache or environment, default ON) = OFF/0 writes
# -movc-keeps-integer-source instead: the rewrite is back on, for images bound for
# a bitstream WITHOUT Q-04 (b), whose MOVC nulls an untagged source (ISSUES C-32;
# the lead's remedy (b), 2026-10-20). musl's build reads the same variable.
if(NOT DEFINED CAPSTONE_VIRTUAL_MOVC_KEEP)
  if(DEFINED ENV{CAPSTONE_VIRTUAL_MOVC_KEEP})
    set(CAPSTONE_VIRTUAL_MOVC_KEEP "$ENV{CAPSTONE_VIRTUAL_MOVC_KEEP}")
  else()
    set(CAPSTONE_VIRTUAL_MOVC_KEEP ON)
  endif()
endif()
if(CAPSTONE_VIRTUAL_MOVC_KEEP)
  set(CAPSTONE_VIRTUAL_MOVC_FEATURE +movc-keeps-integer-source)
else()
  set(CAPSTONE_VIRTUAL_MOVC_FEATURE -movc-keeps-integer-source)
endif()
# CAPSTONE_VIRTUAL_PAD_GLOBALS (cache or environment, default OFF) = ON adds
# -mllvm -capstone-pad-globals: every defined global of 4 KiB or more is aligned
# and tail-padded to exactly representable bounds, and every module narrows a
# global to that length (isa.md, SHRINK's result on the compact profile, rule
# (2)). It must be uniform across an image, so it lives here, where the runtime,
# the builtins, sdk.json (capstone-cc, every application) and musl's build all
# read it: a module built with it narrows a global it only declares to the
# padded length, which covers the next object if the definer was built without.
if(NOT DEFINED CAPSTONE_VIRTUAL_PAD_GLOBALS)
  if(DEFINED ENV{CAPSTONE_VIRTUAL_PAD_GLOBALS})
    set(CAPSTONE_VIRTUAL_PAD_GLOBALS "$ENV{CAPSTONE_VIRTUAL_PAD_GLOBALS}")
  else()
    set(CAPSTONE_VIRTUAL_PAD_GLOBALS OFF)
  endif()
endif()
set(CAPSTONE_VIRTUAL_PAD_GLOBALS_FLAGS)
if(CAPSTONE_VIRTUAL_PAD_GLOBALS)
  set(CAPSTONE_VIRTUAL_PAD_GLOBALS_FLAGS -mllvm -capstone-pad-globals)
endif()
set(CAPSTONE_VIRTUAL_CODEGEN
  -mllvm -capstone-gp-free -mllvm -capstone-image-gp
  -Xclang -target-feature -Xclang +f -Xclang -target-feature -Xclang +d
  -Xclang -target-abi -Xclang lp64
  -Xclang -target-feature -Xclang ${CAPSTONE_VIRTUAL_MOVC_FEATURE}
  ${CAPSTONE_VIRTUAL_PAD_GLOBALS_FLAGS})
