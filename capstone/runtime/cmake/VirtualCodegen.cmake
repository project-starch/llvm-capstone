# Code generation for virtual C: every application, its runtime and builtins.
# capstone-cc reads the same list from sdk.json, and musl's build
# (ports/musl-capstone/survey-musl-capstone.py --profile virtual) passes the same flags.
# Virtual C runs the F and D extensions in hardware; QEMU's supervisor keeps
# each context's f registers and fcsr. The calling convention stays lp64:
# floating-point arguments and results travel in integer registers, as with
# soft float. This target does not accept -mabi=, hence -target-abi.
set(CAPSTONE_VIRTUAL_CODEGEN
  -mllvm -capstone-gp-free -mllvm -capstone-image-gp
  -Xclang -target-feature -Xclang +f -Xclang -target-feature -Xclang +d
  -Xclang -target-abi -Xclang lp64)
