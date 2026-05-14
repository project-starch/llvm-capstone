# Capstone testing matrix and current recommendations

This file is the compact map of which test layer to run for which kind of change.
It is intentionally shorter than the older narrative version.

## Setup once per shell

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
```

## Quick cheat sheet

```bash
# Backend / SelectionDAG regressions
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone"

# Clang builtins
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/capstone-builtins.c" \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/builtins-capstone.c"

# LLD / ELF emulation
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/lld/test/ELF/emulation-capstone.s"

# Linux driver regression
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/Driver/capstone-linux-toolchain.c"

# Runtime proofs
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-shared-region-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh"

# null_blk regressions
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-baseline.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-split-io.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh"
```

## Test layers

| Layer | What it proves | Run when | Entry point |
| --- | --- | --- | --- |
| Backend / SelectionDAG | codegen lowering and backend behavior | backend changes | `llvm/test/CodeGen/Capstone/` |
| Clang builtins | builtin lowering to expected IR/intrinsics | builtin/frontend target changes | `clang/test/CodeGen/capstone-builtins.c`, `clang/test/CodeGen/builtins-capstone.c` |
| LLD / ELF emulation | native `EM_CAPSTONE` emulation behavior | linker/emulation changes | `lld/test/ELF/emulation-capstone.s` |
| Linux driver | hosted driver link-line construction only | driver/sysroot logic changes | `clang/test/Driver/capstone-linux-toolchain.c` |
| Sample/runtime smoke | sample-domain path still works | sample/runtime packaging changes | `capstone/tests/runtime-qemu/run-smoke.sh` |
| Shared-region proof | shared-region mutations are visible again | region/runtime ABI changes | `capstone/tests/runtime-qemu/run-shared-region-probe.sh` |
| HostCall stdout proof | domain -> helper payload flow | HostCall metadata/output flow changes | `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh` |
| HostCall filewrite proof | same ABI reused for a second coarse service | HostCall service-family changes | `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh` |
| HostCall fileread proof | helper -> domain payload flow | reverse-direction payload changes | `capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh` |
| `null_blk` baseline | baseline block path still works | runtime/device baseline checks | `capstone/tests/runtime-qemu/run-nullblk-baseline.sh` |
| `null_blk` split | split I/O path and unload still work | OpenSBI/kernel/module integration changes | `run-nullblk-split-io.sh`, `run-nullblk-split-rmmod.sh` |

## Recommended minimums by change type

### Backend / Clang / LLD only

Run the focused `llvm-lit` layer that matches the modified subtree.
Do not jump straight to QEMU unless the change affects runtime-facing behavior.

### Userspace loader / helper / HostCall / runtime wrapper changes

Run at least:

```bash
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-shared-region-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh"
```

Then add the more specific wrapper that matches the changed service.

### OpenSBI / kernel / module integration changes

Run the runtime proofs plus the `null_blk` regressions.
If the active kernel changed, rebuild dependent modules/packages so their `vermagic` matches.

## Important limitations

- The Linux driver test is a command-line regression, not proof that hosted Capstone Linux userspace already works.
- The current validated path is still the split host/domain runtime path.
- `run-smoke.sh` is useful as a quick probe, but the HostCall wrappers and `null_blk` regressions are stronger current gates.
