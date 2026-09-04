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
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-all.sh"

# null_blk regressions
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-all.sh"

# Benchmark regressions
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-coremark.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/run-all-beebs.sh"
# Opt-in faster full BEEBS gate after focused checks:
RUN_ALL_BEEBS_JOBS=8 bash "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/run-all-beebs.sh"
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
| HostCall file open/close proof | first helper-managed file-handle lifecycle path plus revoke-before-reborrow on a real service flow | handle-table or multi-request file-service changes | `capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh` |
| HostCall file handle write proof | first handle-based byte-movement path on top of helper-managed file tokens | handle-based file-service data-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh` |
| HostCall file handle read proof | first handle-based reverse-direction byte-movement path on top of helper-managed file tokens | handle-based file-service read-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh` |
| HostCall file handle sync proof | first handle-based durability-oriented path on top of helper-managed file tokens | handle-based file-service sync-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh` |
| HostCall file handle stat proof | first handle-based narrow metadata path on top of helper-managed file tokens | handle-based file-service stat-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh` |
| HostCall file handle truncate proof | first handle-based size-mutation path on top of helper-managed file tokens | handle-based file-service truncate-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh` |
| HostCall path access proof | first SQLite-facing path existence/access path on top of the current HostCall boundary | path-level SQLite/VFS-facing changes | `capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh` |
| HostCall path delete proof | first SQLite-facing path delete/unlink path on top of the current HostCall boundary | path-level SQLite/VFS-facing changes | `capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh` |
| HostCall combined file-object proof | first composed end-to-end file-object scenario across modular OPEN/WRITE/SYNC/CLOSE/READ operations | composed file-service behavior changes | `capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh` |
| Second-`PENDING` diagnostic | whether metadata-only multi-`PENDING` re-entry works | targeted runtime/control-flow diagnosis | `capstone/tests/runtime-qemu/run-hostcall-second-pending-probe.sh` |
| Second-`PENDING` payload-reuse diagnostic | whether reusing the same borrowed output payload across rounds triggers the current limitation | targeted runtime/ownership diagnosis | `capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-probe.sh` |
| Second-`PENDING` payload-reuse revoke diagnostic | whether explicit revoke before re-share satisfies the intended borrowed-region rule | targeted runtime/ownership diagnosis | `capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-revoke-probe.sh` |
| `null_blk` baseline | baseline block path still works | runtime/device baseline checks | `capstone/tests/runtime-qemu/run-nullblk-baseline.sh` |
| `null_blk` aggregate | baseline, split I/O, and split unload still work | OpenSBI/kernel/module/QEMU interrupt integration changes | `capstone/tests/runtime-qemu/run-nullblk-all.sh` |
| CoreMark CRC validation | all three algorithms (list, matrix, state machine) run and produce validated CRCs on Capstone PureCap with compiled C `domain_main` | backend codegen changes, CoreMark benchmark changes | `capstone/tests/runtime-qemu/run-coremark.sh` |
| BEEBS `fac` validation | first BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes | `capstone/benchmarks/beebs/run-beebs-fac.sh` |
| BEEBS `insertsort` validation | second BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-insertsort.sh` |
| BEEBS `fibcall` validation | third BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes | `capstone/benchmarks/beebs/run-beebs-fibcall.sh` |
| BEEBS `cnt` validation | fourth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-cnt.sh` |
| BEEBS `bubblesort` validation | fifth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` |
| BEEBS `prime` validation | sixth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-prime.sh` |
| BEEBS `recursion` validation | seventh BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-recursion.sh` |
| BEEBS `janne_complex` validation | eighth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` |
| BEEBS `tarai` validation | ninth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-tarai.sh` |
| BEEBS `cover` validation | tenth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-cover.sh` |
| BEEBS `duff` validation | eleventh BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-duff.sh` |
| BEEBS `levenshtein` validation | twelfth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` |
| BEEBS `jfdctint` validation | thirteenth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` |
| BEEBS `fdct` validation | fourteenth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-fdct.sh` |
| BEEBS `strstr` validation | fifteenth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, selected backend codegen changes | `capstone/benchmarks/beebs/run-beebs-strstr.sh` |
| BEEBS `qrduino` validation | fifty-fifth BEEBS benchmark builds and runs on the split host/domain path with a correctness marker | BEEBS benchmark changes, benchmark runtime wrapper changes, static-data capability handling | `capstone/benchmarks/beebs/run-beebs-qrduino.sh` |

The canonical complete BEEBS validation list is in `state/current-state.md`.
For backend/lowering/ABI changes, run all validated BEEBS wrappers rather than a
representative subset.

## Recommended minimums by change type

### Backend / Clang / LLD only

Run the focused `llvm-lit` layer that matches the modified subtree.
Do not jump straight to QEMU unless the change affects runtime-facing behavior.

For non-trivial backend/lowering/ABI changes, the full validation gate is:

```bash
"$CAPSTONE_LLVM_LIT" -sv "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-coremark.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/run-all-beebs.sh"
```

Smaller BEEBS subsets are appropriate only for narrow wrapper/doc changes or
quick pre-commit smoke checks.

`run-all-beebs.sh` is serial by default. Use `RUN_ALL_BEEBS_JOBS=N` for opt-in
parallel full gates; the aggregate gives each attempt an isolated build/share
workspace and retries only structured QEMU infra flakes that occur before
benchmark execution.

Run the BEEBS wrappers from the benchmark regression list above when changing the
BEEBS benchmark build/run path.

### Userspace loader / helper / HostCall / runtime wrapper changes

Run at least:

```bash
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-shared-region-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-all.sh"
```

Then add the more specific wrapper that matches the changed service.

### OpenSBI / kernel / module integration changes

Run the runtime proofs plus the `null_blk` regressions. If the active kernel
changed, rebuild dependent modules/packages so their `vermagic` matches.

For QEMU interrupt-delivery changes, include at least:

```bash
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-all.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-coremark.sh"
```

### Narrow runtime/QEMU capability-path diagnosis

When the question is specifically about repeated HostCall rounds, use:

```bash
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-second-pending-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-probe.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-revoke-probe.sh"
```

Interpretation in the current environment:

- metadata-only second `PENDING` works,
- reusing/re-sharing the same borrowed output payload across the next round without revoke reproduces the current `helper_csmrev` assertion,
- explicitly revoking that payload region before the second borrowed re-share succeeds,
- this matches the intended runtime rule that an already borrow-shared region must be revoked before it is reused or re-shared.

## Runtime image behavior

The QEMU smoke harness uses snapshot mode so guest writes are discarded and repeated
runtime tests do not mutate the generated Buildroot `rootfs.ext2` image. Buildroot
getty is pinned to `ttyS0`, matching the active QEMU serial console, and
the harness forces QEMU `-smp 1` for deterministic boot progress.

## Important limitations

- The Linux driver test is a command-line regression, not proof that hosted Capstone Linux userspace already works.
- The current validated path is still the split host/domain runtime path.
- `run-smoke.sh` is useful as a quick probe, but the HostCall wrappers and `null_blk` regressions are stronger current gates.
