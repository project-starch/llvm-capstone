# Capstone testing matrix and current recommendations

Application benchmark study: the [Sublet/PoisonCap design](../plans/sublet-poisoncap-memory-study.md)
uses two matched pairs for the nested boundary; default CheriBSD on/off remains
separate reference data. The [planner](../../experiments/study/README.md) supports
PoisonCap plans but blocks execution qualification pending observed inner-policy
accounting. Twenty host checks pass. Upstream mruby lists passes 4/4 original
arms; both PoisonCap SQLite modes complete the artifact's 20 active phases at
size 1. Twelve phases are commented out in that artifact, and the main
result oracle is missing. These are readiness results, not a memory ranking.

Application memory behavior: [twelve paired workload configurations](../../experiments/applications/results/20260927-reuse/README.md)
pass 72/72 attempts. Every recorded Capstone memory phase matches older-QEMU
controls without in-process collection. The results quantify prompt address
reuse and post-release retention, include the large-retained-graph counterexample,
and make no timing or total-RSS claim.

Application memory: [Capstone ports versus default CheriBSD](../../experiments/applications/comparison.md) now covers
FFmpeg and mruby with common allocation counters. The original matrix recorded
six Capstone node-capacity failures. The [QEMU node-reuse follow-up](../../runtime/tests/application/results/20260927-node-reuse/README.md)
passes all 27 Capstone repeats at the same 65,536-node capacity, including all six
previous failures, using unchanged application binaries. Keep the original data
and larger-node controls separate. These are memory observations, not timings.

2026-09-26 application-platform run: the installed managed guest passes the common
acceptance (including exhaustion followed by 1,008 starts). Legacy CoreMark,
shared-region, stdout/filewrite/fileread pass. The available snapshot fails
null_blk (null_submit_bio, bad address 0x6f) and file-open-close (borrowed-region
INIT, cause 29) on both the original and new platforms. These remain baseline
failures; the historical rows below are not a claim that every gate passed in
this run. See [current state](../state/current-state.md).

Allocator trace tooling: [formats, CLI, validation scope and adapter tests](../../ports/common/host/port_trace/README.md).

This file is the compact map of which test layer to run for which kind of change.
It is intentionally shorter than the older narrative version.

Perl's [complete upstream `t/base` run](../../ports/perl/musl/results/2026-09-26/base-tests.txt)
uses host `prove --exec` with `capstone-vm run`: 6/9 files pass; `term.t` has one
failed assertion and `lex.t`/`rs.t` each terminate by SIGSEGV before TAP. This
is a port compatibility gate, not a passing full Perl-suite result.

Shared application runtime: run the native startup/image/stream tests and host
CLI tests, then the [persistent-guest application gate](../../runtime/applications.md#verification).
It checks actual waitpid signals, no-yield and blocked-I/O cancellation,
concurrent ownership, dup/fork/VMA lifetime, rollback, memory scrubbing, two real
interpreters and an unchanged boot ID. `--repeat 200` adds 1,008 mixed starts with
stable resource counters; `--sublet-image` adds transferred-heap reclamation,
200,000-cycle in-process reuse, stale-reference rejection after reuse and
recoverable genuine node exhaustion. It does not add
fork/threads inside a domain or constitute FPGA validation.

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

# Runtime proofs: the bare HostCall wire probes (each boots its own QEMU)
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-hostcall-all.sh"

# Runtime probes as delegated applications, in a running capstone_vm guest
python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-delegated-probes.py" \
  --sdk <application SDK> --work <dir> --state <capstone_vm state>

# null_blk regressions
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-nullblk-all.sh"

# Benchmark regressions
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-coremark.sh"
bash "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/run-all-beebs.sh"
# Opt-in faster full BEEBS gate after focused checks:
RUN_ALL_BEEBS_JOBS=8 bash "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/run-all-beebs.sh"
```

## Test layers

PostgreSQL allocator changes use the native and capstone-domain CTest suites in
`ports/postgres/memory-contexts`. They cover AllocSet plus Generation, Slab and
Bump, paired lifetime faults and mixed-manager replay. See the component README
for build paths, release-layout restrictions and artifact retention.

| Layer | What it proves | Run when | Entry point |
| --- | --- | --- | --- |
| Backend / SelectionDAG | codegen lowering and backend behavior | backend changes | `llvm/test/CodeGen/Capstone/` |
| Clang builtins | builtin lowering to expected IR/intrinsics | builtin/frontend target changes | `clang/test/CodeGen/capstone-builtins.c`, `clang/test/CodeGen/builtins-capstone.c` |
| LLD / ELF emulation | native `EM_CAPSTONE` emulation behavior | linker/emulation changes | `lld/test/ELF/emulation-capstone.s` |
| Linux driver | hosted driver link-line construction only | driver/sysroot logic changes | `clang/test/Driver/capstone-linux-toolchain.c` |
| Sample/runtime smoke | sample-domain path still works | sample/runtime packaging changes | `capstone/tests/runtime-qemu/run-smoke.sh` |
| QEMU runner self-test | `run-domain-smoke.py` itself: an image with an undefined weak symbol is refused before boot; output containing `# ` is not taken for the prompt; a guest command over 1 KiB runs whole; the previous runner (pinned) must fail the same boot | changes to `run-domain-smoke.py` | `capstone/tests/runtime-qemu/run-domain-smoke-selftest.sh` |
| Shared-region proof | shared-region mutations are visible again | region/runtime ABI changes | `capstone/tests/runtime-qemu/run-shared-region-probe.sh` |
| HostCall stdout proof | domain -> helper payload flow | HostCall metadata/output flow changes | `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh` |
| HostCall filewrite proof | same ABI reused for a second coarse service | HostCall service-family changes | `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh` |
| HostCall fileread proof | helper -> domain payload flow | reverse-direction payload changes | `capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh` |
| HostCall file open/close proof | first helper-managed file-handle lifecycle path plus revoke-before-reborrow on a real service flow | handle-table or multi-request file-service changes | `capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh` |
| HostCall file handle write proof | first handle-based byte-movement path on top of helper-managed file tokens | handle-based file-service data-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh` |
| HostCall file handle read proof | first handle-based reverse-direction byte-movement path on top of helper-managed file tokens | handle-based file-service read-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh` |
| HostCall file handle sync proof | first handle-based durability-oriented path on top of helper-managed file tokens | handle-based file-service sync-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh` |
| MOVC of an integer, as the RTL does it | capstone-qemu with `CAPSTONE_MOVC_NULL_SCALAR=1` nulls a non-capability MOVC source as the RTL does (probe `b=5 c=0`, against `b=5 c=5` by default), and reports whether the compiler's C-32 shape loses its pointer to it; a delegated application, run once in a guest started with the switch off and once with it on | capstone-qemu MOVC changes; C-32 and other register-copy codegen changes | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only movc` |
| MOVC exposure: what zeroing an integer MOVC source changes | the nightly, each bare hostcall probe, the delegated runtime probes and the delegated libc-test with `CAPSTONE_MOVC_NULL_SCALAR` off and on, every verdict compared; exits 1 on a difference, 2 if an arm did not measure | a QEMU or compiler change that could move the answer to Q-04 | `capstone/tests/runtime-qemu/movc-null-scalar/exposure.sh` |
| CINCOFFSET and SCC on an integer | case 0 (both on a capability) works, cases 1 and 2 end in SIGSEGV with fault cause 24 today, or give the integer result under `--arith-expect cheri` | the SCC/CINCOFFSET decision; capstone-qemu cap-arithmetic helpers | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only arith-0 arith-1 arith-2` |
| HostCall file handle stat proof | first handle-based narrow metadata path on top of helper-managed file tokens | handle-based file-service stat-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh` |
| HostCall file handle truncate proof | first handle-based size-mutation path on top of helper-managed file tokens | handle-based file-service truncate-path changes | `capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh` |
| HostCall path access proof | first SQLite-facing path existence/access path on top of the current HostCall boundary | path-level SQLite/VFS-facing changes | `capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh` |
| HostCall path delete proof | first SQLite-facing path delete/unlink path on top of the current HostCall boundary | path-level SQLite/VFS-facing changes | `capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh` |
| HostCall combined file-object proof | first composed end-to-end file-object scenario across modular OPEN/WRITE/SYNC/CLOSE/READ operations | composed file-service behavior changes | `capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh` |
| Delegated large I/O | an application reads and writes a 64 KiB file on the 9p share, whole and in odd pieces, through the launcher's bounce buffer; a 5000-byte stdout line with the launcher's stdout on a 9p file | delegated read/write path, launcher bounce buffer | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only large-read big-stdout` |
| Delegated exit | `exit()` ends with the right status and flushed stdio, with and without a `__capstone_at_exit` hook (7, and 42 through the hook), and the `atexit` handler runs through a tagged pointer | runtime exit path, weak-symbol or `atexit` override changes | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only exit-default exit-hook` |
| Delegated return | a program that returns from `main` without `exit()` still delivers its buffered stdout and runs its `atexit` handlers, with the returned status 5 | runtime `domain_main` / exit-path changes | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only return-flush` |
| __thread in an application (C-47): local-exec codegen, the TLS segment, and the runtime's block; with an overrun control | 9 checks at -O0 and -O2: initial values, .tbss, 64/4096 alignment, a second unit, a kept capability, bounds, errno; the overrun must end in SIGSEGV with a fault record | compiler codegen or musl-capstone runtime changes | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only tls-O0 tls-O2 tls-overrun` |
| Delegated constructors | constructors (with and without a priority) run before `main` and destructors at exit, printing exactly what the same file prints natively, one of them reading the launcher's environment | runtime `domain_main` / exit-path changes, `my_first_domain/link.ld` array placement | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only init-fini` |
| Delegated unserved report | the runtime's unserved-syscall line (`214x2`, two `brk`) reaches the task's stderr from a program that closed fd 1 | runtime unserved report | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only unserved-report` |
| Delegated mmap and System V shared memory | mmap/munmap/shm* from the domain's allocator (`mmap_shm_level0.c`), with the refusals' errnos; the control asks the syscall layer for the same mapping and must get ENOSYS and an unserved report | runtime mmap/shm override, the first-fit heap (`level0.c`) | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only mmap-shm mmap-shm-control` |
| Capability-valued and sub-word atomics in an application (C-54, C-51) | pointer atomics through the runtime's generic `__atomic_*` come out tagged; 8/16-bit atomics land on their lane; each at -O0 and -O2 | atomic codegen, `atomic_libcalls.c` | `capstone/tests/runtime-qemu/run-delegated-probes.py` `--only cap-atomics-O0 cap-atomics-O2 subword-O0 subword-O2` |
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

A change to the musl application runtime (`ports/musl-capstone/runtime`, the delegated
runtime, the only one) runs `run-delegated-probes.py`, the delegated libc-test
(`ports/musl-capstone/libc-test/run-libc-test-delegated.py`) and the application gate
(`runtime/tests/application/run.py`) in a capstone_vm guest.

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
