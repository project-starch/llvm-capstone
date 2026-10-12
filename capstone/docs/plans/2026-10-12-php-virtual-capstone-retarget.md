# Merge `origin/dev` and re-target the PHP port to virtual Capstone

## Context

The PHP 5.0.0 port runs as a **physical Capstone domain**: a monitor-created domain over a
contiguous physical region, with a hard ceiling of `code + .bss + declared ≤ ~4 MiB`.
Everything distinctive about the port is a consequence of that ceiling and of having no OS
underneath:

- `main/` (26 TUs, 12,828 lines — PHP's runtime host) is **not linked**; a ~200-line SAPI
  inside `engine/rung_E_domain.c` stands in for it.
- A 48-symbol freestanding libc (`engine/libc/`, ~1,000 lines) and **29 stub headers**
  (`engine/stubinc/`) stand in for the C library.
- No streams, sockets, files, `environ` or real `time()`.
- `-mllvm -capstone-shrink-stack=false` is forced *by the ceiling*
  (`engine/build-engine.sh:49-71`): narrowing costs ~33% of `.text` and does not fit. That
  deletes stack-object bounds, so the port cannot catch a stack defect at all.
- `PHP_DOMAIN_DATA` had to drop 2 MiB → 1.5 MiB the moment the pure `ext/standard` TUs were
  linked (image 1.4 MB → 2.35 MB), and past the ceiling the **monitor itself faults** rather
  than refusing cleanly.
- Every case needs its own guest boot (~30 s) plus a 3-retry budget for lost boots.

The four known porting gaps were *all* in that hand-written substitute layer, never in
`Zend/` or `ext/` (which are byte-identical corpus code): a sink that discarded its bytes,
an error path that discarded its format string, a missing
`zend_register_default_classes`, a missing `zend_deactivate`.

Upstream has since produced **virtual Capstone**: capability applications run as ordinary
Linux user processes — C mode, user virtual addresses, Linux owns page tables, scheduling
and syscalls; a loadable module plus `CSRUNV` connects the two. `origin/dev` pins
`capstone/capstone-qemu` at `1a6dd20732`, the tip of `origin/virtual-capstone`.

Retargeting removes the ceiling *and* supplies an OS — which is precisely what those gaps
were made of.

## What it buys, concretely

The experiment's population is the **31 ASAN-blind cases**: those ASAN detects *only* when
PHP's nested allocator is patched (`ZEND_DISABLE_MEMORY_CACHE 0→1`,
`REAL_SIZE(size)→(size)`). Reproducible from the corpus:

```
results-asan-default.json  outcome != OK    AND    results-asan-stockalloc.json  outcome == OK
→ 31 cases      (23 hidden by the size-class cache, 8 by REAL_SIZE rounding)
```

24 are reachable today and all 24 are caught. **All 7 that are not are blocked by exactly
what a Linux process provides:**

| case | blocked on | class |
|---|---|---|
| CRASH-008 | `ext/standard/datetime.c`, real `time()` | CWE-787 |
| CRASH-084 | `putenv()` / `environ` | CWE-416 |
| CRASH-086 | tick functions (`basic_functions.c`) — image size | CWE-416 |
| CRASH-099 | `stream_select`, `php://memory` | CWE-416 |
| CRASH-112 | stream filters, writable `/tmp` | CWE-416 |
| CRASH-092 | loopback TCP listener | CWE-416 |
| CRASH-100 | HTTP listener **and** `stack-use-after-return` | CWE-562 |

So the target moves **24/31 → 31/31**, and CRASH-100 adds a defect *class* the physical port
structurally cannot see, because `shrink-stack` can go back on once the ceiling is gone.

Three further wins fall out of the profile change:

- **No per-case boot.** `capstone/ports/common/application/run.py` runs images against a
  *persistent* VM (`--state`, `--stdin`, `--result`), so the boot-flake retry loop in
  `run_arm()` disappears along with ~30 s × 2 arms × 31 cases.
- **A trigger is a file on stdin**, not a C string compiled into the image. Arms stop
  differing by image content for reasons unrelated to the variable under test.
- **PHP's real `main/` and `sapi/cli`** become linkable, so the port stops standing in for
  ~20,000 lines of PHP with ~200 lines of its own.

## Decisions taken

1. **Scope: through Phase B.** Prove the profile, then build real PHP and go after all 31.
2. **Keep the physical arm as a comparison column.** `capstone/ports/php/engine/` stays
   working and untouched; the virtual port is new code beside it.
3. **Adopt the shared judge**, `capstone/bug-corpora/tools/verdicts.py`.

## Merge surface (verified, not taken on trust)

- `git merge-base php-5 origin/dev` = `5f0df698` (2026-09-20). We are **31 ahead**, dev is
  **980 ahead**. Our side is 121 files / +11,497: `ports/php/` (72), `tests/cheri-php-blind/`
  (38, new dir), `container/` (9, new dir), the beebs tag-preserving memcpy (1), and a
  3-file LLVM backport.
- **`git merge-tree` emits zero conflict hunks.** Only three files changed on both sides —
  all three `llvm/lib/Target/Capstone/*.cpp` — and all three are our own backport of commits
  that already landed on dev, so git resolves to dev's version and our edit is subsumed.
- `capstone/capstone-qemu`: our pointer `deb7d757` is an **ancestor** of dev's `1a6dd20732`
  (0 behind, 46 ahead). Pure fast-forward; our four local QEMU commits (badaddr assignment,
  cause-6 misaligned store, four corrected PC sources, rev-node watermark) are already
  contained. `caplifive-buildroot` and `capstone-ariane` likewise fast-forward.
- The PHP corpus is **outside** the repo (`../case-studies/php-5.0.0-bug-corpus`, mounted
  read-only at `/corpus`), so the merge cannot disturb it.
- Working tree clean; branch `php-5`, 16 commits unpushed.

**The merge is clean. All the work is downstream of it.**

### Three post-merge breakages to expect

1. `tests/runtime-qemu/run-domain-smoke.py` gained a **hard undefined-weak preflight**
   (C-56) that refuses any `.dom` referencing an undefined weak symbol, before booting.
   Our port declares no weak symbols, but it links compiler-rt builtins and a hand-rolled
   libc. Check with `llvm-nm --undefined-only` before the first run;
   `CAPSTONE_ALLOW_UNDEFINED_WEAK=1` is the escape hatch. **This is the single most likely
   thing to break the physical comparison column.**
2. `run-triggers.sh:57` uses a bare `flock "$CAPSTONE_QEMU_LOCK"`. Dev added
   `capstone_with_qemu_lock()` to `capstone-test-env.sh` precisely because the bare form
   self-deadlocks under the nightly. One-line fix.
3. `domreq.S` and `gct-section-end.S` are now two-line `#include` shims onto
   `capstone/runtime/domain/`. Our build compiles them with `clang -c` on an uppercase `.S`,
   so the preprocessor runs and the quoted include resolves — these keep working, but
   `link.ld` gained PT_TLS, `.capstone_application`, `.capstone_virtual` and
   `SORT_BY_INIT_PRIORITY`, so the build's "domreq moved a loaded byte" assertion now runs
   against a script with more sections.

LLVM (106 files changed) and QEMU (46 submodule commits) both need a full rebuild before any
result is trustworthy.

**CORRECTION to the figure this plan was written with.** Planning said "nearly fresh, 2 pending
ninja steps". That reading was void: `ninja -n` had a pending *CMake regeneration* as its only
step and cannot enumerate the real work until that runs, and the build tree is configured
against `/work/llvm-capstone` (the container path), so a host-side invocation resolves nothing
either. Measured properly -- regenerate first, then count, inside the container -- the figure is
**4,425 steps: 3,532 compiles and 297 links**. `LLVM_PARALLEL_LINK_JOBS` has been dropped from
2 to 1, because a RelWithDebInfo link can take several GB and this host has no swap.

**And the rebuild is a hard prerequisite, not a hygiene step.** Probed with both controls (the
known-good `-capstone-shrink-stack=false` accepted, a nonsense flag rejected):

| flag | current clang |
|---|---|
| `-capstone-shrink-stack=false` | accepted (positive control) |
| `-capstone-gp-free` | accepted |
| **`-capstone-image-gp`** | **REJECTED** |

`-capstone-image-gp` is one of the two flags `runtime/cmake/Application.cmake:42,57` makes
mandatory for the virtual profile, so nothing on that profile -- not even the seam probe --
can be built until LLVM is rebuilt.

## Resource budget — this machine cannot take a wide build right now

Measured at the time of writing: **8 cores, 31 GB RAM with ~10 GB available, no swap, load
average 6.5.** And the machine is already busy with work from another project — four
concurrent OpenROAD synthesis containers (`tile_base_sb4/sb16`,
`tile_rfe_opt2_sb4/sb16`, ~35 min in) plus a 3.3 GB `lean` process. With no swap, an OOM is a
hard kill, and it would take those synthesis runs with it.

`CLAUDE.md`'s `ninja -j90` convention is for a 90-core host and **must not be applied here**.
Rules for this plan:

- **Check before every heavy build.** `nproc`, `free -g`, and a scan for `qemu-system`,
  `podman`/`conmon` and `ninja` — the foreign synthesis jobs are *not* ours to kill, so the
  build waits or runs narrow.
- **Job counts:** `ninja -j4 -l8` for LLVM with `LLVM_PARALLEL_LINK_JOBS=1` (debug-info link
  steps are the real OOM risk, several GB each); `JOBS=4` for the musl SDK, against
  `build-sdk.sh`'s default of 16 for 1,361 TUs; `-j4` for QEMU. Keep the PHP TU loop
  **sequential**, as `build-engine.sh` already is.
- **Never hold a VM and a build at once.** The persistent VM is `-m 2G` plus
  `--cma-mib 1024 --process-cache-mib 768` ≈ 4 GB resident. Bring it `down` before any
  rebuild; that plus a 4-way LLVM build would exceed what is free today.
- **One VM, one QEMU.** The shared `rootfs.ext2` write-lock already forbids two suites in
  parallel; use `capstone_with_qemu_lock` rather than a bare `flock` (see breakage 2 above).
- **No parallel subagents that build.** Exploration can fan out; compilation cannot. A single
  owning session runs every build step in sequence.
- **Prefer the narrow path first.** The seam probe is ~50 lines and needs only the SDK; it can
  be built and run long before the full LLVM rebuild is affordable.

Practical consequence for sequencing: the LLVM + QEMU rebuild in Phase 0 should wait for the
synthesis containers to finish, or be run at `-j2`. It is the single largest resource item in
the plan and it blocks nothing else in Phase A except the final link.

## Host prerequisites (checked after the merge)

Everything the virtual profile needs is present on this machine, which was not a given:

| need | where | state |
|---|---|---|
| riscv64 Linux cross gcc (`CROSS_COMPILE`) | `caplifive-buildroot/build-qemu/host/bin/riscv64-buildroot-linux-gnu-` | gcc 12.3.0 |
| prepared kernel build (`KERNEL_BUILD`) | `caplifive-buildroot/build-qemu/build/linux-custom` | Linux 6.x, has `.config`, `Module.symvers`, `vmlinux`, `autoconf.h` |
| firmware / kernel / rootfs for a VM | `caplifive-buildroot/build/images` | `fw_jump.elf`, `Image` (19.8 MB), `rootfs.ext2` (2 GB), built 2026-09-21 |
| QEMU | `capstone-qemu/build/qemu-system-riscv64` | built 2026-09-20 from the **old** pointer -- predates `virt-capstone` and `x-capstone-u-mode`, so it must be rebuilt |
| container image | `localhost/capstone-build` | 1.77 GB, present, but its **cmake 3.22.1 is too old** -- see below |
| `cmake` >= 3.25 | in the container | **was missing.** `capstone/runtime/application/CMakeLists.txt:1` requires 3.25 and `ports/common/application/build-sdk.sh:15` passes `--fresh` (3.24+); jammy ships 3.22.1, so the SDK could not be configured at all. Fixed by installing Kitware's pinned 3.31.6 into `/opt/cmake` in `container/Containerfile`, matching the host's version |
| `-mllvm -capstone-image-gp` | the rebuilt clang | **accepted now; it was rejected before the rebuild.** This was the hard prerequisite -- verified with the known-good `-capstone-shrink-stack=false` as a positive control and a nonsense flag as a negative control, because a bare `-help-list-hidden` grep had previously reported a false zero for all three flags |

Neither the host nor the container has a cross gcc on `PATH`; it has to be taken from the
buildroot tree above. The container has no `flex`/`bison`, which is why a real PHP `configure`
cannot run inside it -- Phase B reuses the already-configured corpus tree instead, which it was
going to do anyway.

One consequence to decide before Phase A's run step: the merge moved the `caplifive-buildroot`
pointer to `8fd1ea1249a9` while the worktree -- and the images above -- are from `d04bd83b13cd`.
Updating that submodule invalidates the images and costs a buildroot rebuild, so leave it until
something actually needs the newer kernel module.

## Seam probe — RESULT (run 2026-10-12, three independent boots, identical readings)

`ports/php/zend-alloc/sublet/probe-sublet-split.c`, built against the virtual SDK and run as a
Linux user process in the guest. **The seam design is confirmed.** All six stages read as
predicted, and the control passed, so the other five are evidence rather than noise.

| stage | tests | reading |
|---|---|---|
| 0 | **positive control**: split at `base+48`, retire payload, write header, re-issue, senior revoke | **PASS, completed** -- "header still writable after the payload was retired" |
| 1 | write through a retired payload alias | **FAULT** cause 24 (`cincoffset with an UNTAGGED rs1`) |
| 2 | read through a retired payload alias | **FAULT** cause 24 |
| 3 | the `AG(cache)` case: re-issue the region, then use the OLD alias | **FAULT** cause 24 -- re-issue does **not** resurrect the previous tenant's pointer |
| 4 | read the *cursor* of a retired capability, no dereference | **SURVIVED** |
| 5 | `sublet_split` at `base+40`, not a 16-byte multiple | **PERMITTED**, bounds exact |

What this settles, in the plan's own terms:

- **The design claim holds.** Stage 0 is the whole seam in miniature: header and payload can be
  separated so that retiring one leaves the other live. So `zend_alloc.c` can keep reading
  `p->size` and writing `p->cached` after the logical free, and **the physical port's
  unlink-before-revoke deviation is not needed** -- the thing that cost "four false CAUGHT
  verdicts" there does not have to be repeated here.
- **Stage 3 is the one that matters for PHP**, because `AG(cache)` reuse is exactly the trap
  `docs/design/virtual-capstone/ownership.md` warns about: reusing bytes inside a still-live
  outer allocation does not by itself give inner temporal protection. A fresh `sublet_take` over
  the same bytes *does* kill the old alias, so retiring at `_efree` (above PHP's cache) gives
  per-object lifetimes **with `zend_alloc.c` still byte-identical**. That is the combination the
  physical port could only have one half of at a time.
- **Open risk 2 is resolved, and in the harder direction.** A cursor read on a dead capability
  does **not** trap (stage 4). So `ZEND_CAP_HEADER(ptr)` will *not* fault on a double-`efree`:
  the adapter must probe the record explicitly and fault at a named label of its own. This is
  the better outcome -- it means the adapter chooses where that fault lands, instead of it
  landing inside the index lookup where the harness would refuse it as a port artefact.
- **No padding is required**, and the limit of that reading is recorded in the probe itself:
  stage 5 exercises the *split* only. `sublet_give_to` fills in 16-byte steps, so revoking a
  non-grain region remains untested. PHP never needs it -- `sizeof(zend_mem_header)` is 48, a
  grain multiple -- so the adapter will not split off-grain and the question stays closed by
  construction rather than by measurement.
- **Node cost, first real measurement.** Stage 0's `CAPSTONE_VM_STATS`: `nodes=50`,
  `node_capacity=1024`, `node_growths=0` for one block's full lifecycle. Far from the table
  pressure the plan budgets for, but it is one block; the figure to watch is PHP's high-water,
  and the plan's fail-closed rule on cause 30 stands.

Two platform facts also came out of this, both now fixed in the code rather than in prose:

- **`capstone_vm` cannot provision a guest on this project.** Its guest setup runs `dropbearkey`
  unconditionally and `BR2_PACKAGE_DROPBEAR` is unset in our pinned buildroot *and* in dev's.
  The plan's `capstone_vm up --profile virtual` + `ports/common/application/run.py` route is
  therefore unavailable. `runtime/virtual/run.py` uses no SSH at all, so `app/run-virtual.py`
  borrows that mechanism instead: `-serial stdio`, a staged read-only ext4 disk, console
  driving, `VM_EXIT:<name>:<code>` markers, `-snapshot` so the shared 2 GB rootfs is never
  written. The batching benefit survives -- many `capstone-vexec` runs inside one boot, since a
  capability fault kills the child and not the guest -- it just arrives by a different route.
- **The QEMU lock must not be taken twice.** `run-virtual.py` takes it itself, so wrapping it in
  `capstone_with_qemu_lock` deadlocks until the outer timeout, which on the console is
  indistinguishable from a guest that never booted. It now takes the lock `LOCK_NB` and exits
  with an explanation instead of hanging.

## Merge regression gate — RESULT (2026-10-12): 23 of 24 CAUGHT, 1 NO-READING

Re-measured after the LLVM and QEMU rebuilds, which is what the merge owed. The physical arm is
run by two suites, not one: `blind` (11 cases) and `pureext` (13). Together they are the 24.

| suite | cases | result |
|---|---|---|
| `blind` | 11 | **11 CAUGHT**, rc=0 |
| `pureext` | 13 | **12 CAUGHT**, 1 NO-READING, rc=75 |

The 12: CRASH-016, 022, 035, 047, 066, 074, 080, 091, 102, 103, 151, 160. So **23/24 confirmed**,
and the previous "24/24" -- always provisional pending this re-run -- does not survive it.

### CRASH-057 is a NO-READING (`setup-fault`), and the cause is the port's own deviation

The control arm faults `cause=24, pc=0x101cecc30`, which symbolises to
`zend-alloc/zend_capstone_alloc.h:547` -- the **hoisted, unconditional**
`REMOVE_POINTER_FROM_LIST(p)`. The control is built `-DZEND_TEMPORAL -DZEND_NO_REVOKE`, so it
takes that path, and that file's own comment names this symptom exactly: *"a cause-24 whose
faulting instruction is a capability load through a pointer that was read out of freed memory."*

Mechanism: stock PHP returns early for cached blocks and **never unlinks them**
(`zend_alloc.c:271-278`). The port unlinks every block in **both** arms, so that the matched pair
differs in revocation alone. The hoist is needed only by the fault arm -- under revocation a dead
block left in the list poisons its neighbours -- but applying it to the control makes the control
deviate from stock PHP in a way that can fault. CRASH-057's use-after-free write corrupts a block
header and the extra unlink then walks the corrupted link; stock PHP would never walk it.

Measured: the fault site, the arm's flags, and `temporal_sanity` PASSING (benign workload with
revocation ON completes, so the revocation machinery is sound). Inferred but **not isolated**:
that the corrupting write is CRASH-057's own UAF. Isolating it means editing the physical
allocator, which is the comparison column this plan keeps untouched, and it would not change the
classification. `NO-READING / setup-fault` is the honest label and the suite reached it by itself.

### Why this matters for the retarget: the probe proves the deviation becomes unnecessary

The hoist exists only because the physical port has **one capability spanning header and
payload**, so revoking the payload kills the header and every link pointing at it. The seam probe
(stage 0, above) shows the virtual seam separates them: the header stays live through the
payload's retirement. So the virtual port can keep PHP's `_efree` ordering **byte-for-byte in both
arms**, and this whole class of artefact -- CRASH-057's no-reading now, and the four historical
false CAUGHT verdicts -- disappears by construction rather than by careful ordering. That is a
stronger argument for the retarget than the plan originally had.

### Toolchain freshness: the gate fired, and the compiler is exonerated for codegen

`toolchain-fresh.py` reported **STALE**, 3,486 pending steps covering the tools in use including a
clang relink. Two causes, neither of them uncompiled source:

- **`.ninja_log` was missing** while `.ninja_deps` from the build was present, so ninja had lost
  its record of command lines and reported built outputs as dirty ("command line not found in
  log").
- **clang embeds the git revision**, so committing moved it from `c060a688` to `82761ef2`.

Resolved by measurement rather than argument. The build was completed (3,486 steps, rc=0) and one
arm was rebuilt with the finished compiler and compared section by section against the image the
suites actually ran:

| section | old vs new |
|---|---|
| `.text`, `.rodata`, `.data`, `.gct`, `.capstone_domreq`, `.debug_info` | **byte-identical** |
| `.debug_str`, `.symtab` | differ -- the producer string `clang version 22.0.0git (... c060a688)` vs `(... 82761ef2)`, and the offsets that follow it |

So **codegen did not move and the 23/24 stands.** The gate now returns rc=2 ("cannot fully
check") rather than rc=1 ("stale"): it is not stale for the targets it checks, and the residual is
a coverage gap -- it reports that build scripts invoke a tool named `llvm-capstone` that it does
not verify.

One piece of friction worth fixing in dev's tooling rather than here: because clang embeds the
revision, **this gate reports STALE after every commit**, and clearing it costs a 3,486-step
rebuild that changes only a debug string. A revision-only drift is worth distinguishing from a
source drift, or the gate trains people to ignore it -- which is the one thing a C-44 guard must
never do.

## Phase 0 — merge

Per `CLAUDE.md` ("Active plans live in `capstone/docs/plans/`, committed, portable across
machines and agents"), this plan lands first as
`capstone/docs/plans/2026-10-11-php-virtual-capstone-retarget.md`. There is currently no PHP
plan in that directory at all, so the whole port has been running plan-less in the repo.

```bash
git switch php-5 && git merge origin/dev            # expect no conflicts
git submodule update --init --recursive capstone/capstone-qemu
```

Then rebuild LLVM (**in progress**, see the corrected figure above), rebuild QEMU, and re-run
the physical `blind` suite **unchanged**. That
re-run is the regression gate for the merge and it also discharges the pending re-measurement
the physical arm already owed (every image changed in the last session, so 24/24 is currently
provisional). Push at that point — a validated merge is a stable point.

## Phase A — the same engine, as a Linux process

Goal: the narrowest change that answers "does this run as a user process", so that any
difference from the physical arm is attributable to the profile and nothing else.

**Build.** A new `capstone/ports/php/virtual/build-virtual.sh`, modelled on
`capstone/ports/perl/musl/build-perl-domain.sh:44,94-110` (the existing worked example of a
port with a profile switch):

```bash
musl=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh "$out" "$musl"
CC="$out/sdk/capstone-cc"        # supplies triple, ISA features, headers, link script
```

Non-negotiables, from `capstone/runtime/cmake/Application.cmake:42,57` and
`ports/common/application/README.md`:

- Every TU recompiles. `-mllvm -capstone-gp-free -mllvm -capstone-image-gp` are mandatory,
  and **physical objects cannot be relinked** into a virtual image — the loader's
  `.capstone_virtual` ELF marker (`CPONVVM5`, emitted at `runtime/virtual/start.S:136`,
  checked at `runtime/virtual/exec.c:304-316`) rejects them with `ENOEXEC`.
- Do **not** re-specify the triple, `+m`, `-nostdlibinc` or `link.ld`:
  `ports/common/cmake/toolchains/capstone-application.cmake` is explicit that the SDK owns
  them and a second source of truth is the bug.
- `capstone-cc` absorbs `-lpthread -ldl -lm` silently, as `ports/sqlite/app/build-domain.sh`
  relies on.

**Delete, don't port.** `engine/stubinc/` (29 headers) and
`engine/libc/php_capstone_{libc,stdio,os,malloc,php_stubs,ext_stubs,depth}.c` have no virtual
counterpart — musl 1.2.5 compiled for Capstone supplies real stdio, string, wide-char,
`setjmp`, sockets, `pthread`, `posix_spawn`, and a growing heap.

**The beebs tag-preserving `memcpy` goes too.** I expected to keep it — PHP copies zvals,
whose union holds pointers, constantly, and a byte- or word-loop `memcpy` strips the tag. But
`ports/musl-capstone/runtime/string_bounds_safe.c:115-220` already supplies tag-preserving
`memcpy`/`memmove`/`memset` built on a 16-byte `granule_copy`, and documents our exact hazard:
*"8-byte words through a capability that holds 16-byte tagged values, and an 8-byte load-store
pair of half a capability yields an untagged one: a struct with a pointer in it, copied by
musl's memcpy, comes out with a pointer that [has lost its tag]."* It is linked **before**
`libc-capstone.a` so it wins, and it also replaces musl's word-at-a-time `strlen`,
`__strchrnul`, `memchr`, `stpcpy`, `stpncpy`, `strlcpy` and `memccpy`, which fault under
capabilities by reading the word containing a string's last byte. That file's header is worth
reading in full before writing any libc code — it is the list of things a compile survey
cannot see.

**Exactly one file from `engine/libc/` survives:** `php_capstone_zend_hier.c`, the bounds
wrapper, which *is* the experiment.

**Entry point.** `rung_E_domain.c` becomes `virtual/php_virtual_main.c` with a real
`main(argc, argv)` (`runtime/domain/application.c` calls it). The trigger arrives on **stdin**
(`run.py --stdin`) instead of being selected by `-DRUNG_E_CRASHnnn`, which removes ~20
compiled-in case bodies and makes the two arms differ only in the allocator mode. The
stage-bit retval (`ENTERED|SINKED|STARTED|…|EVALED`) is kept verbatim — it is what `reached`
is derived from.

In Phase B this tightens further: the corpus's own native runner invokes
`php -d memory_limit=512M <path>` (`scripts/run5.py:52`), and with real `sapi/cli` and
`main/php_ini.c` the virtual arm can take the **same argv** against a `.php` file staged into
the 9p share. The virtual and ASAN arms then differ in the binary and nothing else, which is
what makes the two columns comparable rather than merely adjacent.

**Run.**

```bash
capstone_vm --state "$VM" up --profile virtual --exact-bounds --launcher …/capstone-vexec \
            --module …/capstone_vm.ko --cma-mib 1024 --process-cache-mib 768
python3 capstone/ports/common/application/run.py --state "$VM" --cwd /tmp \
        --result out.json --stdin corpus/CRASH-110.php php.dom
```

QEMU needs `-M virt-capstone -cpu rv64,…,x-capstone-u-mode=true,x-capstone-exact-bounds=true`;
the launcher refuses to start without the exact-bounds profile (`exec.c:698`).

**Exit A when** the virtual arm reproduces the physical arm's verdicts on the 24 reachable
cases, with each fault at the same function and line.

## Phase A — BUILD RESULT (2026-10-12) and the pre-registration for its first run

**PHP 5.0.0's engine links and runs as an ordinary Linux process on the virtual profile.** 47
objects: 34 Zend TUs (the physical build's own `TUS` list plus `zend_alloc`), 8 `ext/standard`
TUs (`PHP_CAP_EXT=pure`), and 5 support files. `.text` 1,100,780  `.rodata` 125,748  `.data`
34,200  `.bss` 322,544. No undefined weak symbols, so the C-56 preflight has nothing to refuse.

**All 42 of PHP's own translation units compiled against the Capstone musl on the first
attempt.** That was the plan's open question 1 — *"does PHP 5.0.0 build at all as a hosted
process against the Capstone musl SDK?"* — and the answer needed no source changes to `Zend/`
or `ext/`, which stay byte-identical from the read-only corpus mount. The only two failures
were in this port's own new code: `INTERNAL_FUNCTION_PARAMETERS` has no `return_value_ptr` in
5.0.0 (`zend.h:248`), and `php_fault_report` had to be given a virtual-profile definition.

What went away, measured against `engine/build-engine.sh`: the 29-header `stubinc` farm, five of
the six `libc/php_capstone_*.c` files (~900 lines of hand-written libc), the beebs
tag-preserving `memcpy`, the compiler-rt builtin list, `start.S`, `capstone_setjmp.S`,
`domreq.S`, the link script, and every one of `-target`, `+m`, `-ffreestanding`,
`-nostdlibinc`, `-isystem stubinc`. `capstone-cc` supplies all of it.

What stayed, and why each is PHP-level rather than OS-level:

| file | why it has no musl counterpart |
|---|---|
| `libc/php_capstone_zend_hier.c` | the bounds wrapper — it *is* the experiment, reused unchanged |
| `libc/php_capstone_php_stubs.c` | `main/`'s globals (`PG`/`SG`/`BG`, `sapi_module`) and `php_printf`, needed until Phase B links `main/` |
| `libc/php_capstone_ext_stubs.c` | the three externs `ext/standard/url.c` reaches for in `streams` |
| `app/php_virtual_lower.c` (new) | `zend_cap_widen`, which is the IDENTITY here — see below |
| `app/php_virtual_main.c` (new) | a real `main(argc, argv)`, the trigger on stdin |

### `zend_cap_widen` is the identity on this profile, and that is a reading with a test

The physical wrapper must widen any pointer that re-enters the level below, because it narrows
everything it hands up. That widen recovers the block capability from the physical arena's slot
table, which this profile does not have. Reading `runtime/virtual/heap.c:203-217` says the
widen is unnecessary rather than unimplementable: `free()` and `realloc()` use the caller's
pointer for a one-byte probe at the cursor (`lbu zero, 0(p)`) and as a key for
`__capstone_malloc_find`, then operate through `r->raw` — the capability the heap itself
retained over the block. The caller's *bounds* are never the bounds the copy or the free uses.

A reading is not a result, so `app/probe-hier-widen.c` reproduces PHP's exact geometry
(`malloc(48 + n)`, hand back `p + 48`, narrow to `[block_base, object_cursor + n)`) and frees
and reallocs through the narrowed pointer. Its stage 2 is a **negative control**: one byte past
the narrowed end must fault. Without that stage, "free survived a narrowed pointer" is equally
consistent with the narrowing never having happened.

### Phase A measures the SPATIAL axis only — 6 of the 24, not 24

This is a property of Phase A, not of the profile, and it corrects an assumption worth stating
plainly: the physical suite's 23 CAUGHT are **mostly temporal**, measured in `seam` mode with
`-DZEND_TEMPORAL`, where `zend_alloc.c` is dropped and the port's own arena serves `emalloc`
with revocation per object. That arena, its slot table and its revocation are physical-profile
code; replacing them with Sublet regions over a block the virtual heap lends is the allocator
seam, which the section below treats as its own design question. `build-virtual.sh` therefore
**refuses** `PHP_CAP_ALLOC_MODE=seam` out loud rather than quietly building something else.

Derived from `engine/run-triggers.sh` by `app/run-cases.py` rather than retyped — the case list
from `BLIND_REACHABLE` plus the `pureext` list, the axis from `axis_of`, the expected fault file
from `EXPECT_FILE`:

| axis | cases | in Phase A |
|---|---|---|
| spatial (`heap`) | CRASH-016, 066, 073, 091, 102, 110 | **yes, 6** |
| temporal | the other 18 | no — the seam, and silence here is NOT a miss |

### The pair's own positive control, and why it is per-arm

`php_virtual_main.c` carries a third mode, `control`, run through the same engine in the same
state as a case: `emalloc(9)`, write 9 bytes, then read index 9. `REAL_SIZE(9)` is 16, so the
expected outcome is **opposite on the two arms** — the rounded arm must COMPLETE (the slack is
reachable, which is the very mechanism that hides these cases from ASAN) and the exact arm must
FAULT. `arms.json` declares which, so the adapter cannot grade itself. The geometry is
CRASH-110's deliberately: the control does not show that some bound somewhere works, it shows
that *the* bound this case depends on is reachable in this image, on this arm, in this run.

### PRE-REGISTRATION — predicted verdicts, recorded before the first run

Per the root `CLAUDE.md`: the prediction lands on the lane branch before the run, and the result
commit cites this commit's hash.

| arm | predicted | why |
|---|---|---|
| `php-virtual-zend-rounded` (control) | **MISSED** × 6, control `emalloc-overread` = complete | the bound is `REAL_SIZE(n)`, so each over-read lands in the rounding slack exactly as it does on stock PHP. This is the honest-gap column; a CAUGHT here would mean the arms are not what they say |
| `php-virtual-zend-exact` (fault) | **CAUGHT** × 6, each fault in the file `EXPECT_FILE` names (`url.c`, `url.c`, `var_unserializer.c`, `html.c`, `string.c`, `formatted_print.c`), control = fault | all six are CAUGHT physically on this axis; the profile change should not move them |

Three ways this prediction could fail that would be findings rather than noise, named now so
they cannot be rationalised afterwards:

1. **A fault in the right arm at the wrong place.** With real musl underneath, a case can now
   reach further than it did freestanding — a `str_repeat` that used to hit a stub may now run —
   and fault somewhere the physical column never saw. The judge calls that `unattributed`, and
   it must stay that: a catch at an undeclared site is not this case's catch.
2. **The control arm faulting.** It is built to reproduce stock PHP's rounding; if it faults,
   the port deviates from stock PHP and the pair proves nothing (`control-failed`). This is the
   shape CRASH-057 took on the physical column.
3. **Fewer than 6 reached.** `reached` is the "evaluating the trigger" stage line; a case that
   dies in `zend_startup` is `setup-fault`, not a miss.

## Phase B — real PHP

The corpus tree is **already configured** (`main/php_config.h`, `main/build-defs.h`,
`Zend/zend_config.h`, `TSRM/tsrm_config.h` all present), so no cross-configure is needed —
recompile the Makefile's own TU list with `capstone-cc`. Two facts that de-risk reusing a
host-generated config:

- `SIZEOF_CHAR_P` is `#undef` and referenced nowhere outside the config headers, so a
  16-byte capability pointer does not contradict it. `SIZEOF_LONG 8`, `SIZEOF_SIZE_T 8` match
  riscv64; no `WORDS_BIGENDIAN`.
- The TU list is authoritative in `build/php-5.0.0/Makefile`: `PHP_GLOBAL_OBJS` (regex/,
  ext/standard ×65, TSRM/, main/ ×26, main/streams/ ×9, Zend/) plus `PHP_CLI_OBJS`
  (`sapi/cli/php_cli.c`, `getopt.c`, `main/internal_functions_cli.c`).

**The two unavailable syscalls are confined to three TUs**, so this is a flag and an
exclusion, not a redesign:

- No file-backed `mmap`. `HAVE_MMAP` is used only by `main/streams/plain_wrapper.c` (plus
  headers) — build with `-UHAVE_MMAP` and plain-file streams fall back to `read()`.
- No `fork`. Used only by `ext/standard/mail.c` and `ext/standard/proc_open.c`, neither
  needed by any of the 31 cases — drop them from the TU list. (`posix_spawn` *is* available
  if they are ever wanted.)

**Turn `-capstone-shrink-stack` back on.** It was off only because of the ceiling. Doing so
restores stack-object bounds and is what makes CRASH-100 (`stack-use-after-return`, CWE-562)
catchable at all. Treat it as its own variable: measure the 24 known cases with it on before
claiming anything about the stack class.

**Expected order of attack on the 7**, cheapest first: CRASH-086 (image size only) →
CRASH-008 (`time()`) → CRASH-084 (`environ`) → CRASH-112 (files, `/tmp`) → CRASH-099
(`php://memory`, `stream_select`) → CRASH-092, CRASH-100 (loopback listener; both need a
second process in the guest, and CRASH-100 additionally needs shrink-stack on).

## The allocator seam — the one real design question

PHP's allocator must keep giving per-`emalloc` **bounds** and per-`efree` **lifetimes**, and
the current compromise must go away. Today the temporal axis is measured only in `seam` mode
(`zend_alloc.c` dropped, our arena serving `emalloc` directly) because in `hier` mode PHP's
`AG(cache)` — 11 buckets × 256 entries, index `REAL_SIZE(size)>>3`, so blocks ≤ 80 bytes
never reach `free()` — absorbs most frees, and an `efree` that never reaches the arena cannot
revoke. That buys per-object lifetimes at the price of `zend_alloc.c` no longer being under
test, which is the opposite of the point.

### PHP 5.0.0 is not a chunk allocator, and that makes this simpler than the other ports

I had assumed PHP's allocator carves objects out of chunks, like pymalloc's arenas or wmem's
blocks. **It does not, in this build.** `Zend/zend_mm.h:31-36` reads:

```c
#ifdef ZEND_WIN32
#undef ZEND_MM
#else
/* #define ZEND_MM */
#undef ZEND_MM
#endif
```

`ZEND_MM` is unconditionally undefined and defined nowhere in the configured tree, so
`ZEND_DO_MALLOC(size)` is plain `malloc(size)` (`zend_alloc.c:55`) and the structure is
**flat — one `malloc` per `emalloc`**:

```c
/* zend_alloc.c:181 */
p = malloc(sizeof(zend_mem_header) + MEM_HEADER_PADDING + SIZE + END_MAGIC_SIZE);
/* ... returns ... */
return (void *)((char *)p + sizeof(zend_mem_header) + MEM_HEADER_PADDING);
```

with `SIZE = REAL_SIZE(size) = (size+7) & ~7` (`:132`) and `END_MAGIC_SIZE = 0` in a
non-debug build (`:64`). The corpus's two masking mechanisms are exactly these two lines:
the `REAL_SIZE` rounding gives up to 7 bytes of slack, and `_efree` parks the block in
`AG(cache)` and **returns without calling `free()`** whenever
`CACHE_INDEX < 11 && cache_count < 256` (`:271-279`).

Three consequences, all of which cut work:

1. **The linear-block-loan model does not apply.** There is no arena, no pool, no alignment
   requirement — so no `capstone_borrow_aligned_block`, and `pymalloc/block-lifetimes.c`,
   `wmem/chunks.c` and the memcached authority layer are **not** the template here. We are a
   different shape from every existing nested port: flat, one outer object per inner object.
2. **The spatial axis is already done.** Each `emalloc` is its own mallocng object, which
   `heap.c` bounds to the *requested* — i.e. header-inflated — size, and
   `php_capstone_zend_hier.c`'s `zc_rebound` narrows the returned pointer to the caller's
   size. That file carries over unchanged; `-DZEND_CAP_BOUNDS_REAL_SIZE` remains the spatial
   control, reproducing PHP's own rounding.
3. **`AG(cache)` stops being the obstacle, so the seam-mode compromise dies.** Revocation
   belongs at `_efree` in *our wrapper*, above PHP's cache, not at `free()` below it.
   `zend_alloc.c` therefore stays byte-identical **and** we get per-object lifetimes — the
   thing the physical port could only have one of at a time.

**And the intrusive-list problem disappears.** `zend_mem_header` — holding `size`, `cached`,
`pNext` and `pLast` — lives in the **prefix below the payload** (`_efree` recovers it as
`ptr - sizeof(zend_mem_header) - MEM_HEADER_PADDING`, `:250`). Revoking only the payload
sub-range leaves PHP's header live and writable, so the unlink-before-revoke /
re-link-on-cache-hit machinery the physical allocator needed is not required at all.

### The recommended seam: split the block, and never revoke the header

My first instinct was to extend the existing TU-scoped rename (`-D_emalloc=php_raw_emalloc` …,
`build-engine.sh:151-154`) with `-Dmalloc=…` and do the take/give in the wrapper, leaving
`zend_alloc.c`'s text untouched. **A design pass showed that cannot work, and the reason is
worth stating because it is not obvious.**

`zend_alloc.c` computes the client pointer as `p + 48` (`:168`, `:216`, `:370`) and recovers
the header as `ptr - 48` (`:250`, `:320`, `:660`). Both directions need **one capability whose
bounds span header and payload**. If that is what the caller holds, then `_efree` must revoke
it — and that kills the header too, because they share one revocation node. PHP then cannot
read `p->size` (`:263`), write `p->cached = 1` (`:273`), or park `p` in `AG(cache)` (`:272`),
all of which happen *after* the logical free. Worse, every `pNext`/`pLast` alias of that block
held in *neighbouring* blocks' headers dies with it, so `shutdown_memory_manager` walks dead
capabilities (`:530-568`).

**The physical port already paid this bill.** `zend-alloc/zend_capstone_alloc.h:518-552`:
*"a revoked block left in the list poisons its neighbours… the next unlink that walks through
it loads an untagged capability and faults with cause 24, inside the allocator, on a program
with no lifetime error in it… It took four false CAUGHT verdicts to find."* Its fix had to
hoist `REMOVE_POINTER_FROM_LIST` above the revoke and make it unconditional, and add
`ADD_POINTER_TO_LIST` on the cache-hit path — and it admits *"This is a deviation from PHP's
`_efree`"* (`:533`).

So the design is to make PHP's metadata out-of-band **with respect to revocation**, without
moving a byte of it: split the block so no capability spans the boundary, and give PHP the
*header* capability while the payload is reached through a hook.

```
B                 B+48                                    B+48+PAD16(REAL_SIZE(n))
|<- 48 hdr ------>|<------------ payload ------------------>|
  never revoked      one region, revoked on every efree
```

Confirmed against the configured tree, not inferred: `ZEND_DEBUG 0` and `MEMORY_LIMIT 0`
(`main/php_config.h:2373,2382`) and `ZEND_ENABLE_FAST_CACHE 0` (`Zend/zend_fast_cache.h:27`).
So `zend_mem_header` is `{pNext, pLast, size:31, cached:1}` (`zend_alloc.h:49-55`), which at
`PLATFORM_ALIGNMENT = __alignof__(void*) = 16` is **48 bytes with `MEM_HEADER_PADDING = 0`**.
48 ≡ 0 mod 16, so `B` and `B+48` are both 16-aligned — which is what makes the split legal,
since `sublet_give_to`'s fill loop steps 16 and traps on a region that is not
(`sublet.h:124-128`). `ZEND_ENABLE_FAST_CACHE 0` additionally means every zval and every
string genuinely reaches `_emalloc`/`_efree`, which is exactly what the temporal axis needs.

**The patch: six derivation sites and three macros, every hunk guarded.** All in
`zend_alloc.c`, all behind `#ifdef ZEND_CAP_SUBLET`, so with the define absent the file
compiles bit-for-bit as today and the physical arm's results stand. This is the wmem
discipline verbatim (`ports/wireshark/wmem/patches/…-0002-*.patch` is *"inert in full
Wireshark… every hunk is guarded"*).

| site | upstream | patched |
|---|---|---|
| `:55-57` | `malloc`/`free`/`realloc` | `zend_cap_block_{alloc,free,realloc}` |
| `:168`, `:216`, `:370` | `(char *)p + sizeof(zend_mem_header) + MEM_HEADER_PADDING` | `ZEND_CAP_PAYLOAD(p, size)` |
| `:250`, `:320`, `:660` | `(zend_mem_header *)((char *)ptr - …)` | `ZEND_CAP_HEADER(ptr)` |
| new, after `:269` | — | `ZEND_CAP_RETIRE(p);` |

Nothing else moves: not `REAL_SIZE`, not `cache_index`, not the list macros, not the cache
policy, not the control flow. And `_ecalloc`/`_safe_emalloc`/`_estrdup`/`_estrndup` need **no**
wrapper at all — they return `_emalloc`'s pointer, so they inherit the bound from one hook,
where today's hier wrapper re-bounds all six separately and has to recompute `strlen`
(`php_capstone_zend_hier.c:162-190`).

| step | what our layer does |
|---|---|
| `zend_cap_block_alloc(n)` | carve, take the **senior** handle *before* any split, `sublet_split` at `B+48`, delinearise the header, index the record; return the **header** alias |
| `_emalloc` (fresh) | `ZEND_CAP_PAYLOAD` → `sublet_take` the payload region, return it shrunk to `size` (not `REAL_SIZE(size)`) |
| `_efree` | `ZEND_CAP_RETIRE` → `sublet_give` the payload. The caller's pointer dies **while PHP's header stays live**, so the cache park at `:271-278` runs unmodified and *no unlink/re-link deviation is needed at all* |
| `_emalloc` (cache hit) | `sublet_take` again on the same region: a fresh node over the same bytes, so the previous tenant's alias stays dead. (`zend-alloc/probes/revoke-reuse-safety.c` already demonstrates exactly this property.) |
| `_erealloc` | copy **through the allocator's own full-payload alias**, then `sublet_give_to` the senior handle — and only on success, since `:346-358` dereferences the original on the failure path |
| shutdown | `sublet_give_to` on the senior handle ends header, payload region and every object alias in one revoke |

**`_erealloc`'s old bug is eliminated, not fixed.** It faulted because the lower level copied
a whole block's worth through an object-sized bound (`php_capstone_zend_hier.c:142-148`). Here
the allocator never holds the client's capability at all — `ZEND_CAP_HEADER` converts at the
entry and the copy uses the record's own alias. "Narrow going up, widen going down" ceases to
be a rule to remember, and `zend_cap_widen` goes away with it.

**Two mode bits, not one.** pymalloc's `pym_set_mode(0|1)` conflates bound and revocation, and
our suite must not: the spatial pair varies the bound with revocation **on in both arms**
(so the control also exercises the machinery `temporal_sanity` exists to validate), while the
temporal pair varies revocation with the bound held at `REAL_SIZE`.

```c
#define ZEND_CAP_MODE_EXACT   1u   /* 0 = bound to REAL_SIZE(n) — stock rounding; 1 = bound to n */
#define ZEND_CAP_MODE_REVOKE  2u   /* 0 = one take per block, never revoked; 1 = give+take per retire */
```

| pair | control | fault | differs in |
|---|---|---|---|
| spatial | `REVOKE` | `REVOKE\|EXACT` | the bound alone |
| temporal | `0` | `REVOKE` | revocation alone |

The mode comes from `PHP_CAP_MODE` and is **echoed to stderr**, so each run's arm is in its own
log; an absent region or a failed loan must be loud, never a silent fallback to an unprotected
heap.

**Metadata.** One record per live block, holding three `capstone_cap_slot`s. A slot is a
*place*, so a record must never be `memcpy`'d or `realloc`'d — the table is a chain of fixed
slabs. Index on **one key, the payload base**, with `heap.c`'s chained hash (`:23-47`).
Explicitly **do not** reuse the physical port's linear scan with a `base == 0` free marker:
`zend_capstone_alloc.h:340-353` records that sentinel producing a false positive. Starting
sizes, all `-D`-overridable: 64 MiB arena, 32 MiB metadata, 256 Ki records, 16384 buckets —
against `ZEND_MAX_SLOTS=4096` / `ZEND_ARENA_BYTES=512 KB` today, which covered `zend_startup`
and nothing more. Exhaustion must not surface as PHP's own *"FATAL: emalloc(): Unable to
allocate"* + `exit(1)` (`zend_alloc.c:189-195`), which reads as the workload failing rather
than the instrument.

### Why `heap.c`'s own ABI still matters

`runtime/virtual/heap.c` is already a per-object-bounds, revoke-before-reuse `malloc`/`free`,
and it exports the documented **nested-allocator ABI** that the table above uses:

```c
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out);  /* loan a LIN block */
void          __capstone_sublet_free_linear(unsigned long base);           /* revoke descendants */
```

and the Sublet discipline (`capstone/sublet/sublet.h`) defines the moves the table uses:
`sublet_take_linear` for a block from below, `sublet_split` + `sublet_take` to carve and issue
an object, `sublet_give` to retire one, `sublet_give_to` on a senior handle to kill every
descendant at once. The hard rule is that **an allocator under Sublet never holds a capability
in a C variable** — every capability it owns lives in a `sublet_cap` slot in memory. We do
*not* need `borrow-aligned-block.h`: it exists because arena allocators demand an aligned
backing region, and PHP has no arena.

`docs/design/virtual-capstone/ownership.md` states the trap this design has to avoid: *"An
allocator that merely reuses bytes inside one still-live outer allocation does not obtain
inner temporal protection automatically."* For PHP the still-live outer allocation is the
cached block, and `AG(cache)` reuse is precisely that trap — which is why the retire has to
happen at `_efree`, not at `free()`.

The existing adapters remain worth reading for *discipline* rather than structure:
`ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c` and its separate metadata
heap `src/shared/backing.c`, `ports/wireshark/wmem/src/allocators/sublet/chunks.c` for the
open/split/issue/retire/reset/close contract, and
`bug-corpora/memcached/allocator-repros/shared/driver-virtual.c` for the minimal `main()`
that sets the mode. But their chunk-carving structure is not ours.

Two caveats carried forward. **`ports/perl/musl/build-perl-domain.sh:57` refuses its SV-head
arena arm on the virtual profile** ("virtual SV-head arena integration is not qualified") —
Perl's SV-head arenas are the nearest existing analogue to a PHP-like small-object layer, so
expect to be first here, and expect the SDK not to have been exercised in this shape.
Second, **node-table pressure**. Cost per operation: a fresh block mints 4 nodes (2 splits +
senior handle + take), steady state is 1 retired per `efree`/`emalloc` cycle, and a retired ID
is not reusable until a namespace sweep — so *"table demand includes handles, object nodes,
retired IDs awaiting collection and pinned identities"*. On the virtual profile the table
**grows** (`node_initial_pages=4`, `node_batch_pages=16`, `node_max_pages=0` for a 2³¹
ceiling, `runtime/virtual/module/capstone_vm.c:59-63`), so the FPGA's 65,536 is not the limit
here; PostgreSQL's high-water of 84,193 is the right order to plan against, and PHP will be
worse because `ZEND_ENABLE_FAST_CACHE 0` puts every zval through `emalloc`. Three defences:
pre-request with `__capstone_vm_nodes` at every patched entry (`heap.c:48-56` idiom), load the
module with `node_initial_pages=1024` so startup does not provoke a growth-and-sweep storm,
and **refuse any verdict from a run whose node high-water came within `CV_NODE_RESERVE` of
capacity** — a cause-30 resource fault lands at an arbitrary pc and is otherwise
indistinguishable from a catch. Note that **no existing port calls `__capstone_vm_nodes` from
an application TU**; every nested port relies on `heap.c`'s per-`malloc` reserve, which does
not transfer to a layer that mints its own nodes.

**Open, to settle in code rather than on paper.** Four, in the order they would bite:

1. **Does PHP 5.0.0 build at all as a hosted process against the Capstone musl SDK?** The
   whole existing port is freestanding. This dwarfs the allocator work and is orthogonal to
   it — the adapter design is correct either way — which is exactly why Phase A comes first.
2. **Does `__builtin_capstone_cap_get_cursor` trap on a revoked capability?**
   `ZEND_CAP_HEADER` reads the cursor of the client pointer, so a double-`efree` must fault at
   a *named probe label*, not inside the index lookup — a fault inside the allocator is
   refused as a port artefact by the harness's own gate. Two-line directed test.
3. **Does PHP ever `efree` a pointer that is not exactly what `emalloc` returned?** The
   single-key index depends on it. `zend_alloc.c` is verified and no other file does header
   arithmetic (the only other site, `main/main.c:903-904`, is `#if ZEND_DEBUG` and only
   *prints* the address), but every `efree` call site in `Zend/` and `ext/standard/` has not
   been audited. Probe-then-miss-then-fail makes a violation loud rather than silent, which is
   the right failure mode, but it should be audited rather than discovered.
4. **PHP's node high-water is unmeasured.** Everything above is a mechanism for finding out
   and failing closed, not a prediction. The first `temporal_sanity` run *is* the measurement.

A ~50-line standalone probe against the virtual SDK answers (2) and the split legality before
any PHP code is written, and should be the first thing built.

## Measurement — adopt the shared judge

`capstone/bug-corpora/tools/verdicts.py` is the judge; runners only report. Its vocabulary is
three verdicts and a closed reason set, and our gates map onto it one-for-one:

| our gate | dev's reason |
|---|---|
| control did not complete | `NO-READING control-failed` |
| fault outside `EXPECT_FILE` | `NO-READING unattributed` |
| `EVALED` unset (trigger never ran) | `NO-READING not-reached` |
| fault inside the allocator | `NO-READING setup-fault` |
| `temporal_sanity` failed | arm-level control, same invocation |
| exit 75 | `NO-READING` |

`judge()` is a pure function of `(Observation, arms.json entry)`, so stored results can be
re-judged without re-running — and `derive-verdicts.py --check` refuses to let an old verdict
stand after the judge or the arms change. Concretely:

- `run-triggers.sh` stops deciding. It builds an `Observation` (`case`, `arm`, `reached`,
  `completed`, `fault`, `attribution`, `controls`) and calls `judge()`.
- `fault_site()` is replaced by `verdicts.domain_fault(text, symbols)` — our hand-rolled
  `pc − pcc_base + 0x10000` relocation and `llvm-symbolizer` call go away, along with the
  un-anchoring bug that once silently returned an empty site.
- `EXPECT_FILE` becomes `fault_sites` in a per-case `case.json`, which is what promotes a
  fault to `attribution: "function"`.
- Two entries in `capstone/bug-corpora/tools/arms.json` — `php-zend-stock` (zend_alloc over
  one mallocng object; the honest-gap column) and `php-zend-sublet` (per-`emalloc` lifetimes)
  — each with `target: "capstone-virtual"`, a prose `mechanism`, and `controls` +
  `controls_for_missed`, so a *silence* is only a reading on an arm that proved in that run
  that it is what it says.
- Results land as `results/<stamp>/<arm>/{verdicts.jsonl,inputs.json}` via
  `verdicts.write_bundle()`, whose `inputs.json` records sha256s of qemu, kernel, firmware,
  rootfs, launcher, module, compiler, runner **and judge**.

**Scoped out of this plan** (the natural follow-on): restructuring the 127-case corpus into
dev's `NN_<upstream-fix>_<slug>/` layout with `PROVENANCE.md` per case. Without that, PHP's
bundles exist and are re-judgeable but do not appear in the cross-program catch tables. I
propose a thin `capstone/bug-corpora/php/engine-repros/corpus.json` naming the arms and the
bundle locations, so the plumbing is in place and the restructuring is additive later.

**Pre-registration is a process rule, not a tool** (root `CLAUDE.md` on dev): push the
predicted verdict per cell to a lane branch *before* the run and cite that commit's hash in
the result commit. For the first virtual run the prediction is: `php-zend-stock` → MISSED for
every case masked by `AG(cache)` or by `REAL_SIZE` rounding (i.e. all 31, which is the whole
point of the population); `php-zend-sublet` → CAUGHT for the 24 already caught physically.

## Verification

1. **Merge gate.** `git merge-tree` clean (already confirmed); after merging, rebuild LLVM +
   QEMU and re-run the physical `blind` suite — 24/24, each fault at ASAN's function and
   line. Run `llvm-nm --undefined-only` over every `.dom` first, for the new C-56 preflight.
2. **Profile gate.** `capstone/runtime/virtual/run.py` (the 46-check application gate) and
   `run-malloc.py` must pass on this machine before any PHP result is believed. These are the
   positive controls for the *platform*; a PHP failure against an unproven platform says
   nothing.
3. **Seam probe.** `probe-sublet-split.c` must demonstrate, standalone, that a `heap.c`
   linear loan can be split at an arbitrary header offset, that the payload alias dies on
   `sublet_give` while the header stays writable, and that a re-`sublet_take` leaves the old
   alias dead. If any of those fails the seam design changes, and finding that out from a
   50-line probe is much cheaper than from a 2.3 MB PHP image.
4. **Patch inertness.** Rebuild the physical `hier` object from the *patched* `zend_alloc.c`
   with `ZEND_CAP_SUBLET` undefined and `cmp` it byte-for-byte against the current one. If
   that passes, the existing physical results stand unchanged and the guard is proven.
5. **Arms differ for the stated reason — with new gates, since one image breaks the old
   ones.** `run-triggers.sh:321-326` currently refuses byte-identical arms and a fault arm
   with no `revoke` instruction; neither test applies to a single image with a runtime mode.
   Replacements: (a) the image must contain the revoke encoding at all; (b) the two runs must
   report *different* mode lines; (c) **the fault arm must report a strictly larger revoke
   count than the control**. (c) is the positive control for "revocation actually fired", and
   it is the gate that would have caught the physical port's four false CAUGHTs cheaply. Note
   `sublet_stats` is `static` per TU (`sublet.h:46-48`), so `heap.c`'s counters do **not** see
   the adapter's — it needs its own `zend_cap_stats()`.
6. **Revocation sanity.** Keep `temporal_sanity` as an arm-level control in `arms.json`: a
   benign workload with revocation ON must complete. Four use-after-free cases once reported
   CAUGHT while all faulting at the same allocator line, and only this control saw it.
7. **Negative control for the node table.** Deliberately exhaust the node table and confirm
   it is judged `NO-READING setup-fault`, not CAUGHT. A cause-30 fault scored as a catch is
   the most likely way this port produces a false positive.
8. **A/B the profile, not the port.** For the 24 cases measured both ways, physical and
   virtual verdicts must agree. A disagreement is a finding about the profile and must be
   explained before the 7 new cases are attempted.
9. **Then, and only then,** the 7 OS-dependent cases, cheapest first, each with its control
   arm completing.

## Files

**New**, following the existing ports' layout:

```
capstone/ports/php/zend-alloc/sublet/
  probe-sublet-split.c   /* ~50 lines, SDK only. BUILD THIS FIRST — it decides the seam  */
  zend-cap-port.h        /* geometry, the two mode bits, the record struct, fail codes   */
  zend-cap-blocks.c      /* block provider + per-object lifetimes  (~ block-lifetimes.c) */
  zend-cap-backing.c     /* slab metadata heap + the address index  (~ backing.c)        */
capstone/ports/php/zend-alloc/patches/
  php-5.0.0-0001-zend-alloc-sublet-hooks.patch          /* 6 sites, all #ifdef-guarded   */
  php-5.0.0-0001-zend-alloc-sublet-hooks.patch.classes  /* what each hunk does, per wmem */
capstone/ports/php/app/
  php-virtual-main.c     /* borrow block, init, set mode, run   (~ driver-virtual.c)      */
  build-virtual.sh       /* virtual SDK + the Makefile's TU list + link                  */
  port.json
  run-cases.py           /* Observation builder over ports/common/application/run.py     */
```

A `static_assert(sizeof(zend_mem_header) == 48 && MEM_HEADER_PADDING == 0)` belongs **in the
patch**, so a tree configured differently fails the build rather than the run.

**Routing choice, stated explicitly:** use the *borrow* route
(`__capstone_sublet_malloc_linear`), not `build.py --nested`. The latter forces nested arms
onto `--heap level0` (`build.py:122-123`, *"nested discovery arms use level0 for their
separate outer heap"*), which would give up per-object bounds on everything PHP allocates
outside `emalloc` — streams, ini, the SAPI. The borrow route keeps `heap.c`'s revoke-on-free
mallocng underneath *and* gives the inner layer its own lifetimes, which is the two-level
story, and it needs no harness change.

**Modified** — `capstone/bug-corpora/tools/arms.json` (two entries);
`ports/common/application/build-virtual.sh` (a fifth `php` case beside
`cpython|postgres|ffmpeg|tshark`); `engine/run-triggers.sh` (bare `flock` →
`capstone_with_qemu_lock`).

**Untouched** — all of `capstone/ports/php/engine/` and `zend-alloc/`, which remain the
physical comparison column. `Zend/` and `ext/` stay byte-identical from the read-only corpus
mount; that is the whole basis of the result and nothing here changes it.

**Reused, not rewritten** — `runtime/virtual/{build-sdk.sh,build-adapter.sh,heap.c,run.py}`,
`ports/common/application/{build-sdk.sh,build.py,run.py}`,
`ports/musl-capstone/{prepare-musl-capstone.sh,runtime/string_bounds_safe.c}`,
`bug-corpora/tools/verdicts.py`, `runtime/host/capstone_vm/cli.py`, `capstone/sublet/sublet.h`.
(`borrow-aligned-block.h` is deliberately *not* on this list — see the allocator section.)
