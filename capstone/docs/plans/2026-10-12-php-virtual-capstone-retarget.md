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

## Platform gate — RESULT (2026-10-12): `run.py` 34 of 41, `run-malloc.py` 17 of 19

Plan Verification item 2, run before any PHP verdict is believed. `runtime/virtual/run.py` with
`contract.dom`, `sqlite3.dom` and `mruby` built against this SDK. Two attempts were needed, and
the reason is a finding in itself.

| attempt | scope | passed | cut short by |
|---|---|---|---|
| 1 | full battery, `--timeout 2700` | 23 | timed out inside `sparse-stale`, the second 200,000-allocation recycling case |
| 2 | `--skip-recycling`, `--timeout 1800` | 32 of 38 | `mruby` never exited; everything after it is collateral |

**Union: 34 of the 41 distinct checks pass.** The seven that never did are `mruby`,
`mruby_results`, `sparse-stale`, and the four that run after mruby in the same guest script
(`cleanup`, `completed`, `linux_pages_released`, `linux_demand_faults`).

What passed is the part PHP's seam rests on, including the checks that are hardest to satisfy
by accident:

- `fault_causes` — the whole expected cause sequence `[24, 24, 28, 24, 15, 13, 13, 28]`, in
  order. A platform that trapped the wrong thing, or the right thing in the wrong place, fails
  this check and nothing else in the gate is as specific.
- `bounds`, `stale`, `retired`, `linear`, `linear-stale`, `linear_denial_site`,
  `vm_denial_sites`, `vm_denial_operations` — spatial faults, stale-capability faults, retired
  nodes, and the linear-loan denials, each at its declared site.
- `threads`, `threads_output`, `heap-threads`, `preempt`, `parallel_one`, `parallel_two`,
  `spin_entered` — the shared-`mm` thread contract and signal termination.
- `sqlite_write`, `sqlite_read`, `sqlite_results` — SQLite's shell writing a database in one
  process and a second process reading `SQLITE_PERSIST=100` back out of it. This is the closest
  analogue to PHP in the battery: a large C program with its own allocator over musl, touching
  files, and it is the check that most directly says "a program of this shape runs here".
- `sparse-churn` with `sparse_recycling` (attempt 1): 200,000 allocations with protected stale
  tags, `nodes=2600070 collections=124 reclaimed=2598542`. Node collection works at scale.

### `run-malloc.py`, the gate's other half — RESULT: 17 of 19, both failures in the ORACLE

The layer PHP's allocator sits directly on. Every check about the Capstone allocator passes:

- `bounds`, `stale`, `stale-free`, `reuse-free`, `reuse-stale`, `inplace-stale`, `remap-stale`,
  `interior-free` — spatial and lifetime behaviour, including a free through an interior pointer;
- `200000_lifetimes`, `70000_simultaneously_live_objects`, `collection_exercised` — at scale;
- `local_allocation_fast_path`, `fast_path_control_detects_service`, `no_kernel_fault`,
  `real_linux_mremap_growth_and_shrink`, `basic_inplace_mremap_failure`, `cleanup`.

The two that fail, `native_reference` and `nonlinear_and_linear_tag_transfer`, have **one**
cause between them and it is not the platform: the native reference binary is the buildroot
**riscv64 glibc** cross-compiler, because this machine has no native RV64 *musl* toolchain, and
`malloc-contract.c` asserts allocator behaviour that is musl's. It died at
`MALLOC_FAIL:moving realloc trigger` with `MALLOC_NATIVE_EXIT:1` — glibc's `realloc` did not
move the block where musl's does. `nonlinear_and_linear_tag_transfer` then fails for the same
reason and nothing else: it counts `MALLOC_TAG_COPY_OK` twice, once per `ok` run, and the native
run exited before printing it. **The Capstone run printed it**, so the tag transfer the check
exists for — including the linear `sublet_move` case — does hold here.

This was predicted before the run rather than explained after it. Closing it properly needs a
native RV64 musl toolchain, which is a buildroot reconfiguration and its own piece of work; the
gate must not be reported as passing until then.

### The emulator is a debug build, and that is why the long cases are impractical here

Upstream's committed passing record (`runtime/virtual/pthread-compatibility-result.json`)
reports `launch_ns` of 89–107 **ms**. The same launches here cost 1–6 **seconds**, and
`sparse-churn` alone ran 590 s. `capstone/capstone-qemu/build` is configured
`--enable-debug`: no optimisation, assertions on. That accounts for the ~60× gap far better
than host load does, and it is the reason the recycling pair cannot be run casually on this
machine.

It was **not** swapped for a release build, deliberately: the physical column's 23-of-24 was
measured with this exact binary, and every result bundle records its sha256. Changing the
emulator mid-measurement would be a change to the instrument between two columns that are
meant to differ only in the profile. A release build is the right thing to do once, as its own
step, with both columns re-measured.

### mruby: unresolved, and probably the node table rather than the port

mruby ran 1,600+ s at 100% CPU, silent, and was killed by the gate's timeout. One concrete
difference from upstream's passing run is in its own stats: upstream shows
`node_bytes=1048576`, ours `node_bytes=28672`. `run.py`'s guest script loads the module with a
bare `insmod capstone_vm.ko`, so the node table starts at `node_initial_pages=4`, and a GC that
mints nodes faster than the table grows provokes exactly the growth-and-sweep storm this plan
names as open risk 4. That is a hypothesis with an obvious test — load the module with
`node_initial_pages=1024` — not a diagnosis, and it is recorded as such.

**What this does and does not license.** It licenses believing a PHP fault on this platform: the
trap causes, the sites, the bounds, the stale-capability behaviour and the node machinery are
all measured here and all correct. It does not license a claim that this machine passes the
gate as published — it does not, 34 of 41 — and the gap is named rather than rounded off.

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

### `zend_cap_widen` on this profile — the question, and how it was first answered wrongly

The physical wrapper must widen any pointer that re-enters the level below, because it narrows
everything it hands up. That widen recovers the block capability from the physical arena's slot
table, which this profile does not have, so Phase A had to answer what replaces it.

The first answer was **the identity**, on the strength of reading
`runtime/virtual/heap.c:203-217`: `free()` and `realloc()` use the caller's pointer for a
one-byte probe at the cursor (`lbu zero, 0(p)`) and as a lookup key, then operate through
`r->raw` — the capability the heap itself retained over the block — so the caller's *bounds* are
never the bounds the copy or the free uses. That reading is correct. The conclusion drawn from
it was not, and the section below records what it missed, what the probe that blessed it could
not see, and the measurement that settled it. The probe is `app/probe-hier-widen.c`; it now has
five stages because two of them were added after the engine refused to start.

### `zend_cap_widen` — RESULT: the identity was WRONG, and so was its first replacement

Measured, not inferred. `probe-hier-widen.dom` under `capstone-vexec`:

| stage | tests | reading |
|---|---|---|
| 0 | free through the header recovered from a NARROWED object pointer | **PASS**, completed |
| 1 | realloc through that pointer, then check the payload survived the copy | **PASS**, payload intact, block moved `...98050` → `...98230` |
| 2 | **negative control**: write one byte past the narrowed end | **FAULT**, cause 28 |

Stage 2's hardware line is the one that makes the other two mean anything:

```
[CAPSTONE] Cap mem access OOB: ... cursor = 3fa608a080, imm = 24,
           addr = 3fa608a098, size = 1, bounds = (3fa608a050, 3fa608a098)
```

`addr` is exactly `bounds.end`, one byte past a 72-byte block ending at `...098` — so the
narrowing in stages 0 and 1 was real and the frees did not merely survive an unenforced bound.
`zend_cap_widen` is therefore the identity on this profile, `php_capstone_zend_hier.c` is reused
with no edit, and Phase A needs no block registry. Node use for the whole probe: `nodes=49
node_capacity=1024 node_growths=0`.

#### Why the identity was wrong, and what the probe was blind to

Stages 0–2 above are correct and were still not enough. They establish that `free()` and
`realloc()` need no widen, which is true, and that was read as *"`zend_cap_widen` can be the
identity"*, which is false.

`_efree` does not merely hand the pointer DOWN. It **parks** it:
`AG(cache)[CACHE_INDEX][count++] = p` (`zend_alloc.c:272`), and a later `_emalloc` of the same
size class hands that same `p + 48` to a *different* tenant. Narrowing is monotonic, and
`CACHE_INDEX` is `REAL_SIZE(size) >> 3`, so a 25-byte object and a 32-byte one share a bucket:
the 32-byte tenant inherits a capability that ends after 25 bytes, and its own
`p[length] = 0` faults at `zend_alloc.c:404`, inside `_estrndup`, with no defect anywhere near
it.

**Measured, on the first run of the exact arm: all six cases *and the arm's own control* faulted
at `zend_alloc.c:404` during `zend_startup`, before any trigger ran.** The judge refused all six
as `NO-READING / setup-fault` rather than reporting six catches — the gate working exactly as
intended on an arm that was broken.

The hier wrapper's own comment block names this line and this mechanism as the reason the
physical port needed the widen. The probe had tested the half of that comment about descending
into the lower level, and not the half about the cache. **The lesson is not "the comment was
unclear" — it is that a probe which confirms the reading you already had is the cheapest kind
to write and the easiest to over-read.**

#### And the replacement had a second bug, which stage 3 caught in one run

The fix is the physical port's: recover the block capability, using `__capstone_malloc_find`
(a declared part of `arch-capstone64/malloc_capstone.h`) in place of the arena's slot table, so
there is no second source of truth and no table of our own to exhaust.

Transcribing the physical formula directly was wrong. It re-bases on the slot capability's
**base** (`php_capstone_malloc.c:190`), which is right there because an arena slot's cursor *is*
its base. `heap.c`'s `r->raw` is the pointer mallocng returned: its cursor sits at the block
key while its base is lower, since mallocng's slot prefix lies below the object. Using the base
shifted the cursor by exactly that prefix — 16 bytes — so PHP's `ptr - 48` landed inside the
allocator's own metadata and read garbage as a capability.

Two readings, one run apart:

| | reading |
|---|---|
| stage 3 (the fix's own test) | **FAIL**: `narrowed cursor=…080`, `widened cursor=…090` — "the widen moved the cursor" |
| the engine, same build | cause 24 at one pc for every case and both arms, `value = 0x12` — an untagged pointer dereferenced |

The second is what this bug looks like without the probe: a single unreadable fault site shared
by every arm, which is the shape of *"every arm reports the same upstream failure"* that this
project's own debugging rules warn about. Stage 3 names it in one line instead. Re-basing on
`r->raw`'s cursor fixes it, and all five stages then behave:

| stage | reading |
|---|---|
| 0 free through a narrowed-derived header | PASS |
| 1 realloc through a narrowed pointer | PASS, payload intact |
| 3 widen recovers the block extent, cursor intact | PASS |
| 4 cache hand-back: park as 25 bytes, re-issue as 32, write index 31 | PASS |
| 2 **negative control**: one byte past the narrowed end | FAULT, cause 28, `addr == bounds.end` |

**What the widen grants, stated because it is a real cost.** The recovered capability spans the
heap's whole block, mallocng's slot prefix included, so an under-read *below* a cached object is
not caught. The wrapper re-narrows everything it hands up, and PHP holds the wide pointer only
as the parked cache entry and for its own header arithmetic. The physical port makes the same
trade for the same reason.

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

## Phase A — SPATIAL PAIR RESULT (2026-10-12): 6 of 6 CAUGHT, 5 by site and 1 by the audit arm

Pre-registered at `57bed3ca68b1` before the run. Three boots — the diagnostic audit arm, then
one per scored arm — with the trigger on stdin and the verdict from
`bug-corpora/tools/verdicts.py`. Bundles:
`$CAPSTONE_TMP_ROOT/php-virtual-runs-final/bundles/<arm>/{verdicts.jsonl,inputs.json}`, each
recording the image, the judge and the platform by sha256.

**Reproduced across two independent runs on different builds** (the second after the geometry
assertions were added), identical case for case, including the audit arm's numbers. Scored
image `d796157d62e40ecd…`, judge `c5e5872372b392de…`, compiler `3e0ade8b6898db5e…`, qemu
`b1903a3d4160db18…`.

| case | `php-virtual-zend-exact` (fault) | site the fault symbolizes to | ASAN's own frame |
|---|---|---|---|
| CRASH-110 | **CAUGHT** | `ext/standard/url.c:132:11` | `php_url_parse` @ url.c:132 |
| CRASH-016 | **CAUGHT** | `ext/standard/formatted_print.c:592:10` | formatted_print.c |
| CRASH-066 | **CAUGHT** | `ext/standard/var_unserializer.c:390:14` | var_unserializer.c |
| CRASH-091 | **CAUGHT** | `ext/standard/html.c:656:33` | html.c |
| CRASH-102 | **CAUGHT** | `ext/standard/string.c:1332:8` | `php_strspn` @ string.c:1332 |
| CRASH-073 | **CAUGHT** (by the audit arm) | `??:0:0` — inside the copy primitive | memcpy ← `_estrndup` ← `php_url_parse` |

| arm | control `emalloc-overread` | declared | cases |
|---|---|---|---|
| `php-virtual-zend-rounded` | complete — "ran past the over-read" | complete | 5 MISSED, 1 CAUGHT |
| `php-virtual-zend-exact` | fault, cause 28, **at its own probe** | fault | 5 CAUGHT, 1 NO-READING |

**Both arms' controls did exactly what `arms.json` declares, and the fault arm's did so at the
probe rather than somewhere else** — which is the distinction that the first version of this
harness could not make, and the reason it briefly scored a dead arm as a controlled one.

### Against the pre-registration

The prediction was CAUGHT × 6 on the exact arm and MISSED × 6 on the rounded one. Two cells
came out differently and both are explained, not excused:

- **CRASH-102 is CAUGHT on the ROUNDED arm too**, at `string.c:1332`, the same line ASAN
  names. Predicted MISSED; the prediction was wrong because `REAL_SIZE` rounding has no cover
  to offer this case. The object is already 8-aligned, so the rounding yields **zero slack**
  and the 1-byte over-read faults in both arms. The physical column reached this conclusion
  first and `engine/FINDINGS.md` records it: the case is scored **CAUGHT (no pair)**, like the
  `global`-axis CRASH-005, with the safeguard that the fault must land in the defect's own
  file. The virtual column reproduces that, at the same function and line. What this costs is
  honest to state: for CRASH-102 the matched pair proves nothing, and the catch rests on the
  site attribution alone.
- **CRASH-073 needed a third arm, and got one.** Its over-read is performed *by the copy*:
  `ret->path = estrndup(s, ue - s)` with `ue - s` three bytes too long, so the fault lands
  inside the tag-preserving `memcpy` (`musl-capstone/runtime/string_bounds_safe.c`), which
  carries no line info — hence `??:0:0`. The judge refuses it, correctly: a fault in a shared
  copy primitive is attributable to no case by site, and *"it is also the shape of a bound the
  port itself got wrong"* (`php_capstone_zend_hier.c`, which says so because `_erealloc`'s
  narrow-pointer bug once scored as a catch this way). The physical column does not attribute
  it by site either — it builds a third, diagnostic arm (`-DZEND_CAP_ESTRNDUP_AUDIT`,
  `run-triggers.sh:137-154`) that compares the requested length against what the SOURCE
  capability authorises and halts with both numbers. On this profile that word is *printed*
  rather than encoded in `badaddr`, since `php_fault_report` has a stderr to use.

  **The audit arm reproduces the physical column's numbers exactly:**

  | | who | requested | authorised | over by |
  |---|---|---|---|---|
  | physical (`engine/FINDINGS.md:1053`) | 1 (`_estrndup`) | 3 | 2 | 1 |
  | virtual (this run) | 1 (`_estrndup`) | 3 | 2 | 1 |

  `requested = 3` is ASAN's own "READ of size 3" for this case. With that evidence the fault
  becomes `attribution: "probe"` and CRASH-073 is **CAUGHT**; without it the judge returns
  `unattributed`, and `app/test-run-cases.py` asserts both outcomes from the same transcript —
  so the audit arm demonstrably changes the verdict rather than decorating it.

### Phase A's exit criterion, honestly scored

The criterion was *"the virtual arm reproduces the physical arm's verdicts on the reachable
cases, with each fault at the same function and line"*. On the spatial axis — the 6 cases this
phase can reach — **it is met: 6 of 6 CAUGHT, five at ASAN's own file and line and the sixth
attributed at the allocator boundary with the physical column's exact numbers.** The rounded
arm is 5 MISSED and 1 CAUGHT, the one being CRASH-102, which no allocator control can mask.

Both arms' controls behaved as `arms.json` declares, and the fault arm's fired *at its probe*.

The 18 temporal cases are not in Phase A's scope and this run says nothing about them — not
MISSED, nothing. That is the allocator seam, and it is next.

## Phase B — BUILD RESULT (2026-10-15): real PHP builds, links and runs, 145 TUs, nothing dropped

`app/build-virtual.sh realphp` builds **PHP**, not an engine with a SAPI of ours standing in for
it: the Makefile's own `PHP_GLOBAL_OBJS` + `PHP_CLI_OBJS`, **145 translation units** — `main/`
(26), `main/streams/` (9), `sapi/cli` (2), `TSRM/` (3), `regex/` (4), all 65 of `ext/standard`
and 39 of `Zend/`. The `php` case keeps working beside it and remains Phase A's column.

It runs. `php -n hello.php` in the guest produces output **byte-identical** to native glibc
PHP 5.0.0 from the corpus's own pristine build, `print_r` indentation included (`diff` clean,
12 lines each), with `phpversion()` = 5.0.0, `php_sapi_name()` = cli, and `count`, `print_r`,
`parse_url` coming from `basic_functions.c`'s real registration rather than a table of ours.
`php -n -v` exits **1**, and native PHP 5.0.0 exits 1 too — the build reproduces PHP's own
quirk rather than papering over it.

**The whole PHP-side change is one compiler flag and two configure answers.** Measured, not
estimated: 143 of 143 TUs compiled on the *first* attempt, and the one failure was neither a
capability problem nor PHP's.

| what | why | cost |
|---|---|---|
| `-D__off64_t=off_t` | `php_config.h` sets `COOKIE_SEEKER_USES_OFF64_T`, whose branch in `main/streams/cast.c:100` declares the seeker as `(void *, __off64_t *, int)` — a POINTER, which is *exactly* musl's `cookie_seek_function_t`. musl simply has no glibc-INTERNAL `__off64_t`, and defines `off64_t` as `off_t`. | none; the alias is type-correct and `fopencookie` is USED, not disabled |
| `HAVE_RES_NMKQUERY 0`, `HAVE_RES_NSEND 0` | musl 1.2.5 has `res_init`/`res_mkquery`/`res_send`/`res_query`/`res_search` and `dn_expand`/`dn_comp`, but not the reentrant `res_n*`. `ext/standard/dns.h:26` derives `HAVE_DNS_FUNCS` from exactly these two. | `dns_get_record()` and `PHP_MINIT_FUNCTION(dns)` drop out, the way PHP already does on any platform without a reentrant resolver. No corpus case uses them. |

### Four things the plan got wrong, and the plan said to settle them in code

1. **`-UHAVE_MMAP` would have done nothing.** `php_config.h` defines every `HAVE_` *un*conditionally
   and carries not one `#ifndef` guard, so a `-U` on the command line is overridden by the header.
   The flag would have been silently ineffective — and it was not needed anyway: **the SDK has
   `mmap` and `munmap`.** `build-virtual.sh` now asserts the ten symbols the config claims,
   against the archives, so an SDK that loses one fails at build rather than at link.
2. **`mail.c` and `proc_open.c` did not need dropping.** The SDK has `fork`, `vfork`,
   `posix_spawn`, `popen` and `system` — this profile is a Linux process. Dropping them also
   does not work: `basic_functions.c` references `zif_proc_open`, `zm_startup_proc_open` and
   `zm_info_mail`, so the exclusion breaks the function table. **Nothing is dropped.**
3. **`fopencookie` is present too**, so the filtered-stream `FILE*` cast is kept.
4. **The include path cannot correct the config.** `Zend/zend_config.h:1` and
   `TSRM/tsrm_config.h:1` include it as `"../main/php_config.h"` — relative to the including
   FILE, which bypasses every `-I`.

So the tree is a **full mirror**: real directories, symlinked leaves, with `main/php_config.h`
(and, in seam mode, `Zend/zend_alloc.c`) the only real files. "Byte-identical to the corpus" is
then not a claim to audit — it is the filesystem, and the build asserts no other real file
exists. Mirroring only the *top* level does **not** work and fails quietly: with `$T/Zend` a
symlink to the corpus directory, `zend_config.h`'s own `"../main/php_config.h"` resolves against
the link TARGET and walks back out to the pristine config, so the correction is ignored and only
the link tells you. That cost one debugging cycle and is recorded so it costs no more.

### A real toolchain limitation: `-fcommon` crashes the Capstone backend

PHP 5.0.0's Makefile compiles with **`-fcommon`** (`CFLAGS_CLEAN`), and `-fcommon` aborts clang
on 8 TUs: *"Unknown section kind"*, `UNREACHABLE` at `TargetLoweringObjectFileImpl.cpp:631`,
reached from `CapstoneAsmPrinter::emitGlobalVariable` (`CapstoneAsmPrinter.cpp:1492`) through
`SelectSectionForGlobal`. There is no ELF section kind for a **common** symbol, which makes
sense for this target: a common symbol has no definite size or section until link, and a
capability bound needs both.

**Not a stale toolchain** — checked rather than assumed: there is no commit touching
`llvm/lib/Target/Capstone/` or `TargetLoweringObjectFileImpl.cpp` since this clang was built,
so the staleness the env gate reports is unrelated churn. (The gate still cannot tell a
revision-only drift from a source drift; that remains open.)

Consequence: `-fno-common`, which turns every tentative definition PHP relied on into a hard
duplicate. There is exactly **one**, and it is dead: `yytext`, defined in both generated flex
scanners. In each file line 16 maps `yytext` to the prefixed name, line 349 `#undef`s it (which
also wipes any `-Dyytext=` from the command line — that attempt failed), line 478 emits a bare
`char *yytext;`, and line 501 redefines `yytext` as `SCNG(yy_text)`, so every real USE goes to
the scanner globals and the bare word is never read. `llvm-nm` confirms **no object references
it undefined**. It is localized with `llvm-objcopy --localize-symbol` in the INI scanner, so the
file left untouched is PHP's lexer — the one whose codegen the corpus cases exercise — and the
build asserts the no-cross-reference property first. Not `--allow-multiple-definition`, which
would also hide the next duplicate, and that one might be real. (`__llvm_static_cap_gct_begin`
and `_end` appear in four objects but are `W`, so they were never a problem.)

### One image, every arm

`zend_cap_mode()` reads `PHP_CAP_MODE` from the environment, so the seam composes with real
`sapi/cli` without a line of ours — Phase A's SAPI took the mode as `argv[3]`, which `sapi/cli`
owns. `run-virtual.py` grew per-run `NAME=VALUE` assignments (and an `--env`), so the spatial
pair and the three temporal columns are runtime modes over **one 5.08 MB image**, and an arm
cannot differ in anything but the mode word. All four modes complete `hello.php` and report
distinct mode lines — the revocation-sanity control, on real PHP:

```
ZEND_CAP_MODE 0 exact=0 revoke=0     ZEND_CAP_MODE 4 exact=0 revoke=0
ZEND_CAP_MODE 6 exact=0 revoke=1     ZEND_CAP_MODE 7 exact=1 revoke=1
```

The image is **5.08 MB**, which is why the physical column could never have run this: the domain
ceiling was `code + .bss + declared ≤ ~4 MiB`.

### First look at the 7 OS-dependent cases — NOT yet a scored measurement

Pre-registration has **not** been pushed for Phase B, so these are attribution checks against
ASAN's own recorded sites, not judged verdicts. The scored run with per-arm controls out of
`arms.json` is the next step, and it is what the numbers in any table must come from.

Six cases, each at `PHP_CAP_MODE=0` (control) and `7` (bound exact, revocation on), argv exactly
the corpus's own `php -d memory_limit=512M <case>.php`:

| case | ASAN's site | this run | reading |
|---|---|---|---|
| CRASH-110 | `ext/standard/url.c:132` | `url.c:132`, cause 28 | Phase A's spatial catch REPRODUCES on real PHP |
| CRASH-086 | `ext/standard/basic_functions.c:2135` | `basic_functions.c:2135`, cause 24 | newly reachable; same line |
| CRASH-099 | `Zend/zend_execute_API.c:389` | `zend_execute_API.c:389`, cause 24 | newly reachable; same line |
| CRASH-112 | `Zend/zend_execute_API.c:389` | `zend_execute_API.c:389`, cause 24 | newly reachable; same line, and its `requires: writable /tmp` is satisfied |
| CRASH-008 | `ext/standard/datetime.c:440` | `__stpcpy+44`, cause 28 | fault is real but NOT attributed — see below |
| CRASH-084 | `ext/standard/basic_functions.c:1381` | completed on BOTH arms | out of the denominator — see below |

Every control completed. CRASH-086 landing in `basic_functions.c` is the point of the exercise:
that file is PHP's function-table registration, and the physical port could not link it at all.

**099 and 112 share a line, and that is correct rather than suspicious.** Two cases faulting at
one instruction is the signature of a port artefact — it is how the physical port produced four
false CAUGHTs — so it was checked against the oracle before being believed: **ASAN reports
`_zval_ptr_dtor Zend/zend_execute_API.c:389` for both**, because both are zval use-after-frees
that surface at `(*zval_ptr)->refcount--`. The suspicion was right to raise and wrong in fact.

**CRASH-008 is the CRASH-073 category, not a miss.** `datetime.c:440` is
`strcat(Z_STRVAL_P(return_value), tmp_buff)`, so the over-write is performed BY a shared copy
primitive, and the pc lands in musl-capstone's bounds-safe `__stpcpy` with no line info. The
harness already has the mechanism for this — `COPYCASE` plus a diagnostic audit arm that halts
at the allocator boundary with the requested length and what the source capability authorises —
and CRASH-008 needs it. Until then it is `NO-READING unattributed`, which is the gate working.

**CRASH-084 is out of the denominator on a libc behavioural difference**, and this is a finding
rather than a failure. The case is bug #67399: `putenv("=")` then `putenv("=foo")`. The chain
needs the first call to install a string with an EMPTY variable name into `environ`, so that the
second call's `zend_hash_del` → `php_putenv_destructor` → `efree(pe->putenv_string)`
(`basic_functions.c:922`) leaves `environ` pointing at freed memory for the scan at
`basic_functions.c:1381` to read. glibc's `putenv` finds `'='` at index 0 and calls
`__add_to_environ`, which installs it. **musl's does not** — `src/env/putenv.c`:

```c
int putenv(char *s) {
    size_t l = __strchrnul(s, '=') - s;
    if (!l || !s[l]) return unsetenv(s);     /* "=" gives l == 0 */
    return __putenv(s, l, 0);
}
```

so `"="` is routed to `unsetenv` and never enters `environ`. The precondition for the
use-after-free does not exist on this libc, so there is nothing to catch — not a Capstone miss
and not a port defect. Native pristine PHP also exits 0 silently here, which is consistent with
the population being ASAN-blind. This is the same KIND of resolution as CRASH-057: out of the
denominator on evidence, named, with the evidence cited.

### What Phase B has not done yet

- **The scored measurement.** Pre-register to the lane branch, then run all 31 through
  `run-cases.py` with the arms and controls `arms.json` declares, and judge with `verdicts.py`.
  Nothing above is a verdict.
- **CRASH-008's audit arm** (the `COPYCASE` route), and **CRASH-092 / CRASH-100**, which need a
  second process in the guest; CRASH-100 additionally needs shrink-stack.
- **`PHP_VIRTUAL_SHRINK_STACK=1` measured.** The flag is wired and off by default. It is its own
  variable and the 23 known cases get re-measured with it on before anything is said about the
  stack class.
- **The 24 cases Phase A measured, re-measured on real PHP.** Phase A's verdicts were taken
  with our SAPI; agreement on real PHP is what makes the two columns comparable, and CRASH-110
  is the only one checked so far.

## Phase B — PRE-REGISTRATION for the scored run on real PHP (written 2026-10-15, before the run)

Pushed to the lane branch before the measurement and cited by hash in the result commit, per the
root `CLAUDE.md` rule. **Some cells below are not predictions and are marked so**: six cases were
already observed in the exploratory attribution run recorded above, and a pre-registration that
quietly re-predicted them would be worthless. The honest unit here is the **25 unseen cases**.

### The population, derived rather than listed

The 31 are not a list in our repo. They are the ASAN-blind population, recomputed from the
corpus's own two result files at run time:

```
results-asan-default.json    outcome != "OK"
results-asan-stockalloc.json outcome == "OK"
```

That reproduces **exactly 31** today. The expected site for each case is also taken from the
oracle rather than from us — the first PHP source frame in ASAN's own stderr — so a catch is
compared against what ASAN said, not against a table we maintain and could bend.

### Arms

One image, five runtime modes, because `zend_cap_mode()` reads `PHP_CAP_MODE` from the
environment:

| arm | mode | bound | revocation |
|---|---|---|---|
| `php-real-stock` | 0 | `REAL_SIZE` | none (the loan is never returned) |
| `php-real-block` | 4 | `REAL_SIZE` | mallocng, per block |
| `php-real-object` | 6 | `REAL_SIZE` | + the seam's per-`efree` retire |
| `php-real-exact` | 7 | the caller's request | as mode 6 |

Spatial pair: `object` (control, PHP's own rounding) vs `exact` (fault). Temporal triple:
`stock` / `block` / `object`. Same gates as the seam measurement — the image must contain the
revoke encoding, the arms must report distinct mode lines, and revokes must order
`stock == 0 and block == 0 and object > 0`.

**The spatial arms' bound control had to change, and the change is a weakening worth naming.**
Phase A's `emalloc-overread` control lived inside our SAPI: `emalloc(9)`, write 9, read index 9,
which completes on the rounded arm and faults on the exact one. Real PHP owns `main()`, so that
control cannot run in-image, and PHP exposes no way to over-read a byte from a script. The
replacement, declared in `arms.json` as `seam-bound-probe`, is `probe-seam` stage 7 — built from
the same `zend-cap-*.c` sources, doing exactly that sequence, and ending in a fault by design.
It is declared **identically on both spatial arms**, because it sets its own modes internally:
it establishes that the seam's bound logic separates the two arms, **not** that a given image is
running a given arm. What establishes the arm is the `ZEND_CAP_MODE` line the image itself
prints, which the runner gates on. Pretending one probe did both jobs is the kind of control
that reads as holding while proving nothing.

`reached` is derived differently from Phase A and more strongly. Phase A read a stage bit out of
our own SAPI; real PHP has no such bit, so the control arm's **stdout is compared against native
pristine PHP 5.0.0 from the corpus's own build**. A case whose control arm does not reproduce
the oracle's output byte for byte did not run the trigger, whatever its exit status, and is
`NO-READING not-reached`.

### Already observed — NOT predictions (exploratory run, this session)

| case | observed | status |
|---|---|---|
| CRASH-110 | `url.c:132`, cause 28, mode 7 | Phase A's spatial catch reproduces |
| CRASH-086 | `basic_functions.c:2135`, cause 24 | ASAN's own line |
| CRASH-099 | `zend_execute_API.c:389`, cause 24 | ASAN's own line |
| CRASH-112 | `zend_execute_API.c:389`, cause 24 | ASAN's own line |
| CRASH-008 | `__stpcpy+44`, cause 28 | `unattributed` until the COPYCASE audit arm exists |
| CRASH-084 | completed on both arms | `out-of-denominator`: musl's `putenv("=")` has `l == 0` and routes to `unsetenv`, so the empty-named string never enters `environ` and the UAF has no precondition |

### Predictions for the 25 unseen cases

**Temporal (17), at `stock` / `block` / `object`.** Phase A's seam measurement caught all 17;
the prediction is that real PHP reproduces it, with the same split between cases the platform's
own allocator catches and cases that need the per-object retire.

- MISSED / MISSED / **CAUGHT** — the nine `AG(cache)` absorbs:
  CRASH-010, 024, 042, 047, 065, 074, 080, 103, 151
- MISSED / **CAUGHT** / **CAUGHT** — the seven the block-level revoke also reaches:
  CRASH-003, 004, 025, 035, 067, 140, 160
- **CAUGHT** / **CAUGHT** / **CAUGHT** — CRASH-022, whose freed block is overwritten with scalar
  data, so the tag is gone regardless of who revokes

**Spatial (5), at `object` (control) / `exact` (fault).**

- MISSED / **CAUGHT**: CRASH-016, 066, 091
- **CAUGHT** / **CAUGHT**: CRASH-102 — its object is already 8-aligned, so `REAL_SIZE` has no
  slack to offer it, and it is caught on the rounded arm too
- CRASH-073: `NO-READING unattributed` by site, **CAUGHT** only via the audit arm, because
  ASAN's own site for it is `Zend/zend_alloc.c:403` — inside the allocator

**Out of the denominator (1).** CRASH-057: `results-pristine-default.json` records `SIGNAL:11`
for it alone among the 24, so the pair has no premise. Note ASAN's site for it is also
`zend_alloc.c:403`.

**Not reached (2).** CRASH-092 (`requires: network: working loopback`) and CRASH-100
(`requires: network: an HTTP listener on 127.0.0.1:8599`) both need a second process in the
guest, which this run does not provide. CRASH-100 additionally needs shrink-stack on, and
`PHP_VIRTUAL_SHRINK_STACK` stays **off** for this run so the comparison with Phase A holds.

**Totals predicted: 23 CAUGHT of 28 in the denominator** (31 less CRASH-057, 084, and the
unattributed CRASH-008), with 092 and 100 `not-reached` inside that denominator.

### Where this prediction is most likely to be wrong, and why

Stated in advance so a miss is informative rather than retrofitted:

1. **Real PHP's allocation history is not the engine's.** `main/`, `php_ini.c`, the streams layer
   and `basic_functions.c`'s registration all run before the trigger, so a case's freed block can
   land in a different `AG(cache)` bucket than it did in Phase A. `CACHE_INDEX` is
   `REAL_SIZE(size)>>3` with 11 buckets × 256 entries, so a block that reached `free()` in Phase
   A may be absorbed here, or the reverse. **The nine-vs-seven split is the cell most likely to
   move**, and a case shifting between `block` and `object` is a finding about the cache, not a
   regression.
2. **Node-table pressure is higher.** Real PHP already reports `nodes≈12,200` with
   `node_growths=3` on a benign workload, against the engine's ~10,300. A cause-30 resource
   fault lands at an arbitrary pc and is otherwise indistinguishable from a catch, so any run
   whose high-water comes within `CV_NODE_RESERVE` of capacity must be refused rather than read.
3. **`_zval_ptr_dtor` is ASAN's site for nine of the 31.** Site-based attribution is therefore
   coarse here: nine cases legitimately share one line, so "landed at the expected site" is
   weaker evidence for those than for CRASH-110's `url.c:132`. The per-arm controls, not the
   site, are what separate them.
4. **CRASH-035 was the case the seam's own porting defect hid** (`ub_vspprintf` using `malloc`
   while `zend_error` hands the buffer to a zval PHP later `efree`s). Real `main/` supplies the
   genuine `vspprintf`, so this cell is being measured against PHP's own code for the first time.

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

## The allocator seam — RESULT (2026-10-13): built, and the temporal axis separates three ways

The design below was built as specified. What follows is what it measured, what it got wrong
first, and the two things it found that were nobody's prediction.

### THE RESULT: 17 of 17 on the temporal axis, and 9 of them need the seam

18 cases, one image, three runtime columns. Bundles under
`$CAPSTONE_TMP_ROOT/php-final/temporal/<arm>/`, each recording the image, the judge, the arm's
mode and the platform by sha256.

| column | mode | CAUGHT | MISSED | NO-READING |
|---|---|---|---|---|
| `php-virtual-seam-stock` | 0 | 1 | 16 | 1 |
| `php-virtual-seam-block` | 4 | 8 | 9 | 1 |
| `php-virtual-seam-object` | 6 | **17** | 0 | 1 |

The denominator is **17**, not 18: CRASH-057 is out of it, and that is the corpus's own call
rather than ours (below). So the object column is **17 of 17**.

**Nine cases are caught by the per-object retire and MISSED by block-level revocation** —
CRASH-010, 024, 042, 047, 065, 074, 080, 103, 151. Those are the frees `AG(cache)` absorbs:
the block is never handed back, so mallocng has nothing to revoke, and only a retire at
`_efree` sees them. That is the population the corpus describes as hidden from ASAN by the
size-class cache, and it is the seam's contribution stated in the only way that can be
checked — against a column that differs from it in the retire alone.

**Seven are caught by block-level revocation too** — CRASH-003, 004, 025, 035, 067, 140, 160 —
so those catches belong to the platform's allocator and the seam merely also sees them. Saying
so costs nothing and is the difference between a result and a claim.

**CRASH-022 is CAUGHT on all three columns, with no pair**, at `ext/standard/array.c:546`, the
line ASAN names. Its freed block is reused and overwritten with scalar data, which clears the
tag, so the stale capability load faults whether or not anything was revoked. The physical
column reached the same conclusion for the same case; what keeps it from being "it faulted
somewhere" is that the fault must land in the defect's own file.

**CRASH-057 is out of the denominator, on evidence.** The pair has a premise — that stock,
unsanitised PHP runs the script to completion, so a control that completes reproduces
`crashes_on_pristine_build = False`. The corpus records **`SIGNAL:11`** for CRASH-057 on
`results-pristine-default.json`, alone among these 24. A control that completed would therefore
prove the port was *not* reproducing the defect, and one that faults says nothing about
detection. `run-cases.py` reads that field rather than inferring it, and the judge's own
`out-of-denominator` is the label.

That is also the honest resolution of the case the physical column could not score. There, the
control faulted inside the port's own hoisted unlink — a port artefact. Here the stock column
reaches the request teardown, which the physical port never ran, and fails the way stock PHP
does: it passes `payload_base + 480` to `_efree`, which the diagnostic names as
`ZEND_CAP_INTERIOR`. Same case, same non-reading, different and better-understood reason.

### Combined with Phase A: 23 of the 23 reachable cases

| axis | cases | caught | how |
|---|---|---|---|
| spatial (hierarchical pair) | 6 | 6 | 5 at ASAN's own file and line, 1 by the audit arm |
| temporal (the seam) | 17 | 17 | 9 by the per-object retire, 7 also by the block, 1 with no pair |
| | **23** | **23** | |

CRASH-057 is the 24th and is out of the denominator. The spatial axis was re-measured after the
fix below and did not move: 6 of 6, the audit arm reporting the same numbers, both controls as
declared.

### The seam found a porting defect that the looser arrangement hid

CRASH-035 failed on every column with `ZEND_CAP_FOREIGN`: PHP freed an address inside no block
the seam owns. The cause was in this port's own SAPI, not in PHP — `ub_vspprintf` allocated its
buffer with `malloc`, and `zend_error` hands that buffer straight to a zval
(`Zend/zend.c:940`), which PHP later destroys with `efree`. Under the hierarchical allocator
the mistake is invisible, because `_efree` reaches a level that did issue the pointer: it came
from the same musl heap. The seam refuses instead of guessing, and the diagnostic named it in
one run.

**That is the fifth time this port's own substitute layer has been the thing under test rather
than PHP** (a sink that discarded its bytes, an error path that discarded its format string, a
missing `zend_register_default_classes`, a missing `zend_deactivate`, and now this). PHP's real
`vspprintf` uses `emalloc`; `app/php_virtual_main.c` now matches it, and CRASH-035 became a
reading on all three columns. `engine/rung_E_domain.c` has the same `malloc` and is left alone
because it is the comparison column — but its CRASH-035 reading is worth re-examining on these
grounds, which is recorded here rather than acted on.

### The seam works, and PHP's `_efree` did not have to change

`zend_alloc.c` is patched with **eight hunks, every one behind `#ifdef ZEND_CAP_SUBLET`**, and
with the define absent it compiles to a **byte-identical object** (`.text` 6572 bytes, `cmp`
clean) — with a positive control confirming the object *does* differ when the define is
present, so the inertness check is not vacuous. `build-virtual.sh inert` is that gate. The
physical column's results and Phase A's hierarchical results are therefore untouched by the
file having been patched at all.

The central claim held. `ZEND_MM` is undefined in this build, so `REMOVE_POINTER_FROM_LIST` is
real and the intrusive list is live — verified against the macro expansions, not assumed. With
the header outliving every payload issued from its block, PHP's own unlink-before-free ordering
already repairs the list while everything it touches is alive. **Probe stage 4 links three
blocks through their headers, retires the middle payload and performs the unlink: it passes.**
So the physical port's hoisted, unconditional `REMOVE_POINTER_FROM_LIST` — the deviation that
cost four false CAUGHT verdicts and still costs CRASH-057 a no-reading there — is not needed
here, which is what the seam was for.

### The first pair LIED, and the third column is the fix

The plan specified two modes, `EXACT` and `REVOKE`, and a temporal pair of mode 0 against mode
2. Run once: **both arms caught CRASH-003 at the same instruction**, `zend_execute.c:669`.

The reason is a property of the profile, not of the case. `__capstone_sublet_free_linear`
rotates the heap's own authority, which is senior to the loan, so **every alias to a block that
reaches `ZEND_DO_FREE` dies whatever this layer does**. "No inner revocation" is therefore not
"stock PHP lifetimes" here, and a catch in such a pair cannot be attributed to the seam rather
than to mallocng underneath it. The loan's return became its own mode bit:

| mode | column | what revokes |
|---|---|---|
| 0 | **stock** | nothing. The loan is deliberately LEAKED and the record left indexed, so a freed block stays readable exactly as on stock PHP. The honest-gap column |
| 4 | **platform** | mallocng, per BLOCK, when PHP actually frees one. No per-object retire |
| 6 | **seam** | both: the payload region at every `_efree`, cache-absorbed or not |

Mode 4 is what makes a mode-6 catch attributable to the seam, and the probe proves the columns
are distinct rather than nominal: on the platform arm a **cache-absorbed** free survives (stage
8) while a **real** free faults (stage 9). Asking what the instrument could not distinguish
*before* the claim, rather than after it, is the only reason this is a three-column table
instead of a two-column overclaim.

### The probe: ten stages, all behaving

`zend-alloc/sublet/probe-seam.c` drives the real `zend-cap-*.c` through the same hooks the
patched `zend_alloc.c` calls — a probe with its own copy of the logic would pass while the
engine broke.

| stage | tests | reading |
|---|---|---|
| 0 | split, write, retire, header still writable, re-issue, free | PASS |
| 4 | three blocks linked through headers; retire the middle, then unlink it | PASS |
| 6 | the stock column: the same stale use with nothing revoked | PASS (completes) |
| 8 | the platform column on a **cache-absorbed** free | PASS (completes) |
| 1 | write through a retired payload alias | FAULT, cause 24 |
| 2 | read through a retired payload alias | FAULT, cause 24 |
| 3 | the AG(cache) hand-back: re-issue, then use the OLD alias | FAULT, cause 24 |
| 5 | realloc moves the payload; the old alias | FAULT |
| 9 | the platform column on a **real** free | FAULT |
| 7 | the bound: index 9 of a 9-byte request, rounded vs exact | survives, then FAULTS |

Stage 3 is the one that matters most for PHP: the re-issued region has the **same address** as
the dead alias, and the old alias stays dead anyway. That is the trap
`docs/design/virtual-capstone/ownership.md` names — *"an allocator that merely reuses bytes
inside one still-live outer allocation does not obtain inner temporal protection
automatically"* — and `AG(cache)` reuse is exactly it.

### Three gates replace the byte-identical-arms check

A pair that is one image with a runtime mode cannot be checked by comparing images, so the
plan's Verification 5 replacements are enforced by the runner, which refuses to write bundles
otherwise: the image must contain the **revoke** encoding at all (it carries 8, plus 9 `mrev`);
the three runs must report **distinct mode words**; and the object column must revoke
**strictly more** than the block column, which must revoke none. Measured: modes 0/4/6,
revokes 0/0/77. `app/test-run-cases.py` drives the whole runner offline over synthetic
transcripts and shows both cross-arm gates *firing* — 23 checks, no emulator, about a second.

### Revocation sanity, kept as an arm-level control

A benign workload with revocation ON must complete, because four use-after-free cases once
reported CAUGHT while all faulting at the same allocator line and only this control saw it.
It completed on all three columns, and on the seam column with 77 revokes and 47 retires
against 52 blocks — so the machinery is sound on a program with no lifetime error in it.

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
bundles exist and are re-judgeable but do not appear in the cross-program catch tables.

**CORRECTION (2026-10-13): the "thin `corpus.json`" this plan proposed is not available, and
it was not checked before proposing it.** `bug-corpora/tools/check-corpus.py` discovers corpora
by `rglob("corpus.json")` and then validates each case against its declared schema. The
closest schema to PHP's shape is `script-trigger` — a defect whose trigger is an interpreter
script rather than a C reduction, which is exactly right — but it requires, per case,
`case`, `upstream_fix`, `title`, `consumer`, `class`, `trigger`, `fidelity`, `arms` and
`status`. So declaring the corpus *is* the restructuring: a declaration without the per-case
provenance either fails the checker or means writing that provenance, and the checker is a
shared gate that currently passes (28 corpora, 327 cases, 0 problems) and must keep passing.

Dropping the idea rather than half-doing it. The verdict bundles are self-describing in the
meantime — each records the image, the judge, the arm's mode and the platform by sha256, and
`run-cases.py --from-record` re-judges them without booting — so nothing is lost but the
cross-program table, which is the follow-on's job. The schema even has a `verdict_bundles` key
waiting for it, and a `fault_sites` case field where `EXPECT_FILE` belongs.

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
