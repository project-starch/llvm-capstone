# PostgreSQL memory-context defects

Eight upstream use-after-free and double-free defects in PostgreSQL's memory
managers, reduced to programs that run against PostgreSQL's **own** allocator:
`aset.c`, `mcxt.c` and `slab.c` from the pinned 17.0 release, compiled
unmodified but for the capability-ABI and Sublet patches the port applies. The
consumers are reduced, the allocator is not.

The layout is the contract in
[`../../cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md):
one directory per case, `NN_<upstream-fix>_<slug>/`, holding a `case.c` that is
a complete translation unit, a `case.json` of machine-readable claims, and a
`PROVENANCE.md`. Case numbers are dense from zero. One program per case,
because a capability fault ends the run and a case that provokes one cannot
also report results beside it.

## The cases

| # | upstream fix | shape | layer | live in pin |
|---|---|---|---|---|
| 0 | `1f5b6a5e5d` | double free through a stale array entry | aset | **yes** |
| 1 | `3549ffb6af` | stale pointer to a recreated object | aset | **yes** |
| 2 | `83ce20d671` | stale pointer to a recreated object | aset | **open** |
| 3 | `ed394c4bdf` | alias freed through a sibling | aset | **yes** |
| 4 | `727bc6ac33f6` | alias freed through a sibling | aset | **yes** |
| 5 | `9d5ce4f1a00a` | stale pointer into a reset context | aset | **yes** |
| 6 | `a61592253e` | stale pointer into a deleted ancestor | aset | **yes** |
| 7 | `9e0b4b1ab5` | same-address reuse from a fixed-size free list | slab | **yes** |

## Shapes

`case.json`'s `shape` must be one of these, so that two cases sharing a
reduction class are visibly siblings rather than accidentally similar.

| shape | what makes it its own class |
|---|---|
| double free through a stale array entry | the stale access IS the second free, so there is no read probe |
| stale pointer to a recreated object | the object is destroyed and immediately recreated; the holder is never updated |
| alias freed through a sibling | one object, two pointers; freed through one and read through the other |
| stale pointer into a reset context | no free at all — a bulk context reset ends the lifetime |
| stale pointer into a deleted ancestor | an ancestor context dies, taking a grandchild's arena with it |
| same-address reuse from a fixed-size free list | reuse is deterministic, so the successor lands at the identical address |

## The four systems

Two independent capability systems are compared here, each with and without its
temporal mechanism. They are different architectures under different emulators
and different operating systems; what is compared is what each mechanism denies
at the same eight defects, not the systems' performance.

| system | what it is |
|---|---|
| **Capstone** | this project's capability architecture on RISC-V. LLVM fork (clang 22, `capstone64-unknown-elf`), QEMU fork (`virt-capstone`); programs run as freestanding **domains** loaded by a Linux guest |
| **Sublet** | a linear slot discipline over Capstone's revocation, applied through the manager's lifetime hooks |
| **CheriBSD** | CHERI-RISC-V purecap, CheriBSD 15.0-CURRENT under QEMU, clang 17. Its own temporal safety is malloc quarantine plus a revoker sweep |
| **PoisonCap** | a published CHERI extension for temporal safety through poison capabilities, [arXiv:2605.13210](https://arxiv.org/abs/2605.13210). Run against its published research artifact: patched kernel, libc, QEMU and LLVM |

The same `case.c` builds for both targets. That is what makes the comparison
one: it is the same source, not two reimplementations.

## Arms

| arm | target | what it establishes |
|---|---|---|
| `spatial` | Capstone domain | the sequence completes without protection |
| `sublet` | Capstone domain | fault at the labelled read probe |
| `cheribsd` | CheriBSD purecap, libc revocation ON | whether the layer below sees these defects |
| `poisoncap-spatial` | CheriBSD purecap, adapter invalidation off | the matched control for the arm below |
| `poisoncap-protected` | CheriBSD purecap, adapter invalidation on | SIGPROT at the labelled read probe |
| `native-detect` | host, `before.c` | written for cases 3 and 7 only |

The two protected arms name an **instruction**, not merely a fault: the runner
resolves the labelled probe from the child's own map plus the target ELF and
publishes the address, so a relink cannot turn the check into a tautology.

There are three UNPROTECTED arms and they are not interchangeable. `spatial`
and `cheribsd` hand out offsets inside one arena capability -- the backing
allocator narrows nothing per chunk. `poisoncap-spatial` bounds every chunk
exactly, so it is the strictest spatial baseline available on this hardware,
and it still lets all eight through. That is the arm the "bounds do not cover
this class" claim should rest on.

## Building and running

The port builds the cases; case material does not live inside a port. Pass the
corpus root and the port builds one program per case for whichever target it
is configured for:

    -DPG_CORPUS_DIR=<repo>/capstone/bug-corpora/postgres/mmgr-repros

Programs are named as the contract names run artifacts --
`03-live-parts-stale-alias`, and `03-live-parts-stale-alias-sublet` for a
domain arm -- so an archived result tree stays readable away from the corpus.

**The CheriBSD arms** run through
`ports/postgres/memory-contexts/host/cheribsd/poisoncap/run.py`. Build with
`-DPG_POISONCAP=ON` for the PoisonCap arms, without it for the plain CheriBSD
arm; `--runtime-revocation on|off` selects whether the guest's own libc
revocation is active, and the shared runner's ABI probe verifies the setting
took effect, so it cannot silently do nothing.

**The Capstone domain arms** run through `shared/run-defects.py`. It needs four
things that are NOT the defaults and each of which cost a failed attempt to
find:

    # pexpect is not in the system interpreter; the platform venv has it
    /tmp/capstone/venv/bin/python3 shared/run-defects.py OUT \
      --domain-build BUILD_DOMAIN --linux-build BUILD_LINUX

    export CAPSTONE_QEMU_BINARY=<tree>/capstone/capstone-qemu/build/qemu-system-riscv64
    export CAPSTONE_LLVM_BUILD_DIR=<tree>/llvm/build-rel      # NOT cmake-build-debug
    export CAPSTONE_BUILDROOT_DIR=<tree>/capstone/caplifive-buildroot

One of those failures printed `FAIL ... spatial` and `FAIL ... sublet cause=0
pc=0`, which reads like "Capstone caught nothing" and was a guest that never
started. Check for a `serial.log` in the run directory before believing a
domain result: an arm that produced none did not run.

**PoisonCap and the guest's own revocation need the platform fix.** Without
`bug-corpora/cpython/pymalloc-repros/platform/mrs-poison-retire.patch` the two
together panic the guest kernel (`share->excl` in `vm_map_lookup`) at the first
arm that sweeps; eight unprotected arms before it are unaffected. With the
patch applied the same command pairs all eight. Any result taken that way is
"PoisonCap with that fix" and must say so.

## What the four systems do, measured 2026-09-21

| system | what it acts on | caught |
|---|---|:--:|
| Capstone | bounds and tags; no lifetime event | **0 / 8** |
| **Sublet** | the chunk's return to the sub-pool | **8 / 8**, cause 24 |
| CheriBSD default | `free()` → quarantine → revoker sweep | **0 / 8** |
| **PoisonCap** | the same return: poison, then sweep | **8 / 8**, SIGPROT 162 |

The two that catch these defects are the two that listen for the moment the
INNER allocator takes the storage back. The other two listen for an event that
never happens: PostgreSQL asks the system allocator for a block once and hands
out chunks from it itself, so between the `pfree` and the stale read there is
nothing on the layer they watch. The same reason ASan is silent here.

Seven of the eight faults land on the labelled stale access, compared against
an address the run resolves from the child's own map and the target ELF. Case 0
has no probe -- its stale access is a second `pfree` -- and faults in
`GetMemoryChunkMethodID`, where the manager reads the revoked chunk's header.
Both protected systems land in that same function, established independently.

### Provenance of these numbers

- **Capstone and Sublet**: `virt-capstone` QEMU, one domain image per case and
  arm. Two sixteen-arm runs gave 14/16 each with the failures in DIFFERENT
  arms, and the four affected arms passed when re-run singly; the harness is
  flaky at roughly one arm in eight, signature `runner_exit 1` with no
  `serial.log`. Every arm has passed, but not all in one run.
- **CheriBSD default and PoisonCap**: the same CheriBSD purecap guest, libc
  revocation ON in both, verified per run by the ABI probe reporting
  `runtime_revocation=1`.
- **The platform carries the local libc fix**
  (`bug-corpora/cpython/pymalloc-repros/platform/mrs-poison-retire.patch`,
  libc `6726fdb0…`). Without it PoisonCap and the guest's own revocation panic
  the kernel; the CheriBSD-default arm gives 0/8 either way, measured on both.

## What is NOT established

- **The Capstone `spatial` and `sublet` arms have not been re-run** since the
  corpus was split into one program per case. The build for them is written
  but has not been configured on a host with a built Capstone toolchain, so
  two of the four arms are empty for these eight cases.
- **`native-detect` for six of the eight.** Cases 3 and 7 have a `before.c`;
  the others do not.
- **No cost claim.** The counters in each run (`sweeps`, `poison_bytes`,
  `mapped_bytes`, `padding_bytes`) are adapter measurements under a
  deliberately conservative synchronous-sweep policy, in an emulator. They are
  not a PoisonCap overhead figure and must not be quoted as one.

## What IS established, and how

All eight cases carry a `live_proof` in their `case.json`: an inspection at the
pin with file and line, cross-checked against the upstream commit's date, since
every one of these fixes postdates the 17.0 release of 2024-09-26.

Cases 1 and 2 are two DIFFERENT defects at the same site and were initially
attributed the wrong way round. `83ce20d671` (2024-12-04) fixes the callee's
stale local in `parallel_vacuum_reset_dead_items`; `3549ffb6af` (2025-10-03)
fixes the caller's unrefreshed field in `dead_items_reset`. Reading the pinned
tree alone could not separate them — it shows the pre-fix state but not which
fix addresses which half — and the inventory in
`docs/ref/postgres-nested-allocator-defects.md` calling them "same, 2024
instance" reinforced the error. Their `distinguishing` fields now say which is
which.

The PoisonCap arms paired on all eight on 2026-09-20, over two independent
boots for cases 1-7 and a third after case 0's control was repaired. Each
protected arm's trap PC is compared against the probe address that the run
resolves from the child's own map and the target ELF, so the check cannot
become a tautology across a relink — the two boots differ by exactly 0x8 in
every address and still match.
