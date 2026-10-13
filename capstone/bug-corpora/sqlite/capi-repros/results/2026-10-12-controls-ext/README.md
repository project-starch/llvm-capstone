# SQLite 3.22.0 lookaside: parent, sibling and successor controls (2026-10-12)

This is SQLite's row of the pre-registration
[`docs/plans/2026-10-12-nested-lifetime-controls-extended.md`](../../../../../docs/plans/2026-10-12-nested-lifetime-controls-extended.md)
(5d336c73931c). It uses the control definitions of
[`2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md).
The allocator is the per-connection **lookaside**, which the memsys5 controls of
[`../2026-10-11-controls/`](../2026-10-11-controls/) could not reach.

The cases, probes, builder and runner are in
[`../../controls/lookaside/`](../../controls/lookaside/). The full record is `results.json`.

**Outcome: every prediction held, on both arms, in the run of record (r1) and in two repeats (r2,
r3). That is 3 boots per image per arm.**

| control | release (lookaside as shipped) | protected (`sublet-3220000-memsys5.patch`, lookaside half) | prediction |
|---|---|---|---|
| parent | completed. A's former range was read through `lookaside.pStart + 129600`. The typed link read returned `pNext` = D's address (tag 1), then A's bytes. Write and read-back ok. The link was restored, and the lookaside then reissued A, then D | completed. Same, through `lookaside.pStart + 118800`, `pNext` = D (tag 1). Write and read-back ok. A, then D reissued | held, both |
| sibling | completed, B intact, write and read-back ok | completed, B intact, write and read-back ok | held, both |
| successor | completed. C = A at the first request | completed. C = A at the first request | held, both |
| successor, stale read | completed. The read through old A returned C's bytes (`CHECK stale-read-returns-C-bytes ok`) | **fault, cause 25**, at `stale-A-read` (`control-successor/control.c:71`). pc = image offset `0x673b0` in `run_case`, which symbolizes to `control.c:71:3`. Fault address = A's start | held, both |

The protected stale read, r1, as QEMU and the launcher logged it:

    [CAPSTONE] Cap mem access on revoked capability: pc = 3fb14573b0, rs1 = x10, imm = 0, cursor = 3fb175fe90, bounds = (3fb175fe90, 3fb1760340)
    capstone-exec: domain fault cause=25 pc=0x3fb14573b0 address=0x3fb175fe90 … sha256=85d31156ece7a13865c7770ac44a07fc13f9a962e21af4b4b07f5bc410e91756 image=./la-control-successor-stale.dom

The bounds in that line are A's slot (1200 bytes). r2 and r3 logged the same image offset, the
same cause, and a fault address equal to A's start.

## A came from the lookaside

Every block a control takes goes through `la_alloc()` (`la_common.h`), which prints, for example
(protected parent, r1):

    LA sz=1200 nSlot=100 bDisable=0 bMalloced=1 pStart=0x3f90f435c0 pEnd=0x3f90f60a80 hits=0 ptr_bytes=16
    SLOT A addr=0x3f90f605d0 offset=118800 index=99 rem=0 within=1 hit_delta=1 dbmallocsize=1200
    BLOCK A addr=0x3f90f605d0 base=0x3f90f605d0 end=0x3f90f60a80
    CHECK A-is-lookaside-slot ok

`CHECK <name>-is-lookaside-slot ok` requires all four of these:
- the address is in `[pStart, pEnd)`;
- the offset from `pStart` is a multiple of the slot size;
- the lookaside hit counter `anStat[0]` rose by exactly 1;
- `sqlite3DbMallocSize` returns the slot size.

On every arm and run, every A, B, D and C passed. On the protected arm the block's capability
bounds are also exactly the slot, `[A, A+1200)`, as `lookasideIssue` derives them.

**The check can fire.** `slot-check-negative` is a separate program, outside the pre-registered set,
with one boot per arm. On the same connection it requests:

| request | release | protected |
|---|---|---|
| 128 B | `ok`: `within=1 hit_delta=1 dbmallocsize=1200` | `ok`: `within=1 hit_delta=1 dbmallocsize=1200` |
| 2000 B (larger than a slot) | `FAILED`: `within=0 hit_delta=0 dbmallocsize=2048` | `FAILED`: `within=0 hit_delta=0 dbmallocsize=2000` |
| 128 B with the lookaside disabled | `FAILED`: `within=0 hit_delta=0` | `FAILED`: `within=0 hit_delta=0` |

The images are `665e84c1…` (release) and `6c5d9fcb…` (protected).

## Addresses (r1)

Buffer bases differ from boot to boot. Offsets within the buffer were identical in r1, r2 and r3.

| run | lookaside buffer `[pStart, pEnd)` | A | B / D | C |
|---|---|---|---|---|
| release parent | `0x3fbe343020`–`0x3fbe362f10` | `0x3fbe362a60` (slot 108) | D `0x3fbe3625b0` (107) | — |
| release sibling | `0x3f99342690`–`0x3f99362580` | `0x3f993620d0` (108) | B `0x3f99361c20` (107), adjacent | — |
| release successor | `0x3f88b42650`–`0x3f88b62540` | `0x3f88b62090` | — | `0x3f88b62090` |
| release successor-stale | `0x3fbbf428e0`–`0x3fbbf627d0` | `0x3fbbf62320` | — | `0x3fbbf62320` |
| protected parent | `0x3f90f435c0`–`0x3f90f60a80` | `0x3f90f605d0` (99), bounds `[…05d0, …0a80)` | D `0x3f90f60120` (98) | — |
| protected sibling | `0x3fa7b42c30`–`0x3fa7b600f0` | `0x3fa7b5fc40` (99) | B `0x3fa7b5f790` (98), adjacent | — |
| protected successor | `0x3faa342bf0`–`0x3faa3600b0` | `0x3faa35fc00` | — | `0x3faa35fc00`, bounds `[…fc00, …00b0)` |
| protected successor-stale | `0x3fb1742e80`–`0x3fb1760340` | `0x3fb175fe90` | — | `0x3fb175fe90` |

**The release arm:**
- every block's capability carries the whole 1 MiB heap array's bounds, `lookaside.pStart`'s
  included;
- the lookaside has 109 slots, because `sqlite3MallocSize` reports memsys5's 128 KiB block.

**The protected arm:**
- each slot's capability is bounded to its 1200-byte slot;
- `pStart`, `pBlock` and `pLife` each cover the 120000-byte buffer;
- the lookaside has 100 slots, because the patch reports the request.

On both arms the lookaside hands out slots from the top of the buffer.

## Probes

All lines are in `capi-repros/controls/lookaside/`. Each probe printed `done` in every completed
run.

| control | probe | file:line |
|---|---|---|
| parent | A-fill (write) | `control-parent/control.c:39` |
| | parent-read (128 B through `pStart`) | `control-parent/control.c:52` |
| | parent-link-read (typed pointer load) | `control-parent/control.c:54` |
| | parent-write | `control-parent/control.c:63` |
| | parent-readback | `control-parent/control.c:64` |
| | parent-restore (bytes) | `control-parent/control.c:66` |
| | parent-restore-link (typed pointer store) | `control-parent/control.c:67` |
| sibling | A-fill, B-fill (writes) | `control-sibling/control.c:36`, `:37` |
| | sibling-read | `control-sibling/control.c:42` |
| | sibling-write | `control-sibling/control.c:45` |
| | sibling-readback | `control-sibling/control.c:46` |
| successor | A-fill (write) | `control-successor/control.c:43` |
| | C-write | `control-successor/control.c:66` |
| | C-read | `control-successor/control.c:67` |
| | stale-A-read (stale image only) | `control-successor/control.c:71` |

## Positive controls

The stale read is the positive control for the protected rejection, and it held:
- **release** completes at the same probe and reads C's bytes;
- **protected** faults with cause 25. The last probe line is `stale-A-read`, with no `done`. The pc
  resolves to `control.c:71`, and the fault address is A's start.

The patch is live:
- every protected image carries 2 CDERIVE and 1 CREVOKE encodings, and every release image
  carries 0 of each;
- under the patch each slot's bounds are the slot, and `pLife` covers the lookaside buffer.

The judge in `run-controls.py` was negative-tested against r1:
- with `--swap-arms` the stale pair came out not held, while the other six, whose predictions are
  the same on both arms, stayed held;
- edited copies of the protected sibling log each came out not held: one with `CHECK … FAILED`,
  one with a probe that has no `done`, one with exit 1, and one with no `RETURNED`.

The lookaside-slot check fires on non-slot blocks (above).

## Images and platform

| image | release sha256 | protected sha256 |
|---|---|---|
| la-control-parent.dom | `1e9ac29315e04687092f1d1155f26c187d072db0c3ac27f0fce46dcd4cdb0f53` | `6c7e0020bd9dcda6c2eca9aeaebb169470a6435dfa68df7abfec13bbf8dcab85` |
| la-control-sibling.dom | `5bd1741f937110fd5e6b67c99753c3e0682741fbc5642720848a2b8e1da306ba` | `ead8cfd35b76b1fdc8bf7a7027b0f1789b74454e7288633c6f8ca7884e72b83b` |
| la-control-successor.dom | `1e66e273715c2b478128716ad238dba5b55180ee84b8069c4a98fb644a62218e` | `1063bb9a91e4c2b0a19f8cf717fc4a07aaf178790b4c091bb0ee3f2aea251e83` |
| la-control-successor-stale.dom | `649ee9ce4f95e03409e69b827a321c61584fb5b77c0fa586135a7189737615e9` | `85d31156ece7a13865c7770ac44a07fc13f9a962e21af4b4b07f5bc410e91756` |
| staged sqlite3-capstone.c | `19ceadb5d103cab190dc4a76b3a54f457495fa1cf28ec5befcd0d5c1fa0c4d66` | `892dce7120747b9d8fbd007bfe41b6517d4f6069a693e3014bebf49db1e550c4` |

The staged engines are byte-identical to the memsys5 controls' engines. The launcher's fault line
carries the protected stale image's sha256.

**Platform:** the registered virtual profile. Every component hash was measured on the build host
for this run, and each equals the 2026-10-11 run's:
- QEMU `af37cc32`, launcher `3975314a`, module `a1c6cb6b`;
- exact bounds, one hart, 2 GiB.

The kernel, firmware, rootfs, compiler and SDK libraries are in `results.json`. The SDK was rebuilt
from this snapshot with `runtime/virtual/build-sdk.sh`. Its libc and runtime archives are
byte-identical to the 2026-10-11 ones.

**Source:** llvm-capstone `5d336c73931c` (HEAD of `ports/nested-controls-extended`), plus the
**uncommitted** files under `controls/lookaside/`. Their hashes are in `results.json`. The images
were built from a `git archive` of HEAD with those files copied in.

**Build and run host:** p12, 2026-10-11 21:58–21:59 UTC (05:58–05:59 local on 2026-10-12).

## Wall times

- One boot per image, 8 VMs in parallel.
- r1: 9.2 s for all 8. r2: 9.2 s. r3: 9.5 s.
- Each boot took 8.1–9.2 s.
- Builds: 19 s (release) and 20 s (protected), in parallel.

## Deviations and notes

- **A request size of 128 B, not a full slot.** Every probe accesses 128 bytes of the 1200-byte
  slot, as in the memsys5 controls. On the protected arm the bounds are the whole slot.
- **The parent link is read and restored with typed pointer accesses.** The lookaside's in-band
  link is a pointer, unlike memsys5's integer links. Restoring it with a byte copy would leave it
  untagged, and the next request to the lookaside would then fault on both arms. So the parent
  control also reads the link with one typed load (`parent-link-read`) and restores it with one
  typed store (`parent-restore-link`). Both go through `pStart`, and both completed on both arms.
- **The parent control frees two slots.** No slot had been handed out before A (`hits=0` after the
  open), so the free list starts empty, and A freed alone would get the link NULL. Freeing D before A
  makes A's link a real address (D), and the run shows the free list's head was D just before A's
  release (`CHECK setup-D-heads-free-list-before-A ok`). The read through the parent therefore checks
  a value the lookaside wrote, and a NULL could not stand in for it.
- **The lookaside slot check is an addition to the template.** Its positive control
  (`slot-check-negative`) is not part of the pre-registered set, and ran one boot per arm.
- **Which corpus these join.** The pre-registration names `capi-repros`. The heap configuration,
  builder and patch are those of the memsys5 controls, which in turn took them from the security
  corpus.
- N = 3 boots per image per arm (r1, r2, r3), with the same outcome each time. No run was
  discarded.
