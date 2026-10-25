# SQLite 3.22.0 memsys5: parent, sibling and successor controls (2026-10-11)

This is SQLite's row of the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(53325bba0d5c). The cases, probes, builder and runner are in [`../../controls/`](../../controls/).
The full record is `results.json`.

**Outcome: every prediction held, on both arms, in the run of record (r2) and in a repeat (r3).**

| control | release (memsys5 as shipped) | protected (`sublet-3220000-memsys5.patch`) | prediction |
|---|---|---|---|
| parent | completed. Read of A's former range through `mem5.zPool + 1032192` = memsys5's link `{next 15874, prev -1}`, then A's bytes. Write and read-back ok | completed. Through `mem5.zPool + 972288` = `{next -1, prev -1}`, then A's bytes. Write and read-back ok | held, both |
| sibling | completed, B intact, write and read-back ok | completed, B intact, write and read-back ok | held, both |
| successor | completed. C = A at the first request | completed. C = A at the first request | held, both |
| successor, stale read | completed. The read through old A returned C's bytes | **fault, cause 25**, at `stale-A-read` (`control-successor/control.c:73`). pc = image offset `0x672a4` in `run_case`, which symbolizes to `control.c:73`. Fault address = A's start | held, both |

## Addresses (r2)

Pool bases differ from boot to boot. Offsets within the pool were identical in r3.

| run | pool (`mem5.zPool`) | A | B / G | C |
|---|---|---|---|---|
| release parent | `0x3f93681fe0` | `0x3f9377dfe0` (index 16128) | G `0x3f93779fe0` (15872) | — |
| release sibling | `0x3f92281860` | `0x3f9237d860` (16128) | B `0x3f92379860` (15872), not A's buddy | — |
| release successor | `0x3f8e281790` | `0x3f8e37d790` | — | `0x3f8e37d790` |
| release successor-stale | `0x3f9de81a10` | `0x3f9df7da10` | — | `0x3f9df7da10` |
| protected parent | `0x3fa16824d0` | `0x3fa176fad0` (15192), bounds `[…fad0, …fb50)` | G `0x3fa176fb50` (15194) | — |
| protected sibling | `0x3fbbe81d50` | `0x3fbbf6f350` (15192) | B `0x3fbbf6f3d0` (15194), A's buddy | — |
| protected successor | `0x3fbb281c80` | `0x3fbb36f280` | — | `0x3fbb36f280`, bounds `[…f280, …f300)` |
| protected successor-stale | `0x3fa2281f10` | `0x3fa236f510` | — | `0x3fa236f510` |

On the release arm every block's capability carries the whole heap array's bounds. On the
protected arm each block's capability is bounded to its 128-byte request, and the pool's covers
1 MiB. The two arms place A at different indices because the patch takes 4 bytes per atom for
`aReq`, which changes the pool's atom count. Under the patch A and B are buddies, and on the
release arm they are not. On both arms they are children of the same pool.

## Probes

All lines are in `capi-repros/controls/`. Each probe printed `done` in every completed run.

| control | probe | file:line |
|---|---|---|
| parent | A-fill (write) | `control-parent/control.c:43` |
| | parent-read | `control-parent/control.c:58` |
| | parent-write | `control-parent/control.c:69` |
| | parent-readback | `control-parent/control.c:70` |
| | parent-restore (write) | `control-parent/control.c:72` |
| sibling | A-fill, B-fill (writes) | `control-sibling/control.c:41`, `:42` |
| | sibling-read | `control-sibling/control.c:47` |
| | sibling-write | `control-sibling/control.c:50` |
| | sibling-readback | `control-sibling/control.c:51` |
| successor | A-fill (write) | `control-successor/control.c:45` |
| | C-write | `control-successor/control.c:68` |
| | C-read | `control-successor/control.c:69` |
| | stale-A-read (stale image only) | `control-successor/control.c:73` |

## Positive controls

- The stale read is the positive control for the protected rejection, and it held:
  - **release** completes at the same probe and reads C's bytes (`CHECK stale-read-returns-C-bytes ok`);
  - **protected** faults with cause 25. The last probe line is `stale-A-read` with no `done`. The pc
    resolves to `control.c:73`, and the fault address is A's start, so the fault is at the probe and
    nowhere else.
- That the patch is live: the protected image carries CDERIVE and CREVOKE, and the release image
  carries neither. Under the patch each block's bounds are its request, and the pool's are the 1 MiB
  heap.
- The judge in `run-controls.py` was negative-tested against these logs:
  - judging each log against the other arm's prediction turns the stale pair to not held;
  - each of the following edits to a completed log also turns it to not held: one `CHECK … FAILED`,
    a probe with no `done`, exit 1, or no `RETURNED`.
- What the parent read can show is weaker on the protected arm. There the free list was empty
  before A's free, so memsys5's link is `{-1, -1}`. On the release arm `next` is a real index
  (15874). In both cases those 8 bytes differ from the pattern the control wrote into A, so the read
  saw memsys5's own store.

## Images and platform

| image | release sha256 | protected sha256 |
|---|---|---|
| control-parent.dom | `e378fc0541f20f5af2fddae21f292ae1ab22b2d392b86766d1b88cd7322e4aa2` | `86b841fa28db7571ad0a949f382080c941882aba5c1657ef18b4dfc5b4eb25dd` |
| control-sibling.dom | `32a893b22c0da1f7ac40b8292813e7f45ecb5af49720c640220ec113a2458bd9` | `c96f3fb81b814eae9f7cfdc4c773b587f951b0b5b559293f31802b30c11078d7` |
| control-successor.dom | `5554beed5af6281c2c2c206a336fd7210a18cca2210d5384701d1c8ef1483023` | `ced6750303596473d874111a60d34671bf7211d802f80fcbd2b2a101ff7f3cd2` |
| control-successor-stale.dom | `4ff2b868a6e44e0e17fa08e16bfe926f8f5377750f8e52afc8541da1f6a906c2` | `8549d2dfa549be4233719df4b5a2acac6a340e596cdbb79ee50f0ad880f2b3f1` |
| staged sqlite3-capstone.c | `19ceadb5d103cab190dc4a76b3a54f457495fa1cf28ec5befcd0d5c1fa0c4d66` | `892dce7120747b9d8fbd007bfe41b6517d4f6069a693e3014bebf49db1e550c4` |

The launcher's own fault line carries the same sha256 for the protected stale image.

The platform is the registered one: QEMU `af37cc32`, launcher `3975314a`, module `a1c6cb6b`, the
virtual profile, exact bounds, one hart and 2 GiB. The kernel, firmware, rootfs, compiler and SDK
libraries are in `results.json`. The SDK was built from the snapshot with
`runtime/virtual/build-sdk.sh`. Its libc and runtime archives are byte-identical to the ones in
`ports/sqlite/app/results/2026-10-11-same-answers`.

**Source**: llvm-capstone `53325bba0d5c` (HEAD of `ports/nested-lifetime-controls`), plus the
**uncommitted** files under `capi-repros/controls/`. Their hashes are in `results.json`. The images
were built from a `git archive` of HEAD with those files copied in.

## Wall times

- One boot per image, 8 VMs in parallel on the build host.
- r2: 24.5 s for all 8 boots; each boot took 19.7 to 24.2 s.
- r3: 18.5 s for all 8; each boot took 12.1 to 18.2 s.
- Builds: each arm's four images took 40 s (release) and 44 s (protected), the two arms in parallel. The SDK build was not timed.

## Deviations and notes

- **Run r1 was discarded.** The runner's judge compared the per-arm prediction table with a string,
  so every run predicted to complete printed `held=False`. Its observations matched r2's. The judge
  was fixed and negative-tested, then r2 (run of record) and r3 (repeat) were booted from the same
  images.
- **One translation unit per control.** The corpus builds each case against a separately compiled
  engine object. Here the staged `sqlite3-capstone.c` is `#include`d into each control, because
  `mem5` is static to the amalgamation and the parent control has to name `mem5.zPool`. The flags
  are the corpus's flags for a core-group case, plus `-g` for pc-to-line resolution. The engine
  source is staged by the corpus's `build-virtual.py`, unchanged.
- **Local runner.** `capi-repros` has no Capstone runner. The memsys5 corpus runner
  (`engine-repros/shared/run-virtual.py`) attributes a fault by function only, so
  `controls/run-controls.py` is a separate runner. It boots through the shared
  `runtime/virtual/run-staged.py`, unchanged.
- **Which corpus these join.** The pre-registration names `capi-repros` for SQLite. The security
  table's memsys5 cases are `engine-repros`; these controls use that corpus's heap configuration,
  builder and patch.
- `sqlite3_malloc` does not use the lookaside. The patch's lookaside half is therefore not exercised
  here; the memsys5 half is.
- N = 2 boots per image (r2 and r3), with the same outcome each time.
