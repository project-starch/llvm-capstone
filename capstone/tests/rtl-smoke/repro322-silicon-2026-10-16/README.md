# The SQLite 3.22.0 temporal-bug corpus, control arm, on caplifive_sublet_d93a8d66b: pre-registration (2026-10-16, before the boots)

The board campaign's wave 1 item 5 (`docs/plans/2026-10-16-board-campaign-all.md`, sqlite/engine-repros). It repeats
`../repro322-silicon-2026-10-01` on the resident Sublet bitstream, with the images rebuilt from the corrected input.

**Why rebuilt.** The 2026-10-01 images were adapted from sqlite3.c b446615b…, which lacks the sorter's 16-byte
alignment that the virtual bundles' input 7fc92e19… carries. The compiler lane landed that line in
`ports/sqlite/adapt-sqlite-322.sh` (dev 423d13f5b81f). The 2026-10-01 record stays as it was; it is on b446615b.

## The images

`../repro322-silicon-2026-10-01/build-images.sh`, unchanged, on lane b0-silicon-runtime with 423d13f5b81f merged:
- **Adapted input:** sqlite3.c sha256 **7fc92e19a3af61f3961b83447a91f1e1dec06646e343037dfd96a397167700b2**.
  - Built in a fresh temp root. A stale cache at `/tmp/capstone/sqlite-322-adapted` held b446615b; it was moved
    aside first.
  - A first build had picked that cache up through an inherited CAPSTONE_TMP_ROOT. It is void and was never staged.
- **Compiler:** `clang version 22.0.0git (llvm-capstone 612b3ec514c0)`, as 2026-10-01.
- **The cells:** the same cells, rows and entry VAs. `cells.tsv` and `SHA256SUMS` here name the new images.

## The emulator reading (the predictions)

The silicon-mimic QEMU: capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_GP_FABRICATE=0`, the board's own
SQLite host `sqlite_host.user` 2c9e82d101b48160 and the shared QEMU images. Records are in
`~/capstone-artifacts/qemu-pass/<sha256>`.
- **Boot 1:** the seven returning cells in board order. Each prints exactly 2026-10-01's emulator reading: base322 all
  five markers, and every other cell `<tag> NOTRAP done`.
- **Boots 2 and 3:** each FAULT cell alone (on the board, each FAULT cell is instead the last cell of its boot; see
  below).
  - backupattach halts with cause 24 at `sqlite3BtreeUpdateMeta+0x48`;
  - detachtrig halts with cause 24 at `sqlite3DropTriggerPtr+0x154`;
  - the same instructions as 2026-10-01's.

## The boots

`drivers/board-r322.sh` with `R322_FOLDER` pointing here (a new override; the default stays 2026-10-01's), on:
- the shared FPGA buildroot d04bd83, opensbi df0e9a8, capstone-sbi 93440d5 (which carries 2dcd3a5);
- the classic #3 module;
- `REFUSAL_RECORD=1`. The Sublet bitstream carries the record: boot s10a10 latched it on 2026-10-16 (arm probe
  resolved DEAD, revnode id 55).

**The runner tree.** The stage runner, `preflight-board-run.sh` and `board-watchdog.sh` are the main clone's (tree
fa5b23aeb9b8, 2026-10-05). Every classic-line run on 2026-10-16 used the same ones.
- The driver runs from a scratch tree that links those three to the main clone, so its identity check compares the
  runner that runs.
- Their manifest sha256 is c6d705df4948a45a, with run_sqlite_stages_fpga.py ce9ac34a9e288418, refusal_record.py
  e8f37d0af86f3a80 and transcript.py 517b2127880fbec3.
- The runner has to run in the clone that holds the submodules, which is the main clone. Dev's runner differs from
  it only in refusal_record.py's prefix list and the preflight's DTS table. It was also unimportable from
  2c2b01d62002 until 638493a31981.
- This preflight predates the Sublet name, so the run declares `PREFLIGHT_DTS_NODES=65536`, as the 10-16 acceptance
  boots did.

| boot | cells, in order |
|---|---|
| 1 | k800, base322, wschema, blobwrite, mem5design, blobclose, jsonstatic, jsonroot, **backupattach** |
| 2 | k800, base322, **detachtrig** |

## Pre-registered readings

- **k800** returns 4. Otherwise the boot is void.
- **Every returning cell** prints the emulator's lines above and returns.
- These are 2026-10-01's silicon readings on R-43 v2 (`../repro322-silicon-2026-10-01`, "Results").
- **Each FAULT cell** wedges with **cause 24** in the trap log, at the emulator's instruction. The refusal record
  reads EMPTY: nothing is revoked in the control arm, so cause 25 is not expected.

**The informative readings are divergences, as in 2026-10-01:**
- a FAULT cell that returns: the RTL keeps a tag that memsys5's integer freelist stores clear in the emulator;
- a returning cell that faults: silicon refuses an access the emulator runs through;
- cause 25, or a LATCHED refusal record: a refusal with nothing revoked.

**New against 2026-10-01:**
- the bitstream (d93a8d66b, not R-43 v2);
- the corrected input (7fc92e19), which moves the sorter's records to 16-byte alignment. Only cells that sort can
  show it.

**Not established here** (as 2026-10-01):
- N = 1 per cell;
- this is the control arm only: nothing is revoked;
- rows 1, 3, 6, 11, 12, 13, 17 and 22 are not run, and batch 2 (rows 10, 19 and 24) is not built.

## Boot 3, pre-registered before it, after boot 1 had begun: the cells built from the corpus's own case.c

**Why.** `../repro322-silicon-2026-10-01/build-images.sh` compiles the cells from `ports/sqlite`.
- Seven of those sources are byte-identical to the corpus's `bug-corpora/sqlite/engine-repros/<case>/case.c`.
- blobclose's is not. `ports/sqlite/sqlite_blobclose_domain.c` is the legacy standalone domain, 200 diff lines from
  case 03's `case.c`, which was ported onto `REPRO322_MAIN` on 2026-10-05.
- So boots 1-2's blobclose is 2026-10-01's cell and not corpus case 03.

`build-corpus-images.sh` here builds every cell from the corpus's `case.c`, with the same recipe, input and VAs.
- **The seven** come out byte-identical to the board images above (wschema ea36f529…, blobwrite 7bb39642…,
  mem5design 38c3fe7b…, jsonstatic 0508d144…, jsonroot 33e71cce…, backupattach 8fe5a153…, detachtrig 02ca19e2…).
  So boots 1-2's readings of those seven are readings of the corpus's cases, by image.
- **blobclose03**, from case 03, is a new image, 634b15ac….

**The build's allocator configuration**, read from the built images, not from the flags:
- `sqlite_heap` is 0x40000 bytes: `SQLITE_HEAP_SIZE=262144`, build-sqlite-silicon.sh's default;
- `sqlite3Config` holds szLookaside 0 and nLookaside 0, so **lookaside is off**. The silicon build harvests
  `-DSQLITE_DEFAULT_LOOKASIDE=0,0` from build-sqlite-capstone.sh's define block.
- blobclose03's own output says the same: `lookaside status_rc=0 slots_in_use=0 high_water=0`.

**Boot 3's cells**, after k800 and base322, all expected to return:

| cell | source | image | the emulator's reading (silicon-mimic QEMU, `qemu3/`) |
|---|---|---|---|
| blobclose03 | case 03's case.c | 634b15ac… | `blobclose close_v2 rc=0 (zombie path taken…)`, `blobclose NOTRAP done` |
| ctluafmem5 | controls/control_uaf_mem5.c | 30b37a04… | `CONTROL uaf-mem5 mark`, then returns: the freed block's stale read completes |
| ctlboundsmem5 | controls/control_bounds_mem5.c | 2cb47577… | `CONTROL bounds-mem5 mark`, then returns: the write 1 KiB past a 16-byte block completes |
| backupattachrev | the external collaborator's unlanded review of case 02 (branch sqlite-review, ece80f69219f; case.c e0ca83551f1e3241) | 1fc539c1… | `backupattach step-to-done rc=101`, `backupattach NOTRAP done` |

- The two controls are the corpus's. The stock configuration requires both to complete before a silent case can
  read MISSED. Both access memory inside memsys5's one arena, which nothing bounds or revokes.
- backupattachrev is a reading of the review branch only. It is recorded as such, and it does not stand for case 02
  unless the review lands.

**Pre-registered:** k800 returns 4; every cell prints the emulator's lines above and returns; the refusal record
reads EMPTY. Any fault in boot 3 is a divergence.

**The FAULT cells' fault addresses, added here for boots 1-2:**
- QEMU: backupattach `tval = 0x80`, detachtrig `tval = 0x0`.
- 2026-10-01's silicon reading had tval 0 for both.
- backupattach's access is `ldc a1, 0x80(a1)`, and QEMU's tval 0x80 puts its base at 0. That is a null pointer; the
  review names it `pPage1`.
- detachtrig's access is `ldc a3, 0(a1)` at address 0, downstream of its stale read.
- **Both are null-pointer faults of the control arm, not detections.** The external collaborator's review reaches
  the same conclusion for cases 02 and 05.
