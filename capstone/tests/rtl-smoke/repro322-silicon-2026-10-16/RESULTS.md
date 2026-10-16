# The SQLite 3.22.0 temporal-bug corpus, control arm, on caplifive_sublet_d93a8d66b: results (2026-10-16)

Pre-registered in README.md, before each boot: boots 1-2 at lane 423d3a56d0bb, boot 3 at 40256f289171. Three boots on
the classic line (buildroot d04bd83, opensbi df0e9a8, capstone-sbi 93440d5, the #3 module). The result lines are in
`results/`.

**Every cell read as pre-registered.** Each boot's k800 control returned 4, and each boot had one boot banner.

| cell | corpus case | image | boot | silicon | the emulator |
|---|---|---|---|---|---|
| base322 | (the 3.22.0 base domain) | a958958b… | 1, 2, 3 | returned, all five markers | identical |
| wschema | 24 | ea36f529… | 1 | returned; `insert rc=11 (database disk image is malformed)`, NOTRAP done | identical |
| blobwrite | 04 | 7bb39642… | 1 | returned; `write rc=0`, NOTRAP done | identical |
| mem5design | 18 | 38c3fe7b… | 1 | returned; `before=2779054080 after=4294967295`, NOTRAP done | identical |
| blobclose | (the legacy domain, not case 03) | bd12d56d… | 1 | returned; NOTRAP done | identical |
| jsonstatic | 17 | 0508d144… | 1 | returned; `cross_rows=6`, `max=[3,4,5]`, `rows=1` | identical |
| jsonroot | 16 | 33e71cce… | 1 | returned; `cross_rows=6`, `max={"bb":[3,4,5]}`, `rows=1` | identical |
| backupattach | 02 | 8fe5a153… | 1 | wedged, mcause 24 at `sqlite3BtreeUpdateMeta+0x48` (`ldc a1, 0x80(a1)`), tval 0 | the same instruction, cause 24, tval 0x80 |
| detachtrig | 05 | 02ca19e2… | 2 | wedged, mcause 24 at `sqlite3DropTriggerPtr+0x154` (`ldc a3, 0x0(a1)`), tval 0 | the same instruction, cause 24, tval 0x0 |
| blobclose03 | 03 | 634b15ac… | 3 | returned; lookaside OFF, `close_v2 rc=0` (zombie path), `blob_close rc=0`, NOTRAP done | identical |
| ctluafmem5 | control uaf-mem5 | 30b37a04… | 3 | returned after `CONTROL uaf-mem5 mark`: the stale read completed | identical |
| ctlboundsmem5 | control bounds-mem5 | 2cb47577… | 3 | returned after `CONTROL bounds-mem5 mark`: the write 1 KiB past a 16-byte block completed | identical |
| backupattachrev | the review's 02 (branch sqlite-review) | 1fc539c1… | 3 | returned; `step-to-done rc=101`, NOTRAP done | identical |

**The refusal record:**
- Boots 1-2: read at each wedge, EMPTY [204:02 205-208:00]. Nothing is revoked in this arm, and no cause-25 denial
  occurred.
- Boot 3: **not read.** No cell wedged, and this runner reads apertures 204-208 cleanly only at a wedge. Every
  per-cell sample after base322 read 0xfe in all three boots, a contaminated running sample. So boot 3's
  pre-registered "EMPTY" was untestable with this instrument, and it is recorded as not read, not as passed.

## What it shows

- **The control arm reads on silicon exactly as on the emulator and as on R-43 v2** (2026-10-01): every returning
  cell's output line for line, and both FAULT cells at the same instruction.
  - The new input (7fc92e19, the sorter's 16-byte alignment) and the new bitstream (d93a8d66b) change no reading.
- **Both FAULT cells are null-pointer faults of the control arm, not detections.**
  - backupattach's load is `ldc a1, 0x80(a1)` with a null base. The emulator's tval is 0x80, the field offset, and the
    external collaborator's review names the pointer `pPage1`.
  - detachtrig's load is at address 0, downstream of a stale read that succeeds in this arm.
  - The RTL reports tval 0 for both. The emulator reports the effective address. Both name the same instruction.
- **The corpus's two memsys5 controls complete on silicon.** memsys5 hands out offsets inside one arena, which nothing
  bounds or revokes. That completes the stock configuration's requirement before a silent case reads MISSED. The
  controls ran in boot 3, on the same bitstream, input and flags as the cases of boots 1-2.

**Folding into the corpus.** Seven cells are images of the corpus's own case.c, byte-identical by
`build-corpus-images.sh` (README, "Boot 3"). blobclose03 is case 03's. The legacy blobclose of boot 1 is not a corpus
case. backupattachrev is a reading of the review branch only.

**Not established:** N = 1 per cell. This is the control arm only: nothing is revoked. Rows outside batch 1 are not
run.
