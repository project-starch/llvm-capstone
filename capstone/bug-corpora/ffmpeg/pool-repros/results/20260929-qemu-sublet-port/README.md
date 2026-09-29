# The FFmpeg pool corpus on the Sublet port of FFmpeg's own pools (capstone-qemu, 2026-09-29)

**Question.** The corpus's protected Capstone arm is the buffer-pool port's probe cases 36-38.
- They run FFmpeg's `buffer.c`, but with the pool's payloads served by that port's own allocator
  (`pool-allocator.c`), whose leases carry the revocation.
- FFmpeg's pools themselves were not ported: the storage and its lifetime were the substitute's.
- The Sublet port of FFmpeg's own pools (on dev since `584268d1179f`) ports them. Their storage
  comes LINEAR from the Sublet heap, and the pool's own code revokes a buffer when it returns.

Does that port catch the corpus's three defects?

**Answer: yes, all three, in every cell as registered** (`PREDICTIONS.md` and
`../../runners/sublet-port-expect.txt`, `60491a5b4ad8`).

N = 3 per cell:

| case | poolstock (upstream's pools), defect | poolstock, fix | **poolsublet (Sublet port), defect** | poolsublet, fix |
|---|---|---|---|---|
| 0 af_join | DEFECT-REPRODUCED ×3: the stale pointer reads the new owner's payload | FIXED ×3 | **FAULT ×3**, cause 24, at `case.c:85`, the stale read | FIXED ×3 |
| 1 h264_refs | DEFECT-REPRODUCED ×3: a record past `ref_count` names reissued storage | FIXED ×3 | **FAULT ×3**, cause 24, at `case.c:48`, the stale read | FIXED ×3 |
| 2 vidstab | DEFECT-REPRODUCED ×3: the parked pointer corrupts the new owner | FIXED ×3 | **FAULT ×3**, cause 24, at `case.c:33`, the stale write | FIXED ×3 |

How to read it:

- **Every case is the corpus's `case.c`, unchanged,** in the FFmpeg app port's domain.
  `src/capstone-domain/ffapp_corpus.c` stands in for the corpus driver: the same pool, the same
  calls, and the case prints its own verdict line.
- **The arms differ in one macro**, `FF_SUBLET_POOLS`, plus the port's backend `ffsublet.o`, which
  only the Sublet arm links. The case and driver objects are byte-identical across the arms.
- **The poolstock defect column shows each case created its condition.** The storage was reissued,
  and the stale access reached the new owner.
- **The poolsublet fix column shows the port faults on nothing else**, over the same sequence of
  pool calls. In case 0 the kept reference means the storage is never reissued; in cases 1 and 2 it
  is reissued, and the stale pointer has been dropped.
- **Every fault is named by source line.** Its pc, taken into the image's own addresses, maps
  through `case.c`'s line table to the line that dereferences the stale pointer, and its `value_hi`
  is non-zero: a revoked capability, not an integer.
- **The faulting register was traced for each case**, with the claim audit.
  - `case.c:85` also reads the live `reused->data[0]`. The fault is `lbu a4, 0(s10)` at image
    `0x19630`, where `s10` is `out.extended_data[2]`, the stale pointer, loaded at `0x1950c`. QEMU
    reports `rs1 = x26`, which is `s10`. The live read is a later load, at `0x19658`.
  - At `case.c:48` the faulting base is `ref_list[7].data`.
  - At `case.c:33` it is `td.src`.

## What had to be built

- **`ports/ffmpeg/app/host/build-domain.sh`** builds the corpus under `FFAPP_CORPUS_DIR`, on the
  two pool arms only: fixture `40 + 2 * case + fixed`, with `case.c` compiled with `-g`.
- **`run-safety.sh`** takes an expectations file and a verdict script
  (`FFAPP_SAFETY_EXPECT`, `FFAPP_SAFETY_VERDICT`); its defaults are unchanged.
- **`runners/`** holds the expectations, the runner (`run-sublet-port.sh`: each case's fix image,
  then its defect image, one boot) and the verdict (`sublet-port-verdict.py`).

**The verdict's own controls**, recorded in `result-lines.txt`, section 4:

- Four wrong predictions against the real logs all DIFFER.
- An image with its debug information stripped is an ERROR, not a pass.
- Before any boot, the line table was shown to resolve on every built image.

## What this does not establish

1. **QEMU only** (Q-11): on silicon a stale access retires.
2. **The cases are reductions.** They transcribe the consumer's call sequence; the allocator is
   FFmpeg's own. af_join's and vidstab's REAL code ran under the same port in Track B
   (`ports/ffmpeg/app/results/2026-09-29-trackb-{afjoin,vidstab}/`).
   - h264_refs has only this reduction, because its real code needs a malformed stream.
3. **The three defects are not live at the 9.0.1 pin.** Each case re-introduces the reverse of its
   upstream fix, as the corpus always has.
4. **Infrastructure.** One boot (poolsublet round 1, case 2) stalled before any section, its serial
   log ending in the firmware banner. It was re-run, and the cell counts that run. The guests boot a
   private copy of the rootfs, repaired with `e2fsck`, because the shared one carries ext4 errors.
5. **N = 3 shows repeatability, not independent samples.** The emulator is not run under
   `-icount`: repeats reproduce the outcome, but `value_hi` differs across rounds, and one boot
   stalled on inputs identical to the others'.

## Files

- `PREDICTIONS.md`: the registration.
- `result-lines.txt`: every verdict line, in the order run, with the verdict's own controls.
- `SHA256SUMS`: the images every boot loaded, per arm.
