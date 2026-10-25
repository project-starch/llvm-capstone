# The plain corpora on silicon: pre-registration (2026-10-16, before any board run)

This is the board half of `docs/plans/2026-10-15-bug-corpus-on-fpga.md`. It covers the 72 plain case directories of
memcached, FFmpeg and tshark: 46 plain-heap, counting FFmpeg 13, a duplicate of 02, and 26 plain-temporal. Each is
built for the silicon ABI and run on `caplifive_sublet_d93a8d66b`.

The per-case predictions are the helper lane's table, `silicon-predictions/2026-10-16-plain` (dev 510126308bb6),
column "child at the request size, 16-byte placement". This file adds the board side:
- the arms and the images;
- the silicon-mimic QEMU reading of every image;
- the per-row board prediction derived from it.

## The arms

The lead's decisions, as stated to the lead on 2026-10-16 when the lead asked to start (none was picked explicitly,
so this lane's stated recommendations were taken):
- **(a)** predicted spatial differences are accepted, not padded away;
- **(b)** the malloc arm is level0 plus per-object lifetimes;
- **(c)** the plain corpora come first.

| arm | heap | temporal |
|---|---|---|
| `silicon-lifetimes` | level0 built with `CAPSTONE_LEVEL0_LIFETIMES`: every object is a CDERIVE child of one group over the arena, at exactly its request size on a 16-byte placement. | `free` is CREVOKE. A request that is not exact at its placement keeps plain narrowing and no lifetime (none in this corpus). |
| `silicon-level0` | level0 as it ships: per-object bounds by SHRINK. | `free` only marks the block free, so temporal cases are MISSED by construction. This is the control arm. |

## How the images are built

`tools/run-silicon-cases.py build` runs `runtime/silicon/build-b0-hello.sh` per case:
- case.c + shared/driver.c at -O0 (the corpus convention, so the crossing stays on the labelled probe);
- gp-captable, full LTO, the pool carve;
- check-repr under full+renorm, OVERLAP 0;
- no-fp-gate.

The arm's controls are `run-capstone-domain.py`'s CONTROL_C (clean, oob, uaf, subobj), built the same way.
`SHA256SUMS` in each arm's directory names every staged image. `runs.txt` is the run list b0run.sh's `corpus-<arm>`
rung reads.

## The silicon-mimic QEMU

- capstone-qemu d35af47b163c: the physical CDERIVE/CREVOKE decode, and Q-04's MOVC behaviour as silicon.
- GP_FABRICATE=0; firmware fw-b1-top d5c57ee765c9.

silicon-lifetimes (`qemu-and-predictions.json`):
- **Controls:** clean RETURNED; oob CAUGHT, cause 7 in ctl_write_probe; uaf CAUGHT, cause 24 in ctl_read_probe; subobj
  RETURNED.
- **Rows:** 72 of 72 as predicted.
  - 46 spatial CAUGHT at the labelled probe: cause 5 on reads, 7 on writes.
  - 26 temporal CAUGHT at the labelled probe with cause 24.
  - Every FIXED arm printed VERDICT FIXED and exited 0.

## The board prediction

**silicon-lifetimes:** every row as QEMU read it, with the same outcome and the same probe. One cause differs:
- temporal rows and the uaf control read **25** on silicon, not QEMU's 24;
- QEMU's ldc untags a capability whose node is revoked, while the RTL forwards it, and the use then traps 25
  (ISSUES Q-11);
- the stale-access arm on this bitstream read exactly that (dev, sublet-bare-2026-10-15).

A spatial cause may differ only if the RTL reports a bounds fault differently from QEMU. It is read, not assumed.

**silicon-level0** (its QEMU reading is recorded before its own boot):
- spatial CAUGHT as above;
- temporal DEFECT-REPRODUCED;
- uaf RETURNED.

**Judge:** `tools/run-silicon-cases.py board` applies run-capstone-domain.py's classify and verdicts.py's rules to the
board transcript:
- CAUGHT only at the labelled probe or a declared fault site;
- a buggy row counts only after its FIXED run printed VERDICT FIXED and exited 0;
- no row counts unless the boot's controls behaved as the arm declares.

**What a failure would say:**
- **The first deliberate fault (control oob) ends the board session:** the process-ABI fault path does not return to
  Linux on silicon. No domain has faulted on purpose under capstone-exec on silicon before. Every later run is lost;
  the rung's per-run markers show where.
- **A temporal row MISSED on silicon-lifetimes:** CREVOKE did not end the object's capability, or the object took the
  inexact fallback. The image's CDERIVE count and `__capstone_level0_unexact` are read first.
- **A spatial row MISSED:** the RTL's bounds for that object are wider than the model's. Ask the helper lane to check
  the geometry before calling it a silicon result.

## Amendment before the silicon-level0 boot (2026-10-16 04:52)

**The silicon-lifetimes boot has run** (firmware 0409fc297ed52e26, 04:29-04:49; the result is in RESULTS.md):
- 72 of 72 as predicted by outcome and site;
- temporal cause 25, as predicted;
- spatial cause **28**. The RTL's OUT_OF_BOUNDS (capstone_unit.anvilh:321) is the code QEMU reports as 5 / 7.
  qemu-and-predictions.json had carried QEMU's 5 / 7 in that arm's per-row cause field. The text above left the
  spatial cause open, so this is recorded, not re-scored.

**silicon-level0, pre-registered now:**
- QEMU (`silicon-level0/qemu-and-predictions.json`): controls clean RETURNED, oob CAUGHT 7, uaf RETURNED (reissued),
  subobj RETURNED; 72 of 72 as predicted: 46 spatial CAUGHT at the probe, 26 temporal DEFECT-REPRODUCED.
- **On silicon, firmware f1cfbeece7760d6b:**
  - the same verdicts;
  - spatial cause 28, from the lifetimes boot;
  - uaf RETURNED.
