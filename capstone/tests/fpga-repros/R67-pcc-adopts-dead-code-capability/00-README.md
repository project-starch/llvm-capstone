# R-67: a jump through a code capability whose node has ended runs its target

Siblings, for a reader who arrived with a different symptom:
- R-44 is the same optimistic adopt at the CPMP trackers (S/U-mode accesses), and it recorded this one as latent.
- R-35 is the same adopt at the LSU, fixed for M-mode data accesses.
- Q-29 is the fetch check's order and scope.
- None of them is this defect.

## What happens

The core runs in M with capmode set.
1. A code capability C is made on its own node.
2. C's node is ended with crevoke. The lcc validity query then reads 0, so the revocation has completed.
3. `cjalr ra, C, 0` jumps to C.
4. The instruction at the target RETIRES. Here that is a store of a marker word, and no trap is taken.

The ruled behaviour is a refusal at the target's first fetch:
- ISSUES Q-29: the PC-capability check refuses a dead PCC with 25.
- QEMU's `revoked_pcc` and `revoked_cached_pcc` tests refuse it.

## Why (source, e7af48457; read, not inferred)

The PC check at commit raises 25 when the tracker says the PCC's node is dead (`core/commit_stage.sv:389-391`). The
tracker, though, never asks:

```
// On domain change (CALL/RETURN/CAPENTER), assume valid — DYN unit already checked
if (cap_type_t'(pc_cap.metadata.cap_type) != NOT_CAP
    && pc_cap.metadata.revnode_id != pc_revnode_id_q) begin
  pc_revnode_id_d    = pc_cap.metadata.revnode_id;
  pc_revnode_valid_d = 1'b1;
```
(`core/commit_stage.sv:399-404`).
- The comment's premise holds for CALL and RETURN, which query the node in the DYN unit.
- It does not hold for CJALR. CJALR is a FLU op that checks only the tag, so a new code capability installed by
  CJALR is adopted as live without any query.
- The tracker is then cleared only by a later broadcast of its index, and C's broadcast happened BEFORE the jump.

The same lines are in every sealed bitstream's tree: c721c1a2a, dcc6daf6e, e7af48457, 603528b8b, 325966ee4.

## Readings

**Simulation** (testharness, memory delay 12; `tests/lv-t1-*`, conformance rule T1).

| arm | what it does | bitstream 3's RTL (4dcb67e4e = e7af48457's tokens) | with the fix (1fc2cd5aa) |
|---|---|---|---|
| t1-fire | crevoke C, then cjalr | **the marker is written, 0 traps** | 1 trap, cause 25, mepc = the target, no marker |
| t1-norevoke | the same without the crevoke | marker, 0 traps | the same |
| t1-untagged | the jump through an untagged register | 1 trap, cause 24 | the same |
| t1-oob | live C, the target outside its bounds | 1 trap, cause 28 | the same |

- The controls show the PC check works on bitstream 3. Only its liveness arm is never armed for a CJALR target.
- The kit images, simulated on bitstream 3's tokens and on ObjBounds (038b540a3), read the same vectors as each
  other, which are in `PREREG.md`.
- ObjBounds reads the hole by design, because it removes the PC check's 25 altogether.

**Silicon: the hole is present on both sealed bitstreams**, as pre-registered (`PREREG.md`, graded by
`grade/r67-grade.py`, rc 0 each, image hashes and resident names checked):

| bitstream | board, date | t1-fire | controls |
|---|---|---|---|
| bitstream 3 (caplifive_vp2c_e7af48457) | board 1, 2026-11-03 19:25-19:35 | **HOLE**: validity 0 after the crevoke, marker written, 0 traps | all three exact |
| ObjBounds (caplifive_vp2c_603528b8b) | board 2, 2026-11-04 04:52-05:11 | **HOLE**, the same vector | all three exact |

- On ObjBounds the first pass's t1-norevoke read NO DATA. That was an infrastructure fault: the GDB image load timed
  out ("busy to go low"). That image alone was re-run in the next booking and read its vector exactly.
- Each image is its own load, so the graded set (three images from the first pass plus the re-run) is sound.
- N = 1 per bitstream.
- The fixed tree's reading waits for the bitstream that carries 1fc2cd5aa.

## The kit

- `images/` holds four bare images, t1-norevoke, t1-untagged, t1-oob and t1-fire. Each returns on every bitstream,
  and `SHA256SUMS` pins them.
- `build.sh` rebuilds them from `tests/` with `rtl-smoke/objb-bare-2026-10-30`'s wrapper.
  - The test sources are the external collaborator's: capstone-ariane perf/c5a-cached-ldc fe5aa8440,
    `verif/tests/custom/capstone/lv-t1-*`, unchanged.
- The runner is `rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py`, with `SUP_IMG` = the image and
  `SUP_REC_ADDR` = 0x80003040.
- Recommended order: the three controls first, t1-fire last.
- `grade/r67-grade.py <run dir> <bs3|objb|fixed>` grades a run. Its verdicts are:
  - HOLE / FIXED / FINDING for t1-fire;
  - VOID if any control is off;
  - a mismatch if an image hash or the resident bitstream name does not match.
  - Every branch was exercised on fabricated directories before this was pushed.

## The fix

The fix is the external collaborator's: capstone-ariane 1fc2cd5aa, on top of their C.5a e71b236f2, in Verilator only.
- A new PC-capability id is adopted UNKNOWN, and the head is held until the LSU answers from positive evidence: the
  ALLOW or DEAD record, or a probe of the record. A dead PCC raises 25 at the first instruction under it.
- The fix was claim-audited by the RTL lane. No fail-open path was found.
- Behaviour changes recorded by that audit:
  - every PCC id change costs a probe, because only the one-entry ALLOW record is read;
  - a hold blocks interrupts and the quantum until it resolves (bounded by the probe time-out, after which a live PCC
    takes 25);
  - under `R43_PROBE_TIEOFF` every new PCC id is refused.
- The lead ruled (2026-11-03) that it is synthesised in one bitstream with C.5a.
