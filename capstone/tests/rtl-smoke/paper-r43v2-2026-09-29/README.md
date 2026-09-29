# Paper board runs on R-43 v2 (`caplifive_r43_8f6a0af.bit`), from 2026-09-29

These are the board runs the Sublet paper's hardware protocols still need. They were first planned for R-42
(`caplifive_r42_6cbdaeeb4.bit`). The lead moved them to R-43 v2 after its acceptance: ten boots a1-a10 on
2026-09-29, all passing. On v2 R-43 and R-45 are fixed, and the refusal record (switch values 204-208) says why an
access was refused. Every number here is v2 evidence. It is compared with another bitstream only where a
bridge exists: on the acceptance boots, P1 cell 5 and the K=0 ladder read the same as R-42 (+0.06 % and 0.997).

Each study is pre-registered in this file, pushed before its first boot. Result lines are added beside it.

**Images.** Part A images are built with a compiler that contains #119 (the live-source copy, C-32): the
compiler lane's verified 490372bf build, whose `llvm/` and `clang/` are identical to dev fd98642b. The
identity probe passes: `capstone-live-source-copy` is registered and `+movc-keeps-integer-source` is accepted.
The two exceptions are frozen on purpose, for continuity across bitstreams:
- the R-35 stale probe `35fb3fec3196841b`;
- the live-alias control `aed492ab985653f3`.

**Common to every boot.**
- `FPGA_BITSTREAM=caplifive_r43_8f6a0af.bit`, as the board reported it after the flash, and `REFUSAL_RECORD=1`.
- The stock k800 control `b2d60e525f807ea4` at 0x10000 runs first and last. Its failure voids the boot.
- The host is `2c9e82d101b48160`.
- At most one fault probe per boot, and it runs last (M-1: a fault wedges the core).
- A fault is read from the trap log and the refusal record in the wedge dump, and the boot is read only up
  to its first failure.

## H1: calibration, the nodes regression and the linear family (three boots)

**Why.** H1 step 5 asks for the timer overhead and the dependent-access latency over three fresh boots, with
median and range. E4 (§7v of the measurements doc) measured them in one boot on `caplifive_r30r31_1bfff7776`.

**What runs, identical in all three boots.** `board-r1e4.sh` with:
- `R1_IMG` = image `da9c54e578845735`, the R1 harness at defaults, rebuilt with #119, entry 0x410000;
- `R1_LIST` = `drivers/lists/h1-v2-calibration.txt`: E4's calibration list (the latency chase with three timed
  traversals, `--calib` in three fresh domains, and the Sublet nodes regression), plus the linear series.
  The nodes regression is back: on R-42 it false-denied (B3, R-43), and on v2 the R1 campaign completes
  (acceptance a2).

**Emulator record** (`~/capstone-artifacts/h1-v2/`; the pass record is `qemu-pass/da9c54e5…`). All four
invocation types return `R1_RC=0`:
- latency: 15 `R1 lat` lines, 5 instructions per load;
- calib: `cyc_cyc=1`;
- nodes: n = 1, 4, 16, 64, 256, with nd = 2n, `ok=1 bad=0`;
- linear: 12 lines reading 7,1,7,7,0,1,0,1,7,1,0,1. This is the emulator's reading: Q-12 on `ldc`/`stc`, and
  `tighten` moves.

**Pre-registered readings, per boot.**
- k800 returns 4 twice with instret 1089.
- **Latency:** flat across the three traversals.
  - 4 KiB and 16 KiB read the same value, and 256 KiB and 1 MiB read the same, larger one.
  - E4 read 9.00 and 48.2 on 1bfff7776. **A uniform shift on v2 is a result, not a failure.**
  - New readings would be: 4 KiB ≠ 16 KiB, a non-monotone curve, or traversals disagreeing by more than 0.1 cycles.
- **Timer:** `cyc_cyc` = 2 and `ret_ret` = 1 in all three domains (E4's reading).
- **Nodes regression:**
  - nd = 2n, `ok=1 bad=0` at every n;
  - REVOKE is recorded as a change detector against E4's 126 / 182 / 731 / 2,838 / 11,714 for n = 1…256;
  - no cause 25.
- **Linear:** arms 0-11 read 7,1,7,7,7,1,7,1,0,1,0,1, as on R-42 (boot r42e3b).
  - Arms 8 and 10 read 0: `tighten` and `shrinkto` copy, R-21.
  - The controls read 1.

## M1 condition 3 on v2: the refusal record names why a stale reference is refused (six boots)

**Why.** Condition 3 was measured on silicon on `4ad0df694` and R-42, where cause 25 could not separate
"recognised as revoked" from "refused on a cache miss". v2 has no deny-on-miss, and its refusal record gives
the arm. Acceptance a10 is the first reading: the stale probe `35fb3fec` trapped 25 at +0x4354 with the
record LATCHED, arm **probe DEAD**, id 0x5f. These boots bring the stale read to three repetitions and
measure the stale WRITE, which no earlier bitstream reached.

**Images** (all at 0x410000, list `drivers/lists/m1-staletake.txt`, the one a10 used):

| boot | image | built with | role |
|---|---|---|---|
| m1-1 | `9b24aa311881ecda` | #119; source `lane/board-m1-write-control` b977bc52 | **write CONTROL**: stores 0xA5 through the LIVE alias |
| m1-2 | `aed492ab985653f3` | frozen | the live-alias control for the read, first reading on v2 |
| m1-3, m1-4 | `35fb3fec3196841b` | frozen | the stale read, repetitions 2 and 3 on v2 (a10 is 1) |
| m1-5, m1-6 | `067cc96f89152287` | #119 | the stale WRITE: 0xA5 through `m1_ret_alias[0]`, reads off (DEREF=0), no mint |

The two write images differ by exactly `-DM1_STALE_TAKE_LIVE=1`. This was checked two-sided: with the define
off, the controlling commit's source builds `067cc96f`, byte-identical to dev's. On the emulator:
- the stale write halts at image+0x4370, the `sb` of 0xA5 through `ldc 0(ldc 0x190(gp))` in `run_m1`;
- the control prints `stale-write CONTROL`, `stale-write ok` and `readback via_live_alias=165 wrote=165`, then
  returns `R1_RC=0`.

Those emulator runs used `--cap 64 --budget 60000` to keep the emulator fast. The board list uses
`--cap 65532 --budget 900000`, the R-35 probe's geometry, where 43,296 aliases are retained before the probe. The
stale probe `35fb3fec` has had the same split since 2026-09-19, and a10 reached its probe under the board list.

**Pre-registered readings.**
- **m1-1, the write control:** it returns. The three lines above appear, with no trap, and the record reads
  EMPTY in the first running read. A trap here voids the write pair.
- **m1-2, the live-alias read control:** as on 4ad0df694, the dereference commits and the later mint traps cause
  **26**. The record is EMPTY, because 26 is not a refusal. A cause 25 at +0x4354 here would be a false refusal on
  v2, and it stops the M1 series.
- **m1-3, m1-4, the stale read:** as a10. Cause 25 at +0x4354, tval 0xac100000, record LATCHED with a one-hot arm
  (probe DEAD is a10's arm) and an id with consistent parity.
- **m1-5, m1-6, the stale write:** cause 25 at +0x4370, the store refused, with the record LATCHED and a one-hot arm.
  **Integrity violated** would read as `stale-write ok` followed by `readback via_live_alias=165`: the domain
  returns, having written another object's storage through a revoked reference.
- **Not established by these boots:** anything about generation wrap, or about stale references held in memory
  rather than in a register. N is at most 3 per arm.

## S1/S2 safety pilot on v2 (three boots)

**Why.** The paper's `tab:safety` rows that say "stops at access" rest on emulator evidence. On
`caplifive_r30r31_1bfff7776` (E1, §7r) the three touch cells RETURNED:
- s3 read the fill 0x00;
- s5 read the new occupant's byte 0x5B;
- s11 read type 1 and then the byte.

That is unsafe-success. v2 carries the R-35 fix, R-45 and the R-43 redesign. This pilot asks whether those
cells now stop at the access.

**What runs.** `board-b78-w2h.sh` with `REP=1`, `BOOT=1,2,3`, and `E1_DIR` set to the cell directory, whose
`cells.tsv` is `pilot-cells.tsv` in this folder. The nginx UAF cells come from dev with #119 at new entry VAs:
the 2026-09-14 table put s2 at k800's 0x10000 and s10 and s11 at one shared VA. Each boot runs the five
returning cells and then ONE touch cell:

| cell | image | VA | emulator mark | pre-registered on v2 |
|---|---|---|---|---|
| s1 | `72acfe191d83bed5` | 0x90000 | C10000 | C10000 |
| s2 | `2662e257347e05b9` | 0x110000 | C20000 | C20000 |
| s4 | `bb85ad274f1c5a7b` | 0x190000 | C40001 | C40001 |
| s6 | `3eba6adbda05d4df` | 0x210000 | C600FB | C600FB |
| s10 | `c22e7003cf3a0643` | 0x290000 | CA0780 | **CA0180**: the reloaded stale pointer is TAGGED, type 1 (the RTL's `ldc` forwards it; 1bfff7776 read this) |
| s3 (boot 1, last) | `9818ea50f13b2b02` | 0x310000 | halts, cause 24 at +0x6734 | **cause 25 at +0x6734**, the `lbu a0,0(a0)` touch in `domain_main`; record LATCHED |
| s5 (boot 2, last) | `8703d482d9260461` | 0x310000 | halts, cause 24 at +0x6924 | **cause 25 at +0x6924**, the touch; record LATCHED |
| s11 (boot 3, last) | `1f9d93aaefbd59a6` | 0x310000 | halts, cause 24 at +0x6794 | **cause 25 at +0x6794**, the touch after the type read; record LATCHED |

The emulator's cause 24 is expected: its `ldc` untags a revoked capability (Q-11), whereas silicon keeps the
tag and refuses the access. The fault sites were mapped with `fault-locate.py`, and every one is the stale
`lbu` in `domain_main`. R-43 screen: every cell runs on the emulator with the node pool capped at 256; the
cap's positive control (s1 at 16) aborted.

**Refuted if** a touch cell returns a mark, whether `C300xx`, `C5005B` or `CB01xx`. That is unsafe-success on
v2, which would contradict a10 and be a defect in its own right. The pilot is **not** the full E1 matrix
(12-15 boots), which runs only if asked. N = 1 per touch cell.

## Results (2026-09-29, all 12 boots)

Every boot read as pre-registered. The result lines, one file per boot, are in `results/`; the summary and the
readings are in the measurements doc, section "Part A on R-43 v2".
- **H1:** latency 9.000 / 40.83 / 48.16-48.17 cycles per load, identical to E4. Timer 2 / 1. Nodes regression ok at
  every n. The linear family reads as on R-42.
- **M1:**
  - the stale read is refused 3/3, counting a10;
  - the stale WRITE is refused 2/2, a first on any bitstream;
  - each stale boot's refusal record (first refusal since reset) reads the probe-path arm;
  - the write control, matched to the stale write by one define, completes with readback 165;
  - the live-alias read control (an older fixture, not matched to 35fb3fec) reads EMPTY.
- **S1/S2 pilot:** s3, s5 and s11 stop at the access with cause 25 at their pre-registered `lbu` (+0x6734,
  +0x6924, +0x6794). The returning cells read their marks, and s10 reads CA0180.

**Instrument.** The wedge dump's record reads were fixed mid-campaign (dev `ae1b13c2`), and from m1v2-4 on every
byte is fresh. `rr-aperture-check.py` here reports the switch value at which each record byte was actually taken:
exit 1 if any was read at the wrong aperture. It is validated two-sided: it flags m1v2-2, and it passes a10. The
pilot's first three launches refused before any boot, on cells left staged by an earlier refusal. The E1
driver arms its cleanup only after its source checks.
