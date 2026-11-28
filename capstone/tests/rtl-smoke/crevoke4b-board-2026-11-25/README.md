# crevoke4b-bare-2026-11-25 on board 2: the board lane's run record, and the syntheses behind its seal

This folder records the rtl lane's kit for B5b-2 4(b) and its seal. The kit is
`capstone/tests/rtl-smoke/crevoke4b-bare-2026-11-25/` on land/rtl-crevoke4b, pre-registered at 77f0e191b97f. It holds
17 bare images, the sublet-life control first, each expected to return.

The runtime lane granted one sequence on board 2:
1. an instrument control on 904cd9c6b, no flash;
2. the flash of 88db40027's seal;
3. the 17 images on the seal.

Board 2 stays on 88db40027 afterwards. Board 3 stays on 904cd9c6b.

## The syntheses (capstone-ariane b5b2-crevoke4b; the rtl lane's keys)

Each synthesis ran on apollo under vivado-guard (40 GB), gated on mem-headroom, and was graded against its pre-registered
keys.

| hash | what | gate keys | full build |
|---|---|---|---|
| bf8fc0cfb | 4(b) parts 1 and 2 | PASS 3/3: TM 4,976.9 MB (+22.1), loops 0 with the toys firing, core LUT 124,133 (+1,222 / limit 125,411) | not run (rtl's amendment) |
| 88db40027 | 4(b) + the store drain + the residual-2 guard | PASS 3/3: TM 4,968.7 MB (+13.9), loops 0, core LUT 124,041 (+1,130 / 125,411) | **PASS 10/10**, sealed (below) |
| 47d749153 | 88db40027 + observation-only wedge apertures | PASS 3/3: TM 4,968.4 MB (-0.3), loops 0, core LUT 124,167 (+126 / 126,541) | disarmed (a combined hash follows) |

Reported bands, not keys:
- **bf8fc0cfb and 88db40027.** The lifetime unit's LUT reads -4,528 / -2,891 against a predicted +600..+1,500, with
  dyn_unit +5,191 / +2,478 beside it. The four-module cone sums to about 0, so the logic was re-homed, not removed.
- **88db40027's core FF, +1,303** against +450..+750. A read-only FD* census of both gate dcps puts ex_stage's own +270
  in retiming copies: `_fret`/`_bret` copies of dyn_child_repr_q (+210) and dyn_child_q (+64); st_drained_q is +1.
  The scoreboard's +516, on unchanged RTL, is the same flow behaviour (the rtl lane's reading).
- **47d749153's apertures** came out at their predicted size, module by module: load_unit +8 FF, each wt_dcache_ctrl
  +3, the wbuffer +11, wt_dcache's own row +3. There was no FSM re-encoding. The post-synthesis timing estimate is
  unavailable: a synth dcp carries no clocks.

**88db40027's full build** (graded against 904cd9c6b's seal):
- placed LUT 149,188 and FF 69,534, both inside their bands;
- routed clean, 199,232/199,232;
- congestion: placed level 4, no router hotspot;
- loops: check_timing 0, TIMING-23 0 and LUTLP-1 0, each with its toy firing;
- clk_out1 WNS +0.510, inside [-1.47, +6.01] (904cd9c6b +2.266);
- design WNS +0.177, TNS 0;
- rev_node on the worst 500 paths: 223 paths, 211 cells (904cd9c6b 258 / 309).

The seal is `…-pagedtbl-r76-r75d-b5b2n-crev4b_88db40027.bit`, 11,443,722 B, sha256
`bdff45d0a96f68ad1ea3f2077c7ca350960047a1f394d21849bd4a003da58d9a`. The rtl lane graded it in its plan doc at
7ed2ec8be5ff before any flash.

## The board run

The tools are in this folder, with their dry runs described:
- `kit-images-c4b.py` checks all 17 identities across PREREG, the json, SHA256SUMS, `.bin` and `.rec`.
- `grade-c4bkit.py` implements the PREREG's grading rules. Before the run it was dry-run on fixtures built from both
  json columns (`gfix-c4b.py`):
  - base 16/16 and cand 16/16;
  - a bad control is VOID;
  - a wrong E value is a MISS;
  - the key-check sentinel is a KEY-CHECK FAILURE;
  - a missing SR line is NO READING;
  - zeroed latencies are a VOID ARM;
  - base vectors graded as cand read NOT FLASHED;
  - the cand fixture's latency band reads 3,692, the sim number.
- The kit runner (run-c4bkit.sh, sha256 prefix bf737c6adb15c663) is b5b2n-bare's, with this kit's lister. The
  flash orchestrator (orch-c4b-23.sh, 0e2941e70cee8794) flashed only after four gates held: step 1 ended, the
  full-build grade read PASS, the verdict read SEALED under the agreed name, and the plan doc carried the sha. Each gate
  was dry-run to its STOP, and the flash branch was dry-run end to end. Both scripts stay local.

**Step 1, the instrument control on 904cd9c6b** (hub booking 1512, 08:59-09:15 UTC; resident read back before and
after; no flash): **16/16 AS PREDICTED** against the PREREG's 904cd9c6b column, with the control exact. Every
interrupt arm reads position 9 = 0, the base's property. `step1-base-904cd9c6b.grade.txt`. The rtl lane graded the
raw captures independently and agrees, including every R value. On its count, 8 of the 17 images can tell the two
bitstreams apart: the crevint and crevsv arms other than kill.

**Step 2, the flash** (booking 1513):
- FLASH 09:15:31-09:17:48 UTC; the flash tool checked the sha256 first and then power-cycled;
- the resident after the power cycle, and the readback after the flash, both name `…-crev4b_88db40027.bit`;
- fallbacks in bitstreams/: 904cd9c6b's seal and R-75 (d)'s 1c5d31474.

**Step 3, the 17 images on the seal** (09:17-09:33 UTC; readback after the loads names the seal): **16/16 AS
PREDICTED** against the 88db40027 column, with the control exact. `step3-cand-88db40027.grade.txt`.
- **No VOID arm.** At-site timer traps: crevint-mt 5, -sh 2, -lat 6; the crevsv arms 6, 6, 6, 4, 4.
- **The crevint-lat band PASSes.** lat [1222, -119, 565, -130, 565, -130, 0, 0] ticks, at 30,062 / 15,031 = 2.0
  cycles per tick. Every positive latency after the first is 1,130 cycles (bound 15,000; sim 3,692 at memory delay 12).
- **The sbdrain cost rounds** are recorded: cost0 0xe7a and cost 0xee9 on the seal, against 0xe3a and 0xebf on
  904cd9c6b.
