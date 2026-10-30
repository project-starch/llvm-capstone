# ObjBounds on silicon: pre-registered 2026-10-30, before the flash and before any board run

**The two bitstreams.**
- **ObjBounds** is capstone-ariane objbounds-b 603528b8b. It goes on board 2: the runtime lane flashes it once the
  synthesis lane has sealed it and the RTL lane has graded the seal.
- **Bitstream 3** is e7af48457, resident on board 1. It is the reference.

The record behind both, with the synthesis pre-registration, is `docs/plans/2026-10-30-objbounds-variant.md`
(llvm-capstone land/rtl-objb c8be354d28ef and later).

**What the variant is.** The define, plus a two-line change in the DYN unit's Anvil source (Anvil has no
preprocessor). `git diff 1782d20bd 603528b8b -- core`, verbatim:

```diff
--- a/core/anvil_build/capstone_dyn_unit.anvil
+++ b/core/anvil_build/capstone_dyn_unit.anvil
@@ -682,7 +682,9 @@ func LDC(msg){
         let rs1_v = *_debug_inp.cap_rs1;
-        let validity = call get_node_query_validity(rs1_v.metadata.revnode_id) >>
+        // ObjBounds (objbounds-b, site e): an LDC/STC takes no node query. Bitstream 3 (e7af48457) calls
+        // get_node_query_validity(rs1_v.metadata.revnode_id) here, which reads the record and waits for it.
+        let validity = 1'd1 >>
@@ -750,7 +752,9 @@ func STC(data){
         let rs2_v = *_debug_inp.cap_rs2>>
-        let validity = call get_node_query_validity(rs1_v.metadata.revnode_id) >>
+        // ObjBounds (objbounds-b, site e): an LDC/STC takes no node query. Bitstream 3 (e7af48457) calls
+        // get_node_query_validity(rs1_v.metadata.revnode_id) here, which reads the record and waits for it.
+        let validity = 1'd1 >>
--- a/core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv
+++ b/core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv
 `define CAPSTONE_ITFC_EN
+// ObjBounds (2026-10-30): the performance ladder's per-object-bounds rung. This macro removes the lifetime verdict on a
+// capability load, store or fetch (load_store_unit.sv, commit_stage.sv). objbounds-b also drops LDC's and STC's node
+// query in capstone_dyn_unit.anvil, which is not under the macro because Anvil has no preprocessor. Every other
+// behaviour is e7af48457's.
+`define CAPSTONE_NO_LIFETIME_CHECK
```

The macro's two sites are `ifndef` wrappers, already in the control. They sit around the LSU's lifetime verdict
(load_store_unit.sv) and around the PC capability's cause-25 check (commit_stage.sv).

**Why the control 1782d20bd stands for bitstream 3.** Two facts:
- **Its source preprocesses to bitstream 3's.** core/ against e7af48457 differs only in those two wrappers. With the
  macro undefined, `verilator -E -P` gives load_store_unit.sv and commit_stage.sv token streams identical to
  e7af48457's: 3,044 and 2,412 tokens.
- **Its generated units hash to bitstream 3's.** The sha256 prefixes of all five generated Anvil units, ex_stage.sv
  and ariane_pkg.sv equal those of e7af48457's own clean runs. The DYN unit is 7accda6a28c7bf08 on both; the variant's
  is 131818fb5bfda03c.

**Which accesses lose the check.** The consumers were enumerated by the call sites of the implementing function, not
by the cause number:
- **Removed:**
  - (a) the LSU verdict on plain loads, stores and AMOs;
  - (b) the PC tracker's 25 at commit;
  - (e) the DYN unit's node query on every LDC and STC.
- **Unchanged:**
  - (c) the CPMP trackers;
  - (d) the DYN unit's queries on REVOKE, SPLIT, CDERIVE, CREVOKE (both operands), LCC, TIGHTEN, CALL, RETURN and
    CSSUPERVISE.

  So ObjBounds still pays one query per domain entry, through CALL, RETURN and CSSUPERVISE. The rev-node unit's own
  query site lists the ops it fronts (`capstone_rev_node.anvil:428`).

## The images and the order

One boot per board, in this order: control first, identity second, the cost shapes, and `objb-pcc` LAST. Every image
returns on both bitstreams except `objb-pcc` on bitstream 3: a revoked PC capability faults inside its own trap
handler there.

| image | sha256 | board_rec | SR on bitstream 3 | SR on ObjBounds |
|---|---|---|---|---|
| `images/objb-pcc-ctrl.bin` | `fc23a616026e6d10830bc8cb38be52763ef92456bb93859203cee11bceffba96` | 0x80003100 | 3 | 3 |
| `images/objb-stale.bin` | `08b1d9aabb41f6893def5940a23a89baab0a7cf4ee295b259e496c67d1a323df` | 0x80003540 | 9 | 9 |
| `images/objb-perf-s1.bin` | `8073369ae0aa70b7926c423405bc4555775eb3adb3a2e2574ba0543f6269c4f3` | 0x80018000 | 5 | 5 |
| `images/objb-perf-s2.bin` | `2f22d39e3069f915b21d29e2319e565248dc955b3ff1a96c2b3f4799d84bb36a` | 0x80018000 | 5 | 5 |
| `images/objb-perf-s3.bin` | `dc331d90d518d0e65f84a45bff975a4823c4e8967092eaaa4f94f36e5cb4cd79` | 0x80018000 | 5 | 5 |
| `images/objb-perf-s4.bin` | `71a2464eb09747370af2cb77466003bc8b9759dac74ec332070b42889d244288` | 0x80018000 | 7 | 7 |
| `images/objb-perf-s5.bin` | `9ac3442093cc21363d2e71cc1a401466be613b137c9e4ff8b84bcf8f76cd0506` | 0x80018000 | 6 | 6 |
| `images/objb-perf-s6.bin` | `ebc207552864ae09d5112540096e31eb26a107dd19d26928f77fb0c376dd0a3a` | 0x80018000 | 5 | 5 |
| `images/objb-perf-s7.bin` | `4acb86e5e22cec12babe0721d7fa27af225fcdae2977fd644bb7cfa5b0b716f5` | 0x80018000 | 5 | 5 |
| `images/objb-pcc.bin` | `0863f5636359a978a94b45fb7c3debf51955b358aa413060d24f8580d1251ac6` | 0x80003100 | none (no return) | 3 |

**How the vectors were made.** Each vector is the recorder's words at `board_rec`, read by board-vector.py's method,
from simulating THESE images. For each image, its `.bin` equals `llvm-objcopy -O binary` of the simulated ELF, and
its `.rec` equals the ELF's `board_rec`.
- The ObjBounds vectors come from 24e588199 (603528b8b's tokens), with time_out 3,000,000. Every image parks by
  2.4 M cycles on the control.
- The bitstream-3 vectors come from the control 1782d20bd, with time_out 6,000,000.
- Both trees ran through the Anvil prologue, with `rtl_dirty=no` and `S12_MEM_DELAY=12`. Run dirs: vp8
  `.sup/kit-b-on`, vp9 `.sup/kit-b-off`.
- `build.sh`, run on a fresh copy with `VP_SRC` set to 603528b8b's three sources, reproduces `images/SHA256SUMS` and
  every `.rec` exactly.

## Vectors

**The identity images.**

| image | bitstream 3 (control tree) | ObjBounds |
|---|---|---|
| objb-pcc-ctrl | `0xbb5 0x0 0x0` | `0xbb5 0x0 0x0` |
| objb-stale | `0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6` | `0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0` |
| objb-pcc | does not return; the recorder count is never written | `0xbb5 0x0 0x0` |

- **objb-stale on bitstream 3** takes 25 (0x19) at the six stale accesses (positions 0-3, 5, 6). The stale store
  does not land, so position 7 reads 0x77, and the trap count is 6.
  - This source read the same vector on silicon on `caplifive_sublet_d93a8d66b`, 2026-10-16
    (`sublet-bare-2026-10-15`). On e7af48457 the vector is this simulation's; the board reads it for the first time.
- **objb-stale on ObjBounds** takes no trap. The stale store lands (0x3333) and the trap count is 0.

**The cost shapes.** Position by position. The cycle positions are marked, and every other position must be equal
on both bitstreams.

| image | cycle positions | bitstream 3 (control tree) | ObjBounds | sim cycles, bitstream 3 -> ObjBounds |
|---|---|---|---|---|
| s1: 4,096 ld, one capability | 0 | `0x414f 0x1403 0x5a 0x0 0x0` | `0x414f 0x1403 0x5a 0x0 0x0` | 16,719 -> 16,719 |
| s2: 64-node ldc chase | 0 | `0x2423 0xc03 0x80004000 0x0 0x0` | `0x145f 0xc03 0x80004000 0x0 0x0` | 9,251 -> 5,215 |
| s3: 1,024-node ldc chase | 0 | `0x2b709 0x1804 0x80004000 0x0 0x0` | `0x22744 0x1804 0x80004000 0x0 0x0` | 177,929 -> 141,124 |
| s4: 64 nodes after a flush, pass 1 / pass 2 | 0, 2 | `0x290 0xc3 0x2c4 0xc3 0x80004000 0x0 0x0` | `0x1d0 0xc3 0x1d0 0xc3 0x80004000 0x0 0x0` | 656 / 708 -> 464 / 464 |
| s5: churn / two passes | 0, 1 | `0xa0d8f 0x4da4d 0x400a 0xffc00 0x0 0x0` | `0xa0c5f 0x3b049 0x400a 0xffc00 0x0 0x0` | 658,831 / 318,029 -> 658,527 / 241,737 |
| s6: 1,024 ldc, one node | 0 | `0x2430 0x503 0x0 0x0 0x0` | `0x1422 0x503 0x0 0x0 0x0` | 9,264 -> 5,154 |
| s7: 1,024 stc, one node | 0 | `0x2c72 0x503 0x80004000 0x0 0x0` | `0x18c4 0x503 0x80004000 0x0 0x0` | 11,378 -> 6,340 |

The non-cycle positions are:
- the window's minstret delta;
- the value: s1 0x5a; s2-s4 the final cursor; s5 the sum 2 × (0 + … + 1023) = 0xffc00; s6/s7 the cursor stored at
  +16;
- the trap count, 0;
- the last cause, 0.

They are equal across the two trees in simulation, at every position.

**The predicted per-op savings: a PATTERN under memory delay 12, not silicon values.**
- These are the no-query tree's in-tree arms (objbounds-noquery 233e0e81d, equal to the variant on every cycle count)
  against the control, on one resident node: 3.96 cycles per LDC (s6) and 4.67 per STC (s7).
- The kit's own images read 4.01 and 4.92: (9,264 − 5,154) / 1,024 and (11,378 − 6,340) / 1,024.
- Silicon's memory latency is not delay 12, so none of these numbers is graded.

## Grading

- **VOID** unless both of these hold:
  - objb-pcc-ctrl reads SR 3, `0xbb5 0x0 0x0`;
  - every image that returns reads its table SR.
- **Identity on board 2: CONFIRMED** when both of these hold:
  - objb-stale reads the ObjBounds vector exactly;
  - objb-pcc returns `0xbb5 0x0 0x0`.
- **If board 2 reads bitstream 3's objb-stale vector, it is a STOP.** The variant is not on the board, or the macro did
  not reach it. Nothing else is graded, and the reading goes to the RTL lane.
- **Bitstream 3, board 1:**
  - objb-stale reads its vector above. This is its first silicon reading on e7af48457.
  - objb-pcc does not return. That is the expected reading and needs no recovery: it is the last image.
- **Each cost shape, on each board:**
  - Every non-cycle position must equal the table exactly. A mismatch is a finding (a different path or program),
    not a cost, and that image's cycles are not read.
- **The cost pattern** is graded silicon against silicon: board 2 (ObjBounds) against board 1 (bitstream 3), with the
  same images. The predictions:
  - **Equal within 1 %:** s1 (position 0) and s5's churn (position 0).
  - **Lower on ObjBounds:** s2, s3, s4 (both passes), s5's passes (position 1), s6 and s7.
  - The simulated ratios above are for reference only.
- **One boot per board, graded exactly.** A second boot on the same bitstream must read every non-cycle position the
  same as the first.

## Addendum 1 (2026-10-30 19:30, before any board run): both arms on board 2

The runtime lane changed the plan: board 2 is reflashed to bitstream 3 (e7af48457) at about 20:00. That gives
both arms one board, so the pair differs in one variable: this kit runs on board 2 on bitstream 3 first, and on
ObjBounds after the seal. The cost pattern above is therefore graded **board 2 against board 2**. Board 1's reading
is no longer part of the grade, and is not needed.

**One verdict per arm, for the driver.**
- **Bitstream-3 arm: AS EXPECTED** when all of these hold:
  - objb-pcc-ctrl reads `0xbb5 0x0 0x0` and every returning image reads its SR;
  - objb-stale reads `0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6`;
  - every cost shape's non-cycle positions equal the table;
  - objb-pcc does not return.

  Otherwise the arm reads one of these, and every one of them stops the driver:
  - **VOID** if the control fails;
  - **STOP: wrong bitstream** if objb-stale reads the ObjBounds vector or objb-pcc returns;
  - **FINDING** for any other mismatch.
- **ObjBounds arm: CONFIRMED** as graded above. Otherwise **VOID**, **STOP** (bitstream 3's objb-stale vector), or
  **FINDING**, and the driver stops on each.

**objb-pcc on bitstream 3 wedges the core.** It loops in its own trap handler and never returns. In the
bitstream-3 arm it is the last image of the BOOT, not just of the kit, and any measurement kit after it needs a fresh
boot. On ObjBounds it returns, so a measurement kit may follow it in the same boot.

## Addendum 2 (2026-10-31, before the boot): the witness image `objb-stale-w`

**Why.** The ObjBounds arm read CONFIRMED on board 2 (2026-10-31 00:46-00:55, 10 of 10, graded by script against this
file). Its objb-stale vector rules out "the probe does not reach the check": the hash-identical image took 25 at all six
stale accesses on bitstream 3 on the same board, and the stale store #7 retired and wrote 0x3333 on ObjBounds. It does
NOT witness the premise on silicon: that the crevoke and the revoke took effect on the ObjBounds bitstream. Every
ObjBounds reading fits "check removed" and "revocations ineffective" equally: the seven 0s, the 0x3333, trap count 0,
and objb-pcc's arm returning. The evidence against the second was source (the variant's diff touches no revocation
path), simulation and silicon = simulation on this image; none of it was a silicon reading. The datum also covers one
access of six: #2's 0x1111 is overwritten by #7, #4's 0x2222 is never read back, and #1, #3 and #6 rest on mcause 0 and
the trap count. CONFIRMED stands as graded; this image upgrades the attribution, it does not re-grade the arm.

**The image.** `tests/objb-stale-w.S` is `stale-access.S` with one block after #9; `diff` against it shows the title, the
header paragraph and that block, nothing else. Positions 0-8 are therefore stale-access's. The block runs TIGHTEN rd,
rs1, x0 on each node, then reads path B's datum and the trap count:

| position | op | what it reads |
|---|---|---|
| 9 | tighten of the park s3 (never revoked) | the op's control: 0 on both trees |
| 10 | tighten of path A's child s5 (crevoked) | 25 if the crevoke took effect |
| 11 | tighten of path B's piece s11 (revoked through its handle) | 25 if the revoke took effect |
| 12 | tighten of path C's reloaded s8 | 25 if the ldc carried the dead node |
| 13 | ld of region+128 through a fresh capability | path B's store #4: 0x2222 if it landed, 0 if refused |
| 14 | the trap count | |

Why TIGHTEN. Its handler (`capstone_dyn_unit.anvil` func TIGHTEN, :416 on 603528b8b) checks the type (LINEAR and NONLIN
pass), then the rights (an immediate of x0 = 0 is a subset of any rights), then calls `get_node_query_validity` on rs1's
node (:432) and raises 25 on a dead node before any write. That query is outside the variant's diff: the variant
differs from its control (objbounds-off-b 1782d20bd) only in the config package's define and the LDC (:685) and STC
(:753) query lines. The ELF carries exactly four TIGHTEN words (opcode 0x5b, funct3 1, funct7 2, rd x29, rs2 x0) on
x19, x21, x27 and x24; objb-stale carries none.

`build.sh` gains one line, and rebuilding reproduces the ten frozen images' SHA256SUMS lines unchanged; the file gains
one line.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `images/objb-stale-w.bin` | `ae0e1cb35dcc42d3c7bf5e370c6b1ec68a1ed1b9d2305b0a9e9f94191357e582` | 0x80003540 | 15 |

**Predicted from source, then read in simulation** (the image's own ELF, testharness at memory delay 12, time_out
3,000,000, through `objb-kit-inner3.sh` with its Anvil prologue; the frozen objb-stale ran in the same model as a
control and read its table vector on both trees):

| tree | run | SR | vector, positions 0-14 |
|---|---|---|---|
| bitstream 3's RTL: objbounds-off-b 1782d20bd, the control | vp9 `.sup/kit-w-off`, PROVENANCE rtl 1782d20bd dirty=no | 15 | `0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6 0x0 0x19 0x19 0x19 0x0 0x9` |
| ObjBounds: 24e588199 (603528b8b's core; 13c772ee4 is a comment) | vp8 `.sup/kit-w-on`, PROVENANCE rtl 24e588199 dirty=no | 15 | `0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0 0x0 0x19 0x19 0x19 0x2222 0x3` |

Both rows equal the predictions written from source before the sims read. Positions 0-8 equal objb-stale's table
vectors on each tree. Positions 9-12 are the same on both trees, which is the point: TIGHTEN's query is bitstream 3's
on both. Positions 13 and 14 differ as path B's store and the removed check predict.

**The board run.** Runtime's `spec/board/objb-witness.sh`, on board 2 after the ObjBounds main kits and before the
reflash back to bitstream 3, one image, `run_sup_bare.py` 102ab84e762cba50, SUP_TIMEOUT 90. The reflash power-cycles
the board whatever the image does. No wedge is expected on either tree: the trap handler runs under a live PC
capability, and every refused TIGHTEN resumes at mepc + 4 as the stale accesses do.

**Grading (ObjBounds, board 2):**
- **VOID** if the image's sha256 or the resident name (`caplifive_vp2c_603528b8b.bit`) differs, the image does not
  return, SR is not 15, or position 9 is not 0.
- **STOP: wrong bitstream** if positions 0-8 read bitstream 3's objb-stale vector.
- **WITNESSED** when every position reads the ObjBounds row above: the crevoke and the revoke took effect on the
  ObjBounds silicon, so the ObjBounds arm's seven 0s are the removed check and not a dead revocation; and path B's
  stale store landed (position 13).
- **FINDING** for any other reading. In particular 0 at any of positions 10-12 means that node was live on ObjBounds
  silicon: the arm's identity reading for that path is then attributable to an ineffective revocation, and the
  ObjBounds cost readings are re-examined before they are used.
