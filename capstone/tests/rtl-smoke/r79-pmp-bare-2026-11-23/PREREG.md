# R-79 on silicon: pre-registered 2026-11-23, before any board run

One image, one session. It is expected to RETURN on every tree, so the ordering rule is trivially met. The vector is
the recorder's words at `board_rec`: the count at board_rec, and the values at board_rec + 8·(1..17).
- Each vector was taken by simulating the image itself on that tree's RTL through the Anvil prologue: `rtl_dirty=no`,
  `S12_MEM_DELAY=12`, `+time_out=3000000`.
- rc=2 is the bare image's CAP_PASS spin running to the cycle limit, after the recorder is complete.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `images/r79-pmp.bin` | `8eb72553965cfc1c98fd33b4caa44a0b0f1051e661092b63ed41d1cf4d954ec5` | 0x80007080 | 17 |

Built by the llvm-capstone clang `fa5b23aeb9b8` (binary sha256 660fca7da9067f76…) with `build.sh`.

**Symbols:** r_page 0x80002000 (PMP entry 0: NAPOT 4 KiB, R only), w_page 0x80003000, res 0x80007000.

## Which trees

capstone-ariane `core/pmp/src/pmp_data_if.sv` is the same blob (4a8570748e9e) on every Capstone tree resident or
queued on the boards: d83d3a633 (P1), 9e801ab5e (bitstream 1), 1c5d31474 (R-75 (d)), 2ca17ad2a, 0039e34f4 and
6438348ec (the board lane's check, 2026-11-23).
- load_store_unit, load_unit and store_unit are identical to P1's on all of them except 1c5d31474. There, the +33
  lines of R-75 (d) sit behind `R75_EARLYCLR` and touch only the PC-capability tracker (pcc_probe_want, pcc_v_q), not
  the data-side access type or address.
- The pre-fix vector was simulated on BOTH 9e801ab5e and 1c5d31474, and the two are identical word for word.

## Vectors (SV, in order)

| tree | vector |
|---|---|
| **pre-fix: 9e801ab5e and 1c5d31474 (identical)** | `0x0 0x7 0x80007008 0x77 0x0 0x0 0x1234 0x7 0x80003008 0x5a5a5a5a5a5a5a5a 0x0 0x0 0x77 0x77 0x77 0x77 0x2` |
| fixed: capstone-ariane b5b1-step3 d0afdb40f | `0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7 0x80002008 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x77 0x77 0x77 0x77 0x3` |

Positions (0-based) and what each pair says:
- **0, p1:** the control store to w_page: 0 on both.
- **1-3, p2:** a LEGAL ld of r_page with an independent store right behind it. Pre-fix: refused 7, with tval
  0x80007008, which is res + 8 (the following store's own address), and the load's register untouched (0x77). Fixed:
  no fault, and the fill is read.
- **4-6, p3 and p4:** a lone sd 0x1234 to r_page. Pre-fix: no fault (0, 0), and r_page then reads 0x1234: the write
  landed. Fixed: 7 with tval r_page, and the fill is kept.
- **7-9, p5:** an sd to r_page + 8 with an independent sd to w_page + 8 right behind it. Both trees refuse 7, because
  the store behind makes the pre-fix tree judge it as a store too. But the pre-fix tval is 0x80003008, the FOLLOWING
  store's address, where the fixed tree's is 0x80002008, its own. The fill is kept on both.
- **10-11, p6:** lr.d of r_page with nothing behind it. Pre-fix: judged a read, allowed (0, 0). Fixed: 7 with tval
  r_page.
  - That is the semantics owner's ruling of 2026-11-23: LR takes the store/AMO judgement, as CVA6's MMU always did.
  - It is a recorded divergence from the twin, whose LR needs R only, and this position is its positive control.
- **12-15:** markers 0x77.
- **16:** traps, 2 pre-fix and 3 fixed.

## Grading

- **VOID** unless SR is 17 and positions 0 and 12-15 read 0x0 and 0x77 x4. The board's runner must read back the
  resident bitstream's name and STOP on a mismatch with the boot's plan.
- **R-79 CONFIRMED ON SILICON** when a board holding any of the pre-fix trees above reads the pre-fix vector exactly.
- **Not reproduced on silicon** when such a board reads the fixed vector: a finding about the simulation-silicon
  difference, to the RTL lane before any claim.
- Any other vector is a finding, graded value by value against the positions above. The tvals in particular are
  addresses, so a partial match is informative.
- N=2 across boards is welcome. The mechanism is the store unit's next-cycle request and the simulation is
  cycle-deterministic, so the boards should agree; a disagreement is itself a finding.
- The fixed tree has no bitstream yet. Its vector is for step 3's batch, the "after" half.
