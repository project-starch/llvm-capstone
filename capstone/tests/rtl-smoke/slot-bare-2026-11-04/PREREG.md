# PREREG: slot-bare-2026-11-04 (written and pushed before any board run)

## Vectors

The vectors come from simulating these exact ELFs (built by `build.sh` into `out/`, which is not committed; image sha256 in `images/SHA256SUMS`) on
three capstone-ariane trees, each with PROVENANCE `rtl_dirty=no`:
- c5a-eval-ctl 3afd5c9e2: bitstream 3's tokens;
- objb-eval d463206ac: ObjBounds 603528b8b;
- c5a-eval 08c9fbd22: C.5a 1fc2cd5aa.

Run conditions: testbench memory delay 12, time_out 3,000,000. The values are the recorder's words at board_rec,
extracted with the lane's board-vector tool. Each image's recorder count is 5. Field order: cycles, instret, value,
traps, cause.

| image | bitstream 3 | ObjBounds | C.5a |
|---|---|---|---|
| slot-9 | 0x639 0x403 0x0 0x0 0x0 | 0x639 0x403 0x0 0x0 0x0 | 0x632 0x403 0x0 0x0 0x0 |
| slot-6 | 0xb02 0x803 0x5a 0x0 0x0 | 0xb02 0x803 0x5a 0x0 0x0 | 0xafb 0x803 0x5a 0x0 0x0 |
| slot-5 | 0x12f7 0x803 0x1 0x0 0x0 | 0x12f7 0x803 0x1 0x0 0x0 | 0x12f0 0x803 0x1 0x0 0x0 |
| slot-7 | 0x1259 0x603 0x77 0x0 0x0 | 0xa64 0x603 0x77 0x0 0x0 | 0xa5d 0x603 0x77 0x0 0x0 |
| slot-8 | 0x170f 0x603 0x0 0x0 0x0 | 0xd1b 0x603 0x0 0x0 0x0 | 0xd14 0x603 0x0 0x0 0x0 |
| slot-1 | 0x28fd 0x803 0x77 0x0 0x0 | 0x1712 0x803 0x77 0x0 0x0 | 0x170b 0x803 0x77 0x0 0x0 |
| slot-2 | 0x28fd 0x803 0x77 0x0 0x0 | 0x1712 0x803 0x77 0x0 0x0 | 0x170b 0x803 0x77 0x0 0x0 |
| slot-3 | 0x28fd 0x803 0x77 0x0 0x0 | 0x1712 0x803 0x77 0x0 0x0 | 0x170b 0x803 0x77 0x0 0x0 |
| slot-4 | 0x28fd 0x803 0x77 0x0 0x0 | 0x1712 0x803 0x77 0x0 0x0 | 0x170b 0x803 0x77 0x0 0x0 |

## Exact fields

Instret, value, traps and cause are equal on all three trees, and must read exactly as above on silicon:
- instret: 0x403 (slot-9), 0x603 (one op), 0x803 (two ops);
- value: 0x77 through the loop's last ldc result; 1 for slot-5; 0x5a for slot-6; 0 for slot-8 and slot-9;
- traps 0, cause 0.

## Cycles

Cycles are a timing field. They are reported beside the simulation, not required to match it, because the board's DDR
is not the testbench's delay.

## Silicon pattern (graded)

- **P1.** slot-1..4 read the same cycles within 2 %, on every one of the three bitstreams.
  - The reason (source and simulation): the DYN unit takes one op at a time and the ldc dispatches only after the stc
    has left the store buffer, so the ldc never meets the buffered stc. It costs the same whichever slot it reads.
  - A P1 failure means the board's pair cost depends on the address. That would point at the store buffer or the cache
    on silicon, which the simulator does not show.
- **P2.** slot-5 reads more cycles than slot-6: the plain pair waits on the page offset.

## Reported, not graded

The per-op costs over slot-9: ldc, stc, the pair (same slot and other slot), and the plain pairs. In simulation on C.5a
they are ldc 2.08, stc 3.44, pair 8.42 / 8.42, sd+ld 6.37 / 2.39 cycles per iteration over the loop. These are the
numbers the SPEC lane's per-pair estimate needs.

## Verdicts

`grade/slot-grade.py`:
- **AS EXPECTED** (rc 0): exact fields and P1, P2 hold.
- **FINDING** (rc 1): an exact field differs.
- **PATTERN FAILS** (rc 1).
- **WRONG BITSTREAM** (rc 1): the resident name read back must name the arm.
- **NO DATA** (rc 2): an image has no uart.txt / run.log.

The controls were run on fake run dirs built from these vectors:
- the three arms pass, each against its own vectors;
- bitstream 3's run graded as c5a reads WRONG BITSTREAM;
- one value changed reads FINDING;
- slot-1 doubled reads PATTERN FAILS;
- slot-5 halved reads PATTERN FAILS;
- a missing image reads NO DATA.
