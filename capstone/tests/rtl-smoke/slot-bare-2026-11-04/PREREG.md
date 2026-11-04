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

## Stage-A arm (added 2026-11-04, before stage A's first boot)

Bitstream: capstone-ariane c9c7952bd, the DYN handshake trim (record: capstone/docs/plans/2026-11-04-dyn-handshake-trim.md).
Vectors come from simulating these same images (ELF sha256 equal to the other trees' runs) on c9c7952bd, clean, and are in
vectors.txt's fourth column. The exact fields are those of every other arm.

- **P1d replaces P1 on this arm.** slot-1 and slot-4 (same doubleword, page alias) read within 2 % of each other; slot-2
  and slot-3 (other doubleword, other line) read within 2 % of each other; and slot-2 reads more than 5 % below slot-1.
  - Sim: 0x170a against 0x1310, 17 % apart.
  - The reason: with the handshake trimmed, the ldc dispatches while the stc is still buffered. To another doubleword it
    no longer waits. To the same doubleword, or its page alias (the offset compare is bits [11:3]), it waits on the
    store's page offset.
- **P2 is unchanged.**
- **Predicted against C.5a's run:** ldc alone and the pair to another doubleword are lower; slot-1, slot-4, slot-5,
  slot-6, slot-8 and slot-9 are unchanged.

Grader controls (fake run dirs built from these vectors):
- the stage-A arm on its own vectors: AS EXPECTED;
- resident name 1fc2cd5aa: WRONG BITSTREAM;
- slot-2 and slot-3 set to slot-1's cycles: PATTERN FAILS;
- C.5a's flat pair on the stage-A arm: PATTERN FAILS.

The real bitstream 3 run below still grades AS EXPECTED with the edited grader.

## Silicon readings

**bitstream 3, board 1, 2026-11-04 16:46-16:49** (the SPEC lane's run, perf/fa/slot-bs3-b1).
- `slot-grade.py <run> bs3`: **AS EXPECTED**. All 9 image hashes match, the resident name is caplifive_vp2c_e7af48457.bit,
  and every exact field is equal. P1 holds (slot-1..4 all 0x2812 = 10,258); P2 holds (4,624 > 2,567).
- Cycles sit 0.4-9 % under the simulation's (slot-1 10,258 against 10,493; slot-9 1,568 against 1,593).
- Per op over the bare loop (cycles per iteration): ldc 6.01, stc 8.01, stc+ldc 16.97 (same and other slot), sd+ld
  5.97 / 1.95; the loop is 3.06. The simulation of the same images gives ldc 6.06, stc 8.42, pair 17.38.

**What it means.** On silicon the isolated pair costs what the simulation says, within a few percent: 17 cycles per
iteration on bitstream 3. The SPEC lane's estimate of 23-36 silicon cycles per pair removed from a compiled program is
larger than that. The excess is not the pair's own cost in isolation. This kit isolates the pair from everything around
it, so it cannot say what the excess is: the surrounding code, cache or TLB state, or what else the compiler change
removed.

## Cycle band (added 2026-11-04, after bitstream 3's reading and before the C.5a and stage-A runs)

- **The calibration.** Bitstream 3's silicon read every slot 0.4-9 % under its simulated cycles. From now on the grader
  prints each slot IN BAND / OUT OF BAND against its own arm's sim cycles ±10 %, plus a CYCLE BAND summary line.
- **The band is reported, not graded.** It never changes the verdict. It exists because without it, a bitstream whose
  silicon does not track its sim passes silently: C.5a's run reading bitstream 3's cycles grades AS EXPECTED, with
  exact fields equal and P1/P2 holding.
- **Control.** Bitstream 3's real run, relabelled as C.5a: the verdict reads AS EXPECTED, and the band reads OUT OF BAND
  on slot-1..4, slot-7 and slot-8. The real bitstream 3 run reads ALL IN BAND.
- **Predicted:** C.5a and stage A read ALL IN BAND. That means the C.5a pair at about 8.4 cycles per iteration over the
  loop (bitstream 3: 17.0), and stage A's other-doubleword pair at about 6.4.
- **The reading that would matter most.** An OUT OF BAND result on C.5a's pair slots, near bitstream 3's 17, would mean
  the cached-LDC change does not reach silicon in this kit.

## Silicon readings, continued (2026-11-04, graded by the RTL lane with the grader at 98ff08e6f3fb)

| run | arm | verdict | band | ldc | stc | pair (same / other) | sd+ld | loop |
|---|---|---|---|---|---|---|---|---|
| board 1, perf/fa/slot-bs3-b1 | bs3 | AS EXPECTED | ALL IN BAND | 6.01 | 8.01 | 16.97 / 16.97 | 5.97 / 1.95 | 3.06 |
| board 2, perf/c5a-b2/slot-bs3 | bs3 | AS EXPECTED | ALL IN BAND | 6.01 | 8.01 | 16.97 / 16.97 | 5.97 / 1.95 | 3.06 |
| board 2, perf/c5a-b2/slot-c5a | c5a | AS EXPECTED | ALL IN BAND | 2.03 | 3.02 | 8.01 / 8.01 | 5.97 / 2.00 | 3.05 |

Per op is cycles per iteration over the bare loop.

- **Bitstream 3 at N=2 across boards.** Board 2 reads board 1's cycles to within 3 cycles in every slot (slot-3 10,261 against
  10,258).
- **C.5a halves the isolated pair on silicon.** It reads 8.01 against bitstream 3's 16.97; its simulation gives 8.42.
- **Every slot of every run sits 1-9 % under its simulated cycles**, so simulation predicts this kit's silicon cost on
  both bitstreams.
- **On forwarding.** It could remove at most about 2 cycles per same-slot pair, on top of stage A, and about 0 on C.5a alone.
  So it is not pursued on this evidence.
- The board 2 slot-bs3 attempt marked `.noUART-switches` is VOID: odd switches left by a killed driver took the console.
  It was not graded.
