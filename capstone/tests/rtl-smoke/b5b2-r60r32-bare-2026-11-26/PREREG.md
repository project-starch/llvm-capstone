# R-60 and R-32 on Line B silicon: pre-registered 2026-11-26, before any board run

**The seal:** the final Full candidate b42c8c644, sha256 6a09f34d…190a. The resident name is read back, and the seal
is cited by its sha256. The kit is an add-on to a b42c8c644 bare-kit slot (board 2 or 3) that is already booked, not
a session of its own.

**Each vector** is the recorder's words at `board_rec`: SR, then the values, as the UART prints them.
- They were taken by simulating THESE images (`out/*.elf`, built by `build.sh`; SHA256SUMS) through the Anvil
  prologue at `S12_MEM_DELAY=12`, without the simulation-only mint scaffold, as on silicon.
- The tree was capstone-ariane b5b2-hangmon-c 449460a34, whose core is b42c8c644's plus one sim-only `$display` and
  SUP_TRACE-only monitors. rtl_dirty=no (s2pre-wt `.sup/r6032`).
- `expected-vectors.json` carries them, with a class per position.

| # | image | sha16 | rec | SR | sim vector |
|---|---|---|---|---|---|
| 1 | sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | the sibling kits' vector (b5b2-misscost's, identical image) |
| 2 | b5b2-r60-alias | f9d7594ee79135e8 | 0x80019000 | 10 | 0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x0 0x0 0x7 0x1 |
| 3 | b5b2-r32-shrinkto | 862931c38f0893dd | 0x80006000 | 16 | 0x0 0x1000 0x0 0x40 0x0 0x40 0x0 0x1 0x0 0x11 0x0 0x7ff 0xfc0 0x1000 0x10 0x30 |

**Positions:**
- **b5b2-r60-alias:** A (the sentinel through the witness), B (an in-window store and load), C and D (the trap count
  before and after the moved store), E (its cause, 0x18 = 24, an untagged base), F (the sentinel afterwards), two
  unused zeros, G (the moved capability's LCC type, 7 = untagged), then the trap count.
- **b5b2-r32-shrinkto:** pairs of (start, end) minus region, for the parent, SHRINK [0, 64), and SHRINKTO 64 / 1 /
  17 / 2047, 64 at cursor 4032, and 32 at cursor 16.
- Every position of both images is E (exact). sublet-life's classes are the siblings'.

**What silicon can and cannot show:**
- The two-sided half of each reading exists only in simulation. The same images on the same core, with the R-60
  predicate forced off or R-32's `inc - 1` restored, read the alias landing (`0x9999` at F, `0x0` at E, `0x1` at G)
  and the pre-fix bounds (`0x3f`, `0x0`, `0x10`, `0x7fe`, `0xfff`, `0x2f`). Those builds cannot run on silicon.
  So this kit shows the fixed behaviour on the board, and the simulation shows that the test would have seen the
  defect (the test headers, capstone-ariane 1f1740a9c and ce6c77b73).
- A board on any seal without these fixes is not a Line B seal. No Line B column without them exists.

**Grading:**
1. **VOID boot:** sublet-life's E positions are not exact, or any RUN line's image sha16 or rec differs from
   SHA256SUMS / `.rec`.
2. **CONFIRMED ON SILICON** (per issue): its image reads every position of its vector exactly.
3. **FINDING:** anything else, reported with its values. In b5b2-r60-alias: `0x9999` at F, or 1 at G, is the alias
   on silicon, and the fix is absent from what is resident. In b5b2-r32-shrinkto: an end one short of its row is the
   pre-fix SHRINKTO.

## Result (2026-11-26, booking 1583): BOTH CONFIRMED ON SILICON

Board 2, b42c8c644 resident (its name read back before and after the loads, no flash), result lines on board/meas-pw 371f7706e0e5 (capstone/tests/rtl-smoke/b5b2-seals-board-2026-11-25/b42c8c644-board2.result-lines.txt). Graded here by a script against this kit's expected-vectors.json (position classes included; negative-tested: one corrupted expected value turns the image to FAIL), and the images' sha256 recomputed from images/. The board lane graded it independently with the same result.

- sublet-life SR 35: every E position exact (U positions 27-34 differ, as they may): the boot is not VOID.
- b5b2-r60-alias SR 10: `0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x0 0x0 0x7 0x1`, exact. No `0x9999` at F, G is `0x7`: R-60 is fixed in what is resident.
- b5b2-r32-shrinkto SR 16: `0x0 0x1000 0x0 0x40 0x0 0x40 0x0 0x1 0x0 0x11 0x0 0x7ff 0xfc0 0x1000 0x10 0x30`, exact.
- Each RUN line's sha16 and rec equal this kit's (ecbfe917.../80003200, f9d7594e.../80019000, 862931c3.../80006000).
- As registered above, the two-sided half is in simulation only: silicon shows the fixed behaviour.
