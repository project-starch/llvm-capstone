# NOGROW on Line B silicon: pre-registered 2026-11-26, before any board run

**The seal:** the final Full candidate b42c8c644, sha256 6a09f34d…190a. The resident name is read back, and the seal
is cited by its sha256. The kit is an add-on to a b42c8c644 bare-kit slot that is already booked: board 2's "KIT
r6032" step behind b5b2-r60r32 if it lands in time, otherwise the next one.

**Each vector** is the recorder's words at `board_rec`: SR, then the values, as the UART prints them.
- They were taken by simulating THESE images (`out/*.elf`, built by `build.sh`; SHA256SUMS) through the Anvil
  prologue at `S12_MEM_DELAY=12`, without the simulation-only mint scaffold.
- The tree was capstone-ariane b5b2-hangmon-c 336f99a85, whose core is b42c8c644's plus one sim-only `$display` and
  SUP_TRACE-only monitors. rtl_dirty=no (s2pre-wt `.sup/ngk`).
- They equal the in-tree readings in b5b2-nogrow.S's header.

| # | image | sha16 | rec | SR | sim vector |
|---|---|---|---|---|---|
| 1 | sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | the sibling kits' vector (identical image) |
| 2 | b5b2-nogrow-ctl | 7c6f34575a03df51 | 0x8000c000 | 4 | 0x1e 0x7 0x100 0x1 |
| 3 | b5b2-nogrow | a5fb074066b5e58e | 0x8000c000 | 4 | 0x0 0x7 0x100 0x0 |

**Positions:**
1. The CDERIVE's cause (0 if no trap).
2. The LCC type of its rd. rd is zeroed first, so 7 means untagged and 1 a tagged child.
3. The header's next after the CDERIVE (0x100 means nothing was allocated).
4. The M trap count.
Every position is E (exact).

**What the pair shows:**
- The control and the arm differ only in the growth word, so they differ only at positions 1 and 4.
- A NOGROW reading equal to the control's (0x1e … 0x1) means silicon ignores the bit.
- A reading of `… 0x1 0x101 …` means the table grew or allocated, which NOGROW forbids.

**Grading:**
1. **VOID boot:** sublet-life's E positions are not exact, or any RUN line's image sha16 or rec differs from
   SHA256SUMS / `.rec`.
2. **VOID pair:** b5b2-nogrow-ctl does not read its vector. Without the control's 30, the NOGROW arm's no-trap
   reading cannot be told from a CDERIVE that never reached the full-table check.
3. **CONFIRMED ON SILICON:** b5b2-nogrow reads every position exactly, on a pair that is not VOID.
4. **FINDING:** anything else, reported with its values.
