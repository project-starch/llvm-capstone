# SQLLogicTest on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 9 (`docs/plans/2026-10-16-board-campaign-all.md`). It repeats measurements doc §7c
on `caplifive_sublet_d93a8d66b.bit`. §7c ran seven files, one boot each, on `caplifive_s12fix_5097eb166`, in boots
sw23-sw29 and B8.

## The images, rebuilt

§7c's images are no longer on disk, so the four region/heap classes of `slt-corpus/build-slt-corpus-images.sh` are
rebuilt with the recipe unchanged: SQLite at -O0, the support code at -O1.
- Tree: lane b0-silicon-runtime, which carries dev's sqlite port and runtime.
- Compiler: `clang version 22.0.0git (llvm-capstone 612b3ec514c0)`.
- Builder: a private wrapper around the same `ports/sqlite/run-sqlite-slt.sh` calls, because the lane script's
  freshness gate cannot check a copied toolchain directory (rc 2). The wrapper prints the compiler identity
  instead.
- Each class is validated under QEMU (the main clone's capstone-qemu deb7d75756) on its own file against the native
  baseline. The 1m class is also run on the three other files it serves.

| class | heap / stack / region | image | host | files it serves, each run under QEMU and equal to native |
|---|---|---|---|---|
| 1m | 256 KiB / 2 MiB / 1 MiB | `c3ad302bf07931d8` | `d511aada7085e5df` | select2, negctl, aggfunc, select1 |
| s3 | 1 MiB / **1,507,328** / 2 MiB | `9c439f97400fe092` | `99255c63bf641860` | select3 |
| s5 | 2 MiB / **458,752** / 2 MiB | `4f32d97ca1078825` | `99255c63bf641860` | select5 |
| s4 | 1 MiB / **1,507,328** / 4 MiB | `6550d406a2c41193` | `c14ac0bb9589ce39` | select4 |

- **The 1m image rebuilt byte-identical four times**, once per file it serves (`c3ad302bf07931d8` each time), so the
  build is deterministic.
- The two 2 MiB-region hosts are the same file: the host depends only on the region size.
- **Three stacks are smaller than build-slt-corpus-images.sh's** (bold).
  - **Why:** today's code is 1,445,112 bytes, 1.05× the largest that fits the kernel's order-10 allocation beside a
    2 MiB stack. With the script's stacks, all three classes fail its own budget gate: `domdata-budget.py` reads
    "order 11 exceeds the kernel's maximum 10".
  - **The new stacks:** each is the largest 64 KiB multiple that fits beside that class's carve. The order-10 limit
    leaves 2,740,992 bytes of domain data. The carve is 1,187,712 for the 1 MiB heaps and 2,236,288 for the 2 MiB
    heap.
  - **Why the stacks suffice:** stack depth is deterministic for a fixed program and input. So each class's QEMU run,
    equal to native, has already reached the depth the board run will reach, at the same stack.
    - Not shown: whether an overflow would fault or silently overwrite the domain's own storage. QEMU and the board
      would then differ only where their bounds checks differ.
  - select4's first QEMU run at the new stack hit the smoke runner's default guest-command timeout (30 × 12 = 360 s),
    still inside the run. With `CAPSTONE_GUEST_COMMAND_TIMEOUT=7200` it completed equal to native in about 8 minutes,
    on the same image hash.
  - All of it is in `build.result-lines.txt`, failed attempts included (the image build log's result lines).

**The control:** the images enter at 0x10000, so the control is the k800 relinked to 0x20000 (`589ceee3853c6092`).
That control returned 4 twice on this bitstream and line in the Sublet acceptance's boot s10a3.

**The native baseline**, rebuilt from the same tree (`ports/sqlite/build-slt-native.sh`), one summary per file, is in
`native.txt`. Two figures moved since §7c, both in the inputs and neither on silicon:
- **The negative control** now has 23 records: statements 9 pass / 2 fail, queries 7 pass / 5 fail. §7c read 21 records
  and queries 6 / 4. The fixture gained two arms, one for each side of the fix, in the runner's empty-result fix of
  2026-09-30 (9f623adbb3cb). Its deliberate failures stay the comparator's positive control.
- **select4** reads `query_pass=2648 skip_big=184`. §7c read 2,617 passing plus 215 not compared, 31 of which were empty
  results. The same fix now compares those 31, as the correction under §4f's first SLT table says.

## The boots

One file per boot. Every boot runs the relinked k800 first and then the one SLT domain, which is the boot's only
unknown and runs last.
- Order: negctl, aggfunc, select1, select2, select5, select3, then select4, the longest (78 min on s12fix).
- Line: the classic line, d04bd83 / df0e9a8 / 93440d5, with the #3 module, `REFUSAL_RECORD=1` and
  `PREFLIGHT_DTS_NODES=65536`.
- Driver: a private per-boot driver modelled on `../h1-sublet-2026-10-16`'s copy of `drivers/board-r1e4.sh`. It
  checks staging, membership and the module marker, runs the entry watchdog, and keeps each boot's `fw_payload.bin`.
  The verdict is `slt-corpus/slt-read.py`'s, read from the run's own segment of the transcript.

**The region carves.** The plan asks for the 1-4 MiB regions to be checked under this monitor first.
- On the classic line the monitor is 93440d5, which carries the R-55 cursor fixes but not the exact-remainder region
  table.
- Already on this line and bitstream:
  - H1's three boots each carved six 4 MiB arenas and released them (`released pool rc=1` six times), with no refusal;
  - S1's boot 1 ran 2 MiB heaps, and a Sublet arm lent a 2 MiB arena.
  - An SLT region is the same `create_region` path, at 1, 2 or 4 MiB.
- The classic table's extra remainder fragments cost about 11 slots per invocation (ISSUES R-55). One SLT invocation
  per boot stays far below the 96-slot table.
- What a passing result cannot show: whether a remainder's decoded bounds overlap a later carve. On this table that
  is a bookkeeping defect without a fault (R-55), and an SLT result does not depend on it.

## Pre-registered readings

**Per boot:**
- `k800 retval=4`;
- the domain's `SLT-SUMMARY`, up to ` completed`, equal to the native line for its file, as in `native.txt`;
- `completed=1`;
- no refusal and no cause 25.

**The two positive controls:**
- negctl reproduces its native failures exactly: statements 2, queries 5, parse errors 1;
- aggfunc reproduces statements 1 and queries 10.

A clean file is therefore a clean file, not a comparator that cannot fire.

**Not established:** one draw per file, as §7c. This is a correctness run, so no cycle counts are read.
