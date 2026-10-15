# Silicon predictions for the plain corpora of memcached, FFmpeg and tshark (pre-registered, 2026-10-16)

This is section 5's prediction table from `docs/plans/2026-10-15-bug-corpus-on-fpga.md`. It is for the 80 plain
defects (45 plain-heap, 26 plain-temporal and 9 subobject), and it is committed before any of them runs on silicon.
The nested corpora wait for the lead's decision (b), the allocator.

The tool is `tools/predict-silicon.py`, and its docstring states the method. To regenerate:

    python3 capstone/bug-corpora/tools/predict-silicon.py --bounds-model <bounds-model.py> \
        --geometry-tsv capstone/bug-corpora/silicon-predictions/2026-10-16-plain/geometry.tsv --align 16

## Inputs

- **Geometry: `geometry.tsv`.** These are the GEOM lines of each plain-heap case's BUGGY run, measured on the virtual
  profile, where bounds are exact.
  - Build: every case with `-DCORPUS_GEOMETRY` (`tools/corpus-geometry.h`, the three plain-heap `shared/driver.c`),
    through `tools/run-virtual-cases.py --arm virtual-malloc`, one boot per corpus.
  - The hook is off in a default build, which stays byte-identical. The checked hashes were memcached
    `4c818a5709e49d87`, FFmpeg `f15ace270042b7fa` and tshark `cc182edd254d3908`, each the same with and without the
    edit, and the builds are deterministic.
- **Platform:**
  - QEMU: capstone-qemu `d35af47b163c` (sha256 `b745cb491fdc`);
  - SDK: virtual, mallocng, built from llvm-capstone `d7cd518af8dd` (runtime archive `57328b8451c994d2`; clang
    `3b9a85a22b2c`);
  - adapter: capstone-vexec `34a4c18b16a9`, capstone_vm.ko `26ad747d0b73`;
  - images: Image `e58613598c89`, fw_jump `6f2b082cb677`, rootfs `9903242c37a7`.
- **Bounds model:** `bounds-model.py` from capstone-ariane `capstone-sublet` `b7e3287e4`
  (`verif/tests/custom/capstone/`, sha256 `0b0377dc0d20436a`), `cderive_child`. It reproduces the RTL lane's own
  example: 65535 B at a +0x10 cursor has slack 16 below and 113 above. Every object below 4096 B is exact at any
  4-byte-aligned cursor.

## Results

| | child at the request size | child at the size rounded to the alignment |
|---|---|---|
| 16-byte placement | 45/45 spatial CAUGHT, as virtual | 40/45: memcached 00, FFmpeg 01, 04, 05 and tshark 03 MISSED |
| 8-byte placement | 44 caught; tshark 00 (8192 B) DEPENDS on placement (1 of 2) | 41 caught, 3 missed (memcached 00, FFmpeg 04, tshark 03), 1 placement-dependent (tshark 00) |

- **Spatial.**
  - All 45 virtual catches are accesses outside their object: 44 past its end, 1 before its start.
  - Sizes run from 1 to 8192 B. Every object is exact under the encoder at a 16-byte cursor, so an allocator that
    derives each child at the request size keeps every catch.
  - An allocator that rounds the child up to its alignment turns the one-past-the-end reads of small odd-sized
    objects into in-bounds reads.
- **Temporal (26).** Two columns, because decision (b) picks:
  - an allocator that revokes on free gives CAUGHT with cause 25. QEMU reads 24 for these, because its reload
    untags (ISSUES Q-11); silicon was measured on 2026-10-16 (stale-access arm, `caplifive_sublet_d93a8d66b`);
  - level0 gives MISSED, because it does not revoke.
- **Subobject (9).** MISSED, as on the virtual profile. The crossing stays inside one allocation, and rounding only
  enlarges bounds.

## Controls

- **Positive.** The tool refuses to print if a synthetic 65535 B object read one byte past its end is predicted
  CAUGHT at every placement. It reads MISSED, so the rounding reaches the verdict.
- **Cross-check against an independent measurement.** CheriBSD's malloc hands out bounds rounded up to its size
  class. Its measured table reads request 17 → 32 bytes and request 24 → 32 bytes
  (`ffmpeg/plain-heap-repros/results/2026-10-08-cheribsd/size-classes-2026-10-10.txt`): a multiple of 16 for
  these sizes.
  - CHERI recorded NOT CAUGHT for these plain-heap cases (`cheribsd-subobject`): memcached 00, FFmpeg 01, 04 and 05,
    and tshark 03. That is the same five this table predicts MISSED under rounding to 16, with the other 40 caught
    both ways.
  - This is CONSISTENT with CHERI, not an independent proof. CheriBSD's rounding is measured at 17 and 24 but only
    inferred at 1 and 9, the sizes of tshark 03 and memcached 00. Both allocators round the same way, so the
    agreement is partly arithmetic.
  - What it does show is that the measured geometry (size and offset) matches CHERI's record case by case.

## Assumptions (say which one held when the board reads)

- The silicon allocator derives each object's capability with CDERIVE (or the same encoder) at a base placed on its
  alignment. The two columns cover the request size and the rounded size.
- The probe's access is the object's own pointer plus the measured offset. Both profiles use 16-byte capabilities,
  so an object's layout and size as measured on the virtual profile are its silicon layout and size.

## Read on silicon (2026-10-16)

The board ran the plain corpora on `caplifive_sublet_d93a8d66b` (bundle `silicon-runs/2026-10-16-plain`, read into each
corpus as the `silicon-lifetimes` and `silicon-level0` arms by `tools/silicon-verdicts.py`).
- **The assumption that held is the first column: the child at the request size, on a 16-byte placement.** That is
  level0 with `CAPSTONE_LEVEL0_LIFETIMES`. Its mechanism was shown on silicon first by the board lane's B6:
  - a group derived from a malloc'd block has exact bounds;
  - CREVOKE ends a child and its grandchild, and a fresh child reuses the range.
- **silicon-lifetimes:** 46 spatial CAUGHT at the labelled probe (cause 28, the RTL's OUT_OF_BOUNDS) and 26 temporal
  CAUGHT with cause 25. That is every row as predicted, the duplicate FFmpeg 13 included.
- **silicon-level0:** 46 spatial CAUGHT and 26 temporal MISSED, as predicted for an allocator that does not revoke.
- The rounded-size column was not the configuration measured, so its five predicted misses remain predictions.
