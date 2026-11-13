# Compiler-lever census against riscv-base, C8, and the memset switch on the arms' SDK (2026-11-14)

The runtime lane asked which levers that do not wait on the lead's gp-cursor decision are left on FR gcc train01 and
FR mcf, ranked and sized. This records:
- the census;
- the two levers built: C8, and the existing memset switch applied to the arms' SDK;
- the claims withdrawn on the way.

All counts are native-twin user instructions in the image (shift-1 capmix3, window 0), unless marked whole-run.

## Images compared

| | gcc train01 | mcf test |
|---|---|---|
| FR | cpugcc_r.strip.dom 1d2d347b47dbc0c6 | mcf_r.strip.dom b85ebf872e28ccfd |
| riscv-base, board 3 (fair/stages/*-rv-fair3, plain-rv64fre) | e4aba7b6b01392ad | b3fbd4807ef58e6b |

Tools are in perf/cclever:
- `levers.py`: class and function deltas, printing both image hashes;
- `fncost.py`: entries and instructions per call;
- `movctx.py`: moves by context;
- `offfold.py`: the C6 pool;
- `fetchbytes.py`, `idiom.py`.
riscv-base was reprofiled at shift 1 (perf/cclever/rvs1). The older shift-2 riscv-base profiles credit both 16-bit
instructions of a bucket to the first, and so misassign classes: mcf's conditional branches read +2.64 G there, and
-21 M at shift 1.

## Census

**Totals.**
- gcc: FR 12.437 G against 10.860 G, +1.576 G (+14.5 %).
- mcf: FR 26.698 G against 27.365 G (-2.4 %). Whole-run user, outside-image runtime included, is FR 27.616 G, about
  +0.9 %. mcf's gap is therefore CPI, not instruction count. The board split (FR final: 179.61 G in-domain of about
  185.4 G cycles) agrees.

**gcc by class** (FR minus riscv-base):

| class | delta |
|---|---|
| add / addi / cincoffset* | +375 M |
| scc gp | +223 M |
| moves | +219 M |
| constants | +145 M |
| bounds ops | +128.5 M |
| sp-relative traffic | +178 M |

**gcc by function.**
- memset +117 M: the same loop on both sides, but FR runs it 39.96 M times against 21.94 M, so 1.8x the bytes.
  The same call sites pass pointer-width sizes.
- The allocator path, per call: malloc 134 against 74 instructions, free 138 against 78.
- memcpy +71 M: FR makes 4.06 M calls against 1.68 M.

**The levers, and what each came to.**
- **C1, LSR:** about 0 on gcc (FR's slli is below riscv-base's). The mcf ceiling is -100 M.
- **C3, spills:** ceiling +178 M on gcc. On mcf, FR's sp-relative traffic is below riscv-base's.
- **C4, moves:** no concentrated idiom.
  - mcf's +166 M is spec_qsort, where movc replaces riscv-base's ld+add pairs, and the function as a whole runs
    510 M fewer instructions.
  - gcc's +219 M is spread across functions, 5-17 M each.
  - An entry copy of an argument into a callee-saved register pins the prologue to the entry block (__unlock,
    bitmap_set_bit). riscv-base has the same shape in bitmap_set_bit.
- **C5, bounds ops:** 128.5 M is the whole shrink and shrinkto count, not a redundant subset.
- **Allocator:** about 57 extra instructions per free. About 32 of those are the heap-lifetime lookup, which is
  policy. Same-policy trims come to about 10 M on gcc; the lever is parked as below a board run's noise.

## C8: 16-byte copies of unknown alignment, inlined with a run-time alignment test

What it is:
- compiler-memcpy-align-dispatch d5985250bc12, with the record in a9a012ff72d3;
- `-capstone-memcpy-align-dispatch`, default off;
- `capstone/tests/compiler-repros/C8-memcpy-align-dispatch`.
It is C-81's missing run-time alignment test, for 16-byte copies only. At 32 bytes and more, with both cursors 8 mod 16,
the middle granule is whole on both sides, and a two-way test would strip a tag memcpy keeps.

Results on gcc, FR+C8 against FR:
- user instructions -56.1 M (-0.44 %);
- memcpy calls 4.06 M -> 1.59 M;
- outputs identical;
- image 6e5e26eb8e538075.
- On FRCDK, C8 gives -56.3 M: llvm-c9 = compiler-capbase-rvc-c8 29027126f39b, image fbc2d73dbd7e51d3.
- perl and mcf images are byte-identical with and without it.

## The memset switch (CAPSTONE_MEMSET_CAP_ZERO), applied to the arms' SDK

The runtime lane's ranking listed a 16-byte-store memset as a new lever. It is not new: b68bbde23c79 (2026-11-09) was
measured at gcc -0.919 % and is on neither dev nor the arms' SDK.
- It was cherry-picked onto the SDK's source: runtime-memset-cap-zero-arms 7166a346d516, test beb1bbabe26a.
- The rtl lane read the silicon: stc of x0 writes both words as zero and clears the tag.
- The SDKs (perf/memset with the switch ON, perf/memset-off with it OFF) match natevict's by disassembly, except
  string_bounds_safe.c.obj in the ON build.
- The directed test runs on the native route (`tests/native-task/memset-cap-zero`): 832/832 cases both on and off,
  and the mutant fails.

Twins of FR+C8+memset against FR+C8, outputs identical:
- gcc -119.1 M (-0.93 %);
- mcf -51.7 M (-0.19 %);
- perl -5.6 M whole-run, about 4 k in the makerand window.
Images: cpugcc_r 08cc565c91238c66, perlbench_r 9fe973c9cde4e702, mcf_r 1b80e120aa62deaa.

## Withdrawn on the way

- **The first riscv-base was not board 3's.** fair/stages/*-rv-fair was used before *-rv-fair3. memset's excess read
  4.3x instead of 1.8x; the class split barely moved.
- **"mcf -2.4 %, all CPI" counted image instructions only.** Whole-run user is about +0.9 %. CPI still dominates.
- **C6 is not a lever.** C6 means folding an addi into the access offset; its strict pool was sized at -30 M on gcc and
  -63 M on mcf. `performCIncOffsetCombine` already performs the fold, and refuses the general case on purpose: the
  intermediate cursor can lose its tag (R-60, the address policy). The provably safe subset is about 7 M on gcc and 0
  on mcf.
- **The memset lever had already been built and measured** (b68bbde23c79).

What would have caught each:
- hashing the reference against the board stage before comparing (levers.py now prints both hashes);
- reading the code that emits or refuses the shape before sizing it.

## Not this lane's, noted for owners

- **virtual-capstone-runtime, cases 29-31 on the flat layout:** collect_stale_register, collect_stale_memory and
  collect_pins_dead_pcc fail on virtual-capstone's tip 0542c5556c itself. The runtime lane has passed them to the
  lead as unowned.
- **QEMU fidelity, low priority:**
  - helper_reg_set_cap_compressed has no rd == 0 guard;
  - after a fill clears a granule's tag, the granule's other bytes read back as the raw capability encoding rather than
    the cursor a plain load returned.
