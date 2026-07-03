# C1 object-granularity narrowing: coverage matrix + measured overhead

*Status: paper-feeding ground truth for the C1 (granularity) claim. Replaces the
prose "we narrow object bounds / overhead is green" framing the 2026-06-29 audit
flagged as an overclaim. Two honest scopes: (1) a **coverage matrix** — what is
and is not narrowed; (2) **measured overhead** — code size only. Data collected
2026-07-01 on `capstone-bootstrap`; measurement method in
`/tmp/capstone/c1-measurement-results.md` + `delegate-c1-measurement.md`.*

## 0. The claim, stated honestly

C1 narrows capability bounds at **selected object materializations** so that an
over-read/over-write past those objects **faults**. It is **not** a spatial-safety
theorem: broad `gp`/`sp` roots remain, subobjects/interior pointers are not
narrowed, permissions stay RWX, and function/code capabilities are un-narrowed.
The contribution is "measured object-granularity narrowing for the covered
classes, at low code-size cost, no correctness regression, and it catches real
OOB bugs" — see the coverage matrix (§1) for exactly which classes.

Bounds precision caveat (from `capability-bounds-model.md`): this QEMU keeps exact
fat bounds in a side table, so observable `SHRINK` is **exact at all sizes**; the
`<4 KiB exact / grain-above` rounding rule is spec-derived, not measured. Do not
cite rounding as experimental evidence.

## 1. Coverage matrix (what is / isn't narrowed)

| Object class | Narrowed? | Mechanism | Default | Evidence |
|---|---|---|---|---|
| **Sized data globals** | **Yes** | `selectLGA` emits `SHRINK` to `[&g, &g+sizeof(g))` (`CapstoneISelDAGToDAG.cpp`) | **on** (`-capstone-shrink-globals`) | lit `cap-shrink-globals.ll`; authority `global_oob` bounds-fault; §2 sizes; **rijndael OOB found** (8-byte write through `char r[4]`) |
| Unsized / extern globals | No | — | — | no size known at materialization |
| Function / code capabilities | No | — | — | broad; RX not separated from RW |
| **Heap — benchmark allocators** | **Partial** | `rv8_malloc.c` + dtoa `malloc_beebs` call `cap_shrink(p, b, b+n)` on return | source-level (**not** a libc/global policy) | RV8 `SHRINK OFF = 1` residual (§2) = this call, independent of the globals flag |
| Heap — general `malloc`/`free` | No | — | — | **proposed**, `bounded-heap-allocator-proposal.md` (task #78) |
| **Stack — fixed object (whole obj + interior/load-store base)** | **Yes (gated)** | `SHRINK` via shared `narrowToFrameObjectBounds`: bare `ISD::FrameIndex` **and** `materializeFrameIndexAddrBase` (ldc/stc base + interior ptr), narrowed to `[&obj, &obj+size)` | **off** (`-capstone-shrink-stack`) | lit `cap-shrink-stack.ll` on/off (`stack_idx`/`cap_slot`/`field_store`); authority `stack_oob` |
| **Stack — dynamic `alloca` (runtime-sized)** | **Yes (gated)** | `SHRINK` in `lowerDYNAMIC_STACKALLOC`: returned pointer narrowed to `[cursor, cursor+alignedSize)`, `sp`/X2 kept broad | **off** (`-capstone-shrink-stack`) | lit `cap-shrink-dynalloca.ll` on/off; authority `stack_dynalloca_{inbounds,oob}` (runtime OOB traps) |
| Stack — varargs save-area | **Yes (gated)** | covered via the fixed-object `narrowToFrameObjectBounds` path (the save area is a fixed frame object) | **off** (`-capstone-shrink-stack`) | confirmed: `shrink` emitted in a varargs function |
| Stack — variable-size / spill slots | No (by design) | — | — | excluded intentionally |
| Subobject / struct field | No | — | — | authority struct-field over-read = **no-trap-today** (confirms the gap) |
| `gp` root (globals base) / `sp` root (stack base) | No (broad) | — | — | segment-granular; single `PT_LOAD` ≈ whole image |

**Residual gap set** (the honest "not covered" list for the paper): subobjects,
stack varargs/dynamic-alloca, general heap, function caps, RWX permissions, and
the two broad roots. Object bounds re-derive CHERI; the Capstone-specific angle
(linearity / `SPLIT` / root-elimination) is separate — see the heap proposal and
the audit's reframing.

## 2. Measured overhead — code size only

Method: build each domain twice — default (**ON**) vs `-mllvm
-capstone-shrink-globals=false` (**OFF**) — and diff `llvm-size` text and the
count of emitted `SHRINK` R-type encodings (`funct7=1, funct3=1, opcode 0x5b`).
**Full coverage: CoreMark + all 7 RV8 + all 82 BEEBS = 90 domains**, measured in
two passes (a representative 15 BEEBS first, then the remaining 67); all 90 OFF
builds passed their correctness markers (narrowing changes bounds, not results).
The table below shows CoreMark + RV8 + the first 15 BEEBS; the remaining 67 BEEBS
are in the §5 appendix, and the aggregate stats in **Findings** are over all 90.
`Δtext%` is `(text_ON − text_OFF) / text_OFF`; `B/shrink` is
`Δtext / (SHRINK_ON − SHRINK_OFF)`.

| benchmark | text ON | text OFF | Δtext | Δtext% | SHRINK ON | SHRINK OFF | B/shrink |
|---|---:|---:|---:|---:|---:|---:|---:|
| coremark | 23396 | 21908 | 1488 | 6.79% | 95 | 0 | 15.7 |
| rv8 dhrystone | 8234 | 7598 | 636 | 8.37% | 40 | 1 | 16.3 |
| rv8 qsort | 9012 | 8868 | 144 | 1.62% | 10 | 1 | 16.0 |
| rv8 sha512 | 8340 | 8200 | 140 | 1.71% | 10 | 1 | 15.6 |
| rv8 aes | 22532 | 21960 | 572 | 2.60% | 41 | 1 | 14.3 |
| rv8 primes | 43900 | 43740 | 160 | 0.37% | 11 | 1 | 16.0 |
| rv8 norx | 12884 | 12620 | 264 | 2.09% | 19 | 1 | 14.7 |
| rv8 miniz | 68488 | 67512 | 976 | 1.45% | 65 | 1 | 15.3 |
| beebs rijndael | 32925 | 32485 | 440 | 1.35% | 26 | 0 | 16.9 |
| beebs crc32 | 2176 | 2144 | 32 | 1.49% | 2 | 0 | 16.0 |
| beebs nettle-aes | 17404 | 17152 | 252 | 1.47% | 16 | 0 | 15.8 |
| beebs nettle-arcfour | 1972 | 1888 | 84 | 4.45% | 6 | 0 | 14.0 |
| beebs nettle-cast128 | 28076 | 26120 | 1956 | 7.49% | 150 | 0 | 13.0 |
| beebs nettle-des | 31816 | 31660 | 156 | 0.49% | 10 | 0 | 15.6 |
| beebs nettle-md5 | 7412 | 7288 | 124 | 1.70% | 8 | 0 | 15.5 |
| beebs nettle-sha256 | 12607 | 12435 | 172 | 1.38% | 12 | 0 | 14.3 |
| beebs picojpeg | 38438 | 33898 | 4540 | 13.39% | 305 | 0 | 14.9 |
| beebs huffbench | 6633 | 6517 | 116 | 1.78% | 8 | 0 | 14.5 |
| beebs sglib-arraysort | 3012 | 2824 | 188 | 6.66% | 12 | 0 | 15.7 |
| beebs sglib-arrayheapsort | 2544 | 2340 | 204 | 8.72% | 13 | 0 | 15.7 |
| beebs sglib-arrayquicksort | 3012 | 2824 | 188 | 6.66% | 12 | 0 | 15.7 |
| beebs sglib-rbtree | 13360 | 13244 | 116 | 0.88% | 8 | 0 | 14.5 |
| beebs stringsearch1 | 5744 | 5332 | 412 | 7.73% | 22 | 0 | 18.7 |

**Findings (over all 90 domains)**
- **Cost per narrowed global is a near-constant ~15.6 bytes** (range 12.9–19.2,
  mean 15.6), matching the fixed `lcc cursor / add size / shrink` materialization
  sequence. This holds uniformly across the whole corpus — including the extreme
  outliers — so the code-size overhead is essentially **(number of narrowed sized
  globals) × ~15.6 bytes**, a function of *global-materialization density*, not
  program size.
- **Percentage overhead therefore spans a wide range**: **median 1.83%, mean
  4.17%**, from **0%** (4 domains — `cover`/`fac`/`fibcall`/`sqrt` — materialize
  no sized globals, 0 shrinks) to **46.1%** (`statemate`). Distribution: **0%: 4 |
  0–3%: 57 | 3–10%: 22 | >10%: 7** of 90.
- **The high-overhead tail is auto-generated, globals-dense code**: `statemate`
  46.1% (588 shrinks) and `nsichneu` 33.7% (1083 shrinks) are generated WCET
  state machines with huge static tables; then `compress` 24.6%, `dijkstra`
  20.7%, `qrduino` 15.6%, `picojpeg` 13.4%, `ud` 12.8%. The mean is pulled up by
  these; the **median (1.8%) is the representative figure** for ordinary code.
- **No correctness cost**: every OFF build still passes its marker; the ON build
  additionally traps the rijndael OOB write that OFF silently allows.
- **`SHRINK OFF = 1` on RV8 (and `dtoa`)** is the source-level heap `cap_shrink`
  in `rv8_malloc.c` / dtoa `malloc_beebs` — independent of the globals flag. The
  other 82 domains show OFF = 0 (no bounded allocator linked). This confirms
  globals-narrowing and heap-narrowing are separately controlled.

## 3. Stated limitations (do NOT overclaim)

- **Code size only.** No runtime-cycle or dynamic-instruction overhead was
  measured: this QEMU is functional, not cycle-accurate, and no no-edit
  `-icount`/plugin path was available. Any runtime-overhead claim needs a
  cycle-accurate model or an instrumented instruction count — **future work**,
  not claimed here.
- **Not a spatial-safety theorem** — see the §1 residual gap set.
- **Bounds exactness is a property of this QEMU's side table**, not a measured
  compressed-encoding result (`capability-bounds-model.md`).
- Coverage is now **complete for this corpus** (CoreMark + 7 RV8 + all 82 BEEBS =
  90 domains); no BEEBS was skipped. It is still one benchmark corpus — the
  distribution should not be extrapolated verbatim to arbitrary application code,
  though the constant ~15.6 B/global cost model is expected to generalize.

## 4. Pointers
- Raw data: `/tmp/capstone/c1-measurement-results.md` (CoreMark+RV8+15 BEEBS) and
  `/tmp/capstone/c1-measurement-results-beebs-rest.md` (remaining 67 BEEBS);
  method: `/tmp/capstone/delegate-c1-measurement.md`,
  `delegate-c1-remaining-beebs.md`.
- Mechanism: `CapstoneISelDAGToDAG.cpp` (`selectLGA`, `-capstone-shrink-globals`);
  lit `cap-shrink-globals.ll`, `cap-shrink-stack.ll`.
- Evidence suite: `../../tests/capstone-authority/` (`global_oob`/`stack_oob`/
  struct-field over-read).
- Related: `capability-bounds-model.md` (precision), `bounded-heap-allocator-proposal.md`
  (heap, task #78), audit `../history/29-06-2026_15-08-22_granularity-provenance-audit.md`.

## 5. Appendix — remaining 67 BEEBS (completes the 90-domain corpus)

Same method as §2 (text bytes, `SHRINK` counts; all OFF builds passed markers).

| benchmark | text ON | text OFF | Δtext | SHRINK ON | SHRINK OFF |
|---|---:|---:|---:|---:|---:|
| aha-compress | 3580 | 3516 | 64 | 4 | 0 |
| aha-mont64 | 3160 | 3044 | 116 | 9 | 0 |
| bs | 1140 | 1092 | 48 | 3 | 0 |
| bubblesort | 2080 | 2032 | 48 | 3 | 0 |
| cnt | 3020 | 2892 | 128 | 8 | 0 |
| compress | 7870 | 6318 | 1552 | 103 | 0 |
| cover | 8876 | 8876 | 0 | 0 | 0 |
| crc | 2344 | 2172 | 172 | 11 | 0 |
| ctl-stack | 4732 | 4448 | 284 | 20 | 0 |
| ctl-string | 10218 | 9602 | 616 | 40 | 0 |
| ctl-vector | 10500 | 9840 | 660 | 44 | 0 |
| cubic | 45600 | 45352 | 248 | 15 | 0 |
| dijkstra | 4056 | 3360 | 696 | 46 | 0 |
| dtoa | 32096 | 31612 | 484 | 32 | 1 |
| duff | 2544 | 2512 | 32 | 2 | 0 |
| edn | 5464 | 5196 | 268 | 18 | 0 |
| expint | 2004 | 1928 | 76 | 5 | 0 |
| fac | 1044 | 1044 | 0 | 0 | 0 |
| fasta | 13432 | 13272 | 160 | 10 | 0 |
| fdct | 3976 | 3928 | 48 | 3 | 0 |
| fibcall | 1064 | 1064 | 0 | 0 | 0 |
| fir | 7360 | 7324 | 36 | 2 | 0 |
| frac | 44128 | 44032 | 96 | 6 | 0 |
| insertsort | 2132 | 2116 | 16 | 1 | 0 |
| janne-complex | 1616 | 1584 | 32 | 2 | 0 |
| janne_complex | 1244 | 1188 | 56 | 4 | 0 |
| jfdctint | 5748 | 5684 | 64 | 4 | 0 |
| lcdnum | 1560 | 1528 | 32 | 2 | 0 |
| levenshtein | 2447 | 2351 | 96 | 6 | 0 |
| ludcmp | 35392 | 34972 | 420 | 27 | 0 |
| matmult-float | 15204 | 15084 | 120 | 8 | 0 |
| matmult | 5508 | 5360 | 148 | 9 | 0 |
| mergesort | 6284 | 6156 | 128 | 8 | 0 |
| miniz | 67879 | 66943 | 936 | 62 | 0 |
| minver | 36920 | 36600 | 320 | 21 | 0 |
| nbody | 44012 | 43816 | 196 | 13 | 0 |
| ndes | 7484 | 6952 | 532 | 35 | 0 |
| newlib-exp | 33728 | 33580 | 148 | 9 | 0 |
| newlib-log | 34264 | 34204 | 60 | 4 | 0 |
| newlib-mod | 33452 | 33344 | 108 | 7 | 0 |
| newlib-sqrt | 33408 | 33156 | 252 | 17 | 0 |
| ns | 3848 | 3796 | 52 | 3 | 0 |
| nsichneu | 66404 | 49676 | 16728 | 1083 | 0 |
| prime | 2308 | 2260 | 48 | 3 | 0 |
| qrduino | 28304 | 24492 | 3812 | 248 | 0 |
| qsort | 33984 | 33664 | 320 | 20 | 0 |
| qurt | 35664 | 34936 | 728 | 46 | 0 |
| recursion | 1768 | 1736 | 32 | 2 | 0 |
| select | 33680 | 33320 | 360 | 23 | 0 |
| sglib-arraybinsearch | 1448 | 1384 | 64 | 4 | 0 |
| sglib-dllist | 9148 | 9048 | 100 | 7 | 0 |
| sglib-hashtable | 7120 | 6924 | 196 | 13 | 0 |
| sglib-listinsertsort | 5124 | 5024 | 100 | 7 | 0 |
| sglib-listsort | 2652 | 2552 | 100 | 7 | 0 |
| sglib-queue | 3480 | 3448 | 32 | 2 | 0 |
| slre | 12889 | 12749 | 140 | 9 | 0 |
| sqrt | 32316 | 32316 | 0 | 0 | 0 |
| st | 44572 | 44324 | 248 | 15 | 0 |
| statemate | 25828 | 17680 | 8148 | 588 | 0 |
| stb_perlin | 48240 | 48092 | 148 | 9 | 0 |
| strstr | 1932 | 1900 | 32 | 2 | 0 |
| tarai | 2008 | 1960 | 48 | 3 | 0 |
| trio-snprintf | 37890 | 37182 | 708 | 46 | 0 |
| trio-sscanf | 37774 | 37026 | 748 | 49 | 0 |
| ud | 3280 | 2908 | 372 | 20 | 0 |
| whetstone | 49448 | 48396 | 1052 | 70 | 0 |
| wikisort | 23052 | 22956 | 96 | 5 | 0 |
