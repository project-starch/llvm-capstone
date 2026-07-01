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
| **Stack — whole object** | **Yes (gated)** | `SHRINK` on bare `ISD::FrameIndex` | **off** (`-capstone-shrink-stack`) | lit `cap-shrink-stack.ll` on/off; authority `stack_oob` |
| Stack — interior ptr / varargs / dynamic `alloca` | No | — | — | task #77 (coverage toward default-on) |
| Subobject / struct field | No | — | — | authority struct-field over-read = **no-trap-today** (confirms the gap) |
| `gp` root (globals base) / `sp` root (stack base) | No (broad) | — | — | segment-granular; single `PT_LOAD` ≈ whole image |

**Residual gap set** (the honest "not covered" list for the paper): subobjects,
stack interior/varargs/alloca, general heap, function caps, RWX permissions, and
the two broad roots. Object bounds re-derive CHERI; the Capstone-specific angle
(linearity / `SPLIT` / root-elimination) is separate — see the heap proposal and
the audit's reframing.

## 2. Measured overhead — code size only

Method: build each domain twice — default (**ON**) vs `-mllvm
-capstone-shrink-globals=false` (**OFF**) — and diff `llvm-size` text and the
count of emitted `SHRINK` R-type encodings (`funct7=1, funct3=1, opcode 0x5b`).
All 23 OFF builds passed their correctness markers (narrowing changes bounds, not
results). `Δtext%` is `(text_ON − text_OFF) / text_OFF`; `B/shrink` is
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

**Findings**
- **Code-size overhead is low and predictable**: Δtext ranges **0.37%–13.39%**,
  **median ≈ 1.8%**. It scales with the *number of narrowed sized-global
  materializations*, not program size — `picojpeg` (305 shrinks, table-heavy) is
  the high end; `primes` (11) the low end.
- **Cost per narrowed global is ~13–19 bytes** (mean ≈ 15.5), consistent with the
  fixed `lcc cursor / add size / shrink` materialization sequence — i.e. a small
  constant, not a super-linear blowup.
- **No correctness cost**: every OFF build still passes its marker; the ON build
  additionally traps the rijndael OOB write that OFF silently allows.
- **`SHRINK OFF = 1` on every RV8** is the source-level heap `cap_shrink` in
  `rv8_malloc.c` — independent of the globals flag. CoreMark/BEEBS show OFF = 0
  (no bounded allocator linked). This confirms globals-narrowing and
  heap-narrowing are separately controlled.

## 3. Stated limitations (do NOT overclaim)

- **Code size only.** No runtime-cycle or dynamic-instruction overhead was
  measured: this QEMU is functional, not cycle-accurate, and no no-edit
  `-icount`/plugin path was available. Any runtime-overhead claim needs a
  cycle-accurate model or an instrumented instruction count — **future work**,
  not claimed here.
- **Not a spatial-safety theorem** — see the §1 residual gap set.
- **Bounds exactness is a property of this QEMU's side table**, not a measured
  compressed-encoding result (`capability-bounds-model.md`).
- Sample is CoreMark + all 7 RV8 + 15 representative BEEBS (globals-heavy chosen);
  the remaining ~67 BEEBS are expected to fall in the same 0–13% band but were not
  all measured.

## 4. Pointers
- Raw data: `/tmp/capstone/c1-measurement-results.md`; method:
  `/tmp/capstone/delegate-c1-measurement.md`.
- Mechanism: `CapstoneISelDAGToDAG.cpp` (`selectLGA`, `-capstone-shrink-globals`);
  lit `cap-shrink-globals.ll`, `cap-shrink-stack.ll`.
- Evidence suite: `../../tests/capstone-authority/` (`global_oob`/`stack_oob`/
  struct-field over-read).
- Related: `capability-bounds-model.md` (precision), `bounded-heap-allocator-proposal.md`
  (heap, task #78), audit `../history/29-06-2026_15-08-22_granularity-provenance-audit.md`.
