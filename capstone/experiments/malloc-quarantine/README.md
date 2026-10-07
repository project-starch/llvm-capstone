# CheriBSD heap quarantine: what it costs the malloc/free interface

Experiments on default CheriBSD purecap (CHERI-RISC-V, QEMU). The question is
what the Malloc Revocation Shim (MRS) quarantine does to the heap of
unmodified programs. Capstone follows once the CheriBSD side is understood.

**Evaluated configuration:** the CheriBSD default, revocation enabled,
asynchronous, quarantine ratio 1/4. Nothing is normalised to a run without
revocation; every figure is per byte the program itself holds, measured by a
tracer in the same process.

**Programs:** the fixed-work programs of mimalloc-bench, unmodified apart from
portability fixes (listed under Build fixes).

QEMU time is not a performance measure here. Everything reported is counted:
bytes from jemalloc's ledger, bytes the program holds, the kernel's max RSS,
revocation epochs, kernel sweep counters, resident pages per mapping, and
allocation counts.

## Setup

| Item | Value |
|---|---|
| Guest | `cheribsd-riscv64-purecap.img`, kernel CHERI-PURECAP-QEMU (CheriBSD source 88f39900c), booted `-snapshot -smp 1 -m 2048` by `vm.sh` |
| Allocator | jemalloc 5.2.1 behind MRS (`lib/libc/stdlib/malloc/mrs/mrs.c`) |
| System defaults | `security.cheri.runtime_revocation_default=1`, `..._async=1` |
| Benchmarks | mimalloc-bench 69c41ed, built by `build-mb.sh` with the SDK's clang (`cheribsd-riscv64-purecap.cfg`, `-O3`) |
| Parallel VMs | `vm.sh` takes `MQ_WORK` and `MQ_PORT`; three guests ran side by side on ports 10461–10463 |

`run.sh OUTDIR ARMS -- PROGRAM ARGS` starts every run from an empty
environment and sets only MRS's own switches. The arm used for results is
`on`: `_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1`, which
is what the system does by default. A different ratio is set per run with
`env _RUNTIME_QUARANTINE_NUMERATOR=1 _RUNTIME_QUARANTINE_DENOMINATOR=N` in
front of the program. The arm `off` (`_RUNTIME_REVOCATION_DISABLE=1`) is used
once, as an instrument check of the tracer, never as a result. `run.sh` also
knows `sync`, `q8` and `q2`; they were not evaluated.

## Instruments

- **`build-mb.sh BENCH_DIR OUT_DIR`** cross-builds cfrac, espresso, barnes,
  glibc-simple, sh6bench, mstress, malloc-large and alloc-test.
- **`build-probes.sh OUT_DIR`** builds the two LD_PRELOAD probes.
- **`mqstat.so`** (`mqstat.c`) prints at exit the revocation epochs,
  jemalloc's ledger and the kernel's sweep counters
  (`cheri_revoke_stats`). It interposes `cheri_revoke` so that MRS's own
  `TAKE_STATS` calls cannot zero the counters before they are read; `calls=`
  on its output line must be non-zero whenever `passes` is.
- **`mqtrace.so`** (`mqtrace.c`) interposes malloc, calloc, realloc, free,
  posix_memalign and aligned_alloc and forwards each call to libc unchanged.
  Every `MQ_SAMPLE` allocations (default 4096) it prints jemalloc's ledger,
  the usable bytes the program holds (`live_req`), the bytes it asked for
  (`live_asked`) and the epochs. With `MQ_TRACK=1` it also records the reuse
  distance of every address and the distinct lines and pages handed out per
  window; those tables are mmap'd outside jemalloc, but they are inside max
  RSS, so footprint runs leave tracking off. At exit it prints the exact peak
  of the held bytes (`peak_live`), the reuse histogram and the stride
  histogram. Single-threaded programs only.
  - Cost: about 20% more QEMU time without tracking (mstress: 7.1 s against
    5.9 s), 2.2–3× with tracking (glibc-simple: 2173 s against 986 s).
  - `./traced PROGRAM ...` runs a program under both probes. The result
    directories name the tracer version that ran: `traced` (v1, no
    `live_asked`), `traced2` (adds `live_asked` and stride), `traced3` (adds
    `peak_live`).
- **`run-maps.sh OUT PROGRAM ...`** (guest side) runs a program and every
  20 s sums the resident pages of its mappings by owner with `procstat -v`:
  jemalloc's named extents, MRS's named descriptor slabs, files, other.
- **Analysis and figures:** `analyze.py` (parser), `plot-scatter.py`,
  `plot-held.py`, `plot-ratio.py`, `plot-reuse-cdf.py`, `plot-frag.py`,
  `plot-maps.py`, `plot-live.py`; `analyze-mb.py` and `plot-mb.py` summarise
  untraced runs by max RSS.

## How MRS decides to revoke (read in `mrs.c`)

- `free` paints the shadow bitmap and appends the object to the active
  quarantine arena, with one 16-byte capability per object in an mmap'd
  descriptor slab (`mrs.c:354-362`, `548-553`). jemalloc does not get the
  object back.
- The check runs only on allocation (`mrs.c:764-791`).
  - Nothing happens until `allocated_size` (live + all quarantine) reaches
    8 MiB (`MIN_REVOKE_HEAP_SIZE`).
  - After that it fires when `active_arena.size * den >= allocated_size * num`.
- **Asynchronous mode** (`mrs.c:810-861`): the full arena is swapped out and
  the kernel sweeps it in the background. It is flushed back to jemalloc at a
  later check, once its epoch has cleared. Until then its bytes stay in
  `allocated_size` (subtracted only at flush, `mrs.c:1037`), yet the trigger
  compares only the *active* arena.

**Model** for a steady live set L and ratio r, with q the arena size at the
trigger: q ≥ r(L + 2q), so q = rL/(1−2r), and jemalloc's `allocated` cycles
between L+q = L(1−r)/(1−2r) and L+2q = L/(1−2r). For the default r = 1/4
that is **1.5·L to 2·L**, not the documented 4/3·L. For r ≥ 1/2 the trigger
cannot settle.

## Results

All runs are single runs unless a range is given. Result files are under
`/tmp/capstone/malloc-quarantine/results/` on the host that ran them; only
the result lines quoted here are committed.

### R1. Heap per live byte, all programs (`results/e11e-mb`, `fig/scatter.png`)

Peak of jemalloc `allocated` (live + quarantine) against the peak bytes the
program held; `plot-scatter.py`. Max RSS is the kernel's.

| Program | holds (MiB) | heap (MiB) | heap / holds | max RSS (MiB) | passes |
|---|---|---|---|---|---|
| glibc-simple | 0.12 | 10.83 | 87 | 34.5 | 429 |
| cfrac | 0.92 | 10.73 | 11.6 | 24.9 | 339 |
| espresso | 1.15 | 11.60 | 10.1 | 28.2 | 963 |
| mstress 1 50 25 | 7.40 | 17.27 | 2.33 | 18.6 | 24 |
| alloc-test 1 | 12.54 | 25.36 | 2.02 | 37.8 | 273 |
| sh6bench 1 | 295.2 | 579.3 | 1.96 | 750.9 | 179 |
| malloc-large | 436.0 | 805.1 | 1.85 | 1034 | 159 |
| barnes (control) | 894.0 | 894.1 | 1.00 | 82.4 | 0 |

- **Below the 8 MiB floor** (glibc-simple, cfrac, espresso) the heap is
  10.7–11.6 MiB whatever the program holds: the quarantine must reach 8 MiB
  before the first pass, and the revoking arena is held on top.
- **Steady live sets** (alloc-test, sh6bench, malloc-large) sit at the model
  peak: 1.85–2.02.
- **mstress** is above it (R3).
- **barnes** never frees in its loop: no pass, heap = live. It touches only
  82 MiB of the 894 MiB it allocates, hence its max RSS.
- mstress at SCALE 25/50/100/200/400 holds 3.6/7.4/11.4/29.8/64.2 MiB and
  has heap/holds 2.65/2.33/2.55/2.30/2.34 (`results/e11b-mstress`).

Earlier untraced runs (`results/e3-mb`, `results/e9-rep`) give the same max
RSS: mstress 18.5–19.8 MiB and espresso 27.4–28.9 MiB over three runs each.

### R2. Steady live set: alloc-test (`fig/alloctest-held.png`)

`plot-held.py` on the traced alloc-test run, window 20.0–21.0 M allocations,
`--model 0.25`. The program holds 12.5 MiB throughout; `allocated` saw-tooths
between the two model lines. Over the whole plateau after the first pass
(12 137 samples): min 1.48, max 2.03, mean 1.77 of live; model 1.50 / 2.00 /
1.75. Two repeats (`results/e15-rep`) give the same three numbers to three
decimals and the same 273 passes; max RSS 37.8–37.9 MiB.

### R3. Bursty release: mstress (`fig/mstress-held.png`)

mstress builds up to 30 MiB per iteration and frees it in one burst. While
the program holds 2–3 MiB, `allocated` stays near 37 MiB; its peak is
68.5 MiB, 2.30 of the held peak. MRS checks its threshold only in `malloc`
(`mrs.c:764`), so nothing throttles a release burst, and a smaller ratio
barely helps (`results/e11c-ratio`, mstress 1 200 25, heap/holds at the
peak):

| r | 1/16 | 1/8 | 1/4 | 1/2 |
|---|---|---|---|---|
| heap / holds | 2.14 | 2.30 | 2.30 | 3.66 (not settled) |
| passes | 50 | 24 | 23 | 2 |

### R4. The ratio knob on alloc-test (`results/e11f-ratio-alloctest`, `fig/alloctest-ratio.png`)

`plot-ratio.py`. Band = min to max of `allocated`/live over the plateau;
sweep pages are `pages_scan_ro + pages_scan_rw` per 10⁶ allocations.

| r | measured | model | passes | sweep pages / 10⁶ allocs |
|---|---|---|---|---|
| 1/16 | 1.10–1.16 | 1.07–1.14 | 1911 | 98 589 |
| 1/8 | 1.20–1.36 | 1.17–1.33 | 819 | 48 055 |
| 1/4 (default) | 1.48–2.03 | 1.50–2.00 | 273 | 21 820 |
| 1/2 | 3.06–32.0, growing | diverges | 15 | 12 733 |

The measurement sits 0.02–0.03 above the model throughout; not investigated.
Halving the footprint overhead costs 2.2× the sweep work, quartering it
4.5×.

### R5. Reuse distance and working set (`results/e11a-reuse`, `fig/reuse-cdf.png`)

`MQ_TRACK=1`. Reuse distance = allocations between the `free` of an address
and its next allocation.

| Program | median | share ≤ 2¹² | fresh addresses | lines per 4096 allocs | pages per 4096 allocs |
|---|---|---|---|---|---|
| mstress 1 50 25 | ≤ 2¹⁴ | 0.4% | 10.0% | 75 716 | 1310 |
| espresso | ≤ 2¹⁷ | 0.6% | 3.3% | 14 097 | 243 |
| glibc-simple | ≤ 2¹⁹ | 0.0% | 0.9% | 2390 | 38.3 |
| glibc-simple, revocation off (instrument check) | ≤ 128 | 99.98% | 0.0% | 311 | 5.6 |

With the quarantine, essentially no address comes back within 2¹²
allocations; glibc-simple's LIFO pattern touches 7.7× the lines and 6.8×
the pages per window. This measures how scattered the allocator's output is,
not the program's cache misses: QEMU models no caches.

The allocation stride (distance between consecutive allocations,
`plot-stride.py`) was also recorded. It mostly reflects how many size
classes a program mixes (alloc-test: 45% within 64 B; mstress: 6%), not the
quarantine, and is not used.

### R6. Where the bytes go (`results/e12-frag`, `fig/e12-frag.png`)

`plot-frag.py`, tracer v2 with `MQ_TRACK=1 MQ_WIN=0`. Shares of jemalloc's
resident bytes, averaged over every sample at which the program holds at
least half of its peak:

| Program | resident (MiB) | asked | rounding | quarantine | holes | dirty |
|---|---|---|---|---|---|---|
| glibc-simple | 15.0 | 0.6% | 0.0% | 45.0% | 0.1% | 54.3% |
| cfrac | 15.3 | 4.1% | 0.5% | 41.6% | 17.6% | 36.3% |
| alloc-test | 30.4 | 37.7% | 3.4% | 31.6% | 11.2% | 16.2% |
| espresso | 16.7 | 4.3% | 0.1% | 38.9% | 2.5% | 54.2% |
| mstress 1 200 25 | 68.2 | 35.8% | 1.8% | 52.6% | 0.7% | 9.1% |
| barnes (control) | 902.5 | 89.8% | 9.3% | 0.0% | 0.0% | 0.9% |

asked = bytes requested; rounding = jemalloc's size classes; quarantine =
`allocated` − usable live; holes = `active` − `allocated`; dirty =
`resident` − `active`. The quarantine is the largest layer of every program
that frees, larger than all classic fragmentation together. For the three
smallest heaps the program itself is almost invisible (1–4%).

### R7. Bookkeeping outside the heap (`results/e14-maps`, `fig/e14-maps.png`)

`run-maps.sh` samples `procstat -v` every 20 s and sums resident pages by
mapping owner; MRS and jemalloc name their mappings. Peak resident MiB:

| Program | jemalloc | MRS descriptor slabs | binary + libs | other | sum | `time -l` max RSS |
|---|---|---|---|---|---|---|
| sh6bench 1 | 601.1 | 139.7 (26% of RSS) | 2.1 | 6.2 | 749.2 | 750.8 |
| alloc-test 1 | 26.7 | 5.9 (16%) | 3.4 | 0.9 | 37.0 | 39.2 |
| glibc-simple | 21.4 | 11.1 (42%) | 2.1 | 0.8 | 35.4 | 35.7 |
| mstress 1 200 25 | 74.3 | 1.2 (1.7%) | 2.1 | 1.1 | 78.7 | 80.0 |

MRS keeps one 16-byte capability per quarantined object in descriptor slabs
it mmaps and never unmaps (`mrs.c:354-362`, `548-553`); for sh6bench's small
objects that is 23% on top of jemalloc's heap, and for glibc-simple's 16-byte
objects the descriptor is as large as the object it guards (11.1 MiB beside a
heap whose quarantine holds about 10 MiB). The sums match the kernel's
max RSS within 2 MiB, so nothing is unaccounted. It is not jemalloc's
time-based page return: sh6bench with
`MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0` has the same max RSS
(`results/e7-decay0`).

### Instrument checks

- **Tracer does not change the heap:** mstress under the tracer ends with
  the same jemalloc ledger as without it (`allocated` 18 112 264 in both).
- **Zero where zero is right:** barnes, which never frees in its loop, shows
  0 passes and a 0.0% quarantine layer.
- **Reuse distance fires:** without the quarantine, glibc-simple reuses an
  address within 128 allocations (median); with it, not within 2¹⁸.
- **Sweep counters are read before MRS zeroes them:** `calls=359` on the
  sh6bench line with 179 passes.
- **Decay setting takes effect:** glibc-simple's max RSS drops from 36.5 to
  25.6 MiB with decay 0, while the same program without revocation (0.1 MiB
  of heap) stays at 3.1 → 3.0 MiB (`results/e7-decay0`).

### Kernel sweep work, default configuration (`results/e10-sweep`, `results/e11b-mstress`)

| Program | passes | pages scanned (ro + rw) | pages skipped fast | caps found | caps cleared |
|---|---|---|---|---|---|
| mstress 1 50 25 | 24 | 16 035 | 110 182 | 741 694 | 122 716 |
| sh6bench 1 | 179 | 2 000 664 | 1 166 735 | 390 253 567 | 1 344 399 |

### Caveats

- Each cell is one run unless a range is given. Asynchronous revocation is
  timing-dependent: mstress's max RSS varied by 7% over three runs.
- Part of max RSS is jemalloc's wall-clock page decay, which depends on
  QEMU's speed. The ledger (`allocated`) does not; the plots use it.
- The tracer samples every 4096 allocations; a band measured from samples can
  only be narrower than the true one. malloc-large (2001 allocations) was
  re-run with `MQ_SAMPLE=8`.
- Allocations libc makes without going through its PLT are invisible to the
  tracer and land in the quarantine layer of R6.
- Max RSS of a run with `MQ_TRACK=1` includes the tracer's tables and is not
  reported.

### Build fixes

- **sh6bench** needs mimalloc-bench's `-DBENCH=1 -DSYS_MULTI_THREAD=1`;
  without them it reads `argv[1]` as a file name and fails.
- **alloc-test** reads `/proc/self/statm`, which FreeBSD lacks; `fread` on
  the NULL `FILE*` raises a CHERI exception. `build-mb.sh` takes its
  `getrusage` path, and its `clock_gettime` path instead of `rdtsc`.
- **barnes** needs a `gets` shim.

## Not yet done

- What the same objects would need in Capstone's Sublet heap (256-byte
  buddy atoms, 65 536 identities): `peak_live_objects` and
  `peak_live_pow2_256` from tracer v4 (`results/e16-fit`, running).
- Instruction counts (deliberately left out for now).
- The Capstone side.
