# Juliet on the virtual Capstone profile

NIST SARD's Juliet C/C++ 1.3 test suite, C cases only, run on the virtual Capstone profile in QEMU: every case built
twice with the virtual SDK (a bad image with the defect, a good image without it), each image run as its own program
under the launcher, and each run judged from the fault the launcher and QEMU report.

Juliet is public domain and is not committed: `fetch-juliet.sh` downloads it, checks a pinned sha256 and always
extracts the tree afresh from the verified archive.

## What is measured

Two sets (`select.py`):

| Set | Classes | C cases |
|---|---|---|
| temporal | CWE-415 double free, CWE-416 use after free | 366 |
| spatial | CWE-121 stack overflow, CWE-122 heap overflow, CWE-124 buffer underwrite, CWE-126 buffer over-read, CWE-127 buffer under-read | 9,138 |

Left out, by name and counted in each manifest: cases whose data comes from a socket, stdin, `rand()` or the
environment, and Windows-only cases (950 in the spatial classes, none in the temporal ones). Every selected case is
self-contained.

The unit is a Juliet test case (SARD's unit). PoisonCap (arXiv 2605.13210, §5.4) counts differently: C and C++,
cases after Juliet's own Linux Makefile filter (no `w32`, no `wchar_t`), one bad and one good image each. Its
CWE-415 figure, 1,636 tests, is 818 such cases x 2. Only its published numbers are cited here; it is not run.

The protection under test is the profile as shipped: the compiler's stack bounds (SHRINK) and the SDK's mallocng,
which gives each object its own lifetime with CDERIVE and ends it with CREVOKE (`capstone/runtime/virtual/heap.c`).
The measured images are Juliet's sources unchanged. Two kinds of derived images change one statement each and are
built separately: seeded images (main's `srand(time(NULL))` replaced by a fixed seed) and twins (one print added at the
statement that decides whether a defect happens); both are described below.

## How a reading is made

- **Batches** (`run.py`, `tools/virtualvm.py`): one boot each, the shared QEMU lock, 32 cases, a case's good image
  before its bad image. Each boot runs, in this order: `ok`, both corpus controls, the guest `rand()` check (seeded and
  twin builds), the images, both controls again, `ok` again, and the module's `--stats`.
- **A batch counts** (`judge.py`) only if the boot completed, its steps are exactly the sequence derived from the
  build, `ok` completed at both ends, and both controls read as they must at both ends: `bounds-malloc` prints its
  mark, faults with cause 28 in `control_write_probe` and exits on SIGSEGV without RETURNing; `uaf-malloc` the same
  with cause 25 in `control_read_probe`; neither times out. The `rand()` check, where present, must exit 0 with the
  four expected seed patterns. Every observation must name an image the build holds, with the build's hash and
  metadata, and every built image must be observed exactly once. All runs, the probe and the build must share one
  platform and one compiler.
- **Timeout**: every program runs under `juliet-timeout` (static, the guest's cross toolchain), which kills the whole
  process group at 10 s, reaps it and prints a marker. The guest's busybox has no `timeout`.
- **Verdicts**: a good image is CORRECT when it printed `Finished good()` and exited 0; a fault there is a
  FALSE-POSITIVE? to examine. A bad image is CAUGHT only when it printed `Calling bad()` before a domain fault with the
  class's cause (25 for 415/416, 28 for 121-127), did not print `Finished bad()`, ended on SIGSEGV, the strict
  check below passes, and the case's good image (the same flow without the defect) is CORRECT; a fault with the right
  cause that the strict check does not tie to the defective object is NO-READING:unattributed, a caught bad image
  whose good image is not CORRECT is NO-READING:good-image-not-correct. A bad image is MISSED when it printed
  `Finished bad()` and exited 0 without a fault. Anything else is NO-READING with a reason.
- **Strict check**, from QEMU's own line about the faulting access, against the objects the case's bad section
  declares or allocates (the good object and copy sources, `source`/`src`, are excluded): temporal faults are on a
  capability QEMU reports as revoked, whose length is such an object's, with the cursor inside it; spatial faults are
  outside the capability in the class's direction (past the end for 121/122/126, below the start for 124/127), on a
  capability whose length is such an object's (STRICT), or a declared stack object's size rounded up to 16 bytes
  (STRICT-ROUNDED16: a stack slot; the bytes between the object's end and the rounded bound were not protected).
  The check attributes by SIZE, so a fault on any candidate of a matching size passes. The suffix counts the candidates'
  distinct NAMES: `:unique` when there is one, `:ambiguous-<k>` when there are k. One name declared in several places
  counts once (e.g. flow variant 12's `buffer[10]` in both branches of `bad()`, only one of which runs), so a
  `:unique` row can still have more than one object of that name and size.
- **rand-dependent cases** (flow variant 12): the bad path runs only when `rand()` says so. `build.py --seeded`
  replaces main's seed with a fixed one, four seeds per case covering every pattern of the first two `rand()%2`
  results under musl 1.2.5 (`common.SEEDS`). CAUGHT if any seed's image is caught in a batch that counts; MISSED only
  if exactly one image per pattern was read and all four completed.
- **MISSED cases are relabelled only on evidence that holds for the measured execution** (`common.FAMILIES`):
  - `sizeof_double/int64_t/struct` (CWE-122): `malloc(sizeof(pointer))` then one element stored. `probe.py` reads, in
    the guest, `sizeof(T*) = 16 >= sizeof(T) = 8` for all three, so the allocation holds the element:
    NOT-TRIGGERED:sizeof-ptr.
  - wide-character `snprintf` (CWE-121/122, both `SNPRINTF(data, 100, L"%s", source)` and
    `SNPRINTF(dest, wcslen(data), L"%s", data)`): the case's own twin prints `wcslen` of the destination right after the
    copy. `%s` takes a narrow string, so swprintf copies one character; a twin reading `wcslen=1` relabels the case
    NOT-TRIGGERED:wide-snprintf. What swprintf writes does not depend on the run, so the twin's reading holds for the
    measured image.
  - CWE-170 (`char` and `wchar_t`, CWE-126): the twin prints `dest[99]` before the unterminated buffer is printed. That
    is an uninitialised stack word of the twin's execution, which the measured execution need not share, so the
    reading is reported in the `twin` column and the case stays MISSED.
- **ASan** (`asan.py`): each MISSED case's bad image is also built natively with AddressSanitizer and run (rand cases
  once per pattern, with seeds found for glibc's `rand()`); the result is reported in the `asan` column and relabels
  nothing. A native run is not the guest's run, and ASan does not see accesses made inside an uninstrumented libc (all
  of its positive-control misses are reads inside glibc's wide-character printing).
- **Intra-object** cases (`type_overrun`, an overflow inside one struct) are reported apart: the profile has no
  sub-object bounds.

## Running it

```sh
J=$(capstone/bug-corpora/juliet/fetch-juliet.sh <dir>)                      # Juliet, hash-checked
capstone/bug-corpora/juliet/build-kit.sh <work> <llvm-capstone commit>      # kit + SDK + juliet-timeout
capstone/bug-corpora/juliet/run-set.sh temporal <out> "$J" <work>/kit <work>/virtual-runtime/sdk \
    <work>/build-llvm/bin <work>/virtual-runtime/juliet-timeout
capstone/bug-corpora/juliet/run-set.sh spatial  <out> ...                   # same arguments
```

`run-set.sh` runs, per set: select, build (census and seeded), run (census and seeded), probe, a first judgement,
the twins of the MISSED cases (build and run), ASan, and the final judgement. Results land in `<out>/<set>/results/`.

Measured on an 80-thread host, one QEMU at a time, for the results below: building the kit 49 minutes (clone 18,
LLVM 7.5, QEMU 1, buildroot 22, SDK + adapter + kit under a minute); `run-set.sh temporal` 4.4 minutes end to end
(census run 3.1); `run-set.sh spatial` 94 minutes end to end (census build 2.5, census run 75, seeded run 3.7, twins
1.2, probe, ASan and judging the rest). From nothing to both sets' results: about 2.5 hours.

## Results

`results/<date>-virtual/` holds, per set, `<set>-verdicts.tsv` (one row per case: flags, the final verdict, the good
image's verdict, the fault's cause and site, the strict check, the ASan and twin readings for MISSED rows) and
`<set>-summary.json` (counts per class, each run's batches counted and identities, the platform's hashes, the probe's
readings, the ASan oracle's control rate), plus the kit's `PROVENANCE` (llvm-capstone commit, submodule commits,
compiler, `capstone-cc` and `juliet-timeout` hashes) and `SHA256SUMS`. No raw logs. `juliet-timeout` rebuilds
bit-identically from this source; the launcher and module do not (a rebuild differs in hash), so the summaries record
the hashes of the binaries that ran.

`results/2026-10-19-virtual/`, llvm-capstone 2adf45e20340 (QEMU gitlink d35af47b), every batch counted (temporal
12 + 1 seeded, spatial 286 + 17 seeded + 5 twin), every good image CORRECT (no false positive in 9,504 good images):

| CWE | cases | CAUGHT | MISSED | not triggered | intra-object |
|---|---|---|---|---|---|
| 415 double free | 228 | 228 | 0 | | |
| 416 use after free | 138 | 138 | 0 | | |
| 121 stack overflow | 3,846 | 3,630 | 0 | 144 wide-snprintf | 72 |
| 122 heap overflow | 2,314 | 2,052 | 0 | 76 wide-snprintf, 114 sizeof-ptr | 72 |
| 124 buffer underwrite | 1,098 | 1,098 | 0 | | |
| 126 buffer over-read | 782 | 722 | 60 | | |
| 127 buffer under-read | 1,098 | 1,098 | 0 | | |

Every CAUGHT row passed the strict check (temporal: 366 `STRICT:unique`; spatial: 5,257 `STRICT:unique`, 3,269
`STRICT:ambiguous-2`, 74 `STRICT-ROUNDED16:unique`). The 60 MISSED are all CWE-170 cases (unterminated copy, then
printed): their twins read `dest[99] = 0`, so in the twin's execution the copy happened to be terminated, and ASan
reports none of them natively; they stay MISSED by the rule above. The 144 intra-object cases are neither caught nor
missed: 102 end on an untagged-capability fault (cause 24, not the class's cause), 34 on a bounds fault on a 32-byte
capability the strict check does not name, and 8 seeded ones on a mix of those and MISSED. ASan's positive control
reported 189/200 (temporal) and 198/200 (spatial) sampled CAUGHT cases, and none of the 394 spatial MISSED or
not-triggered cases.

What these numbers are not: a measurement on the FPGA (this is QEMU, with exact bounds), of C++ cases, of other
optimisation levels than -O0, or of protection beyond the two allocator lifetimes and the stack bounds named above.
