# PHP 5.0.0 nested layer with the stock layout: checks, reuse and corpora (2026-11-15)

The record of `capstone/docs/plans/2026-11-15-php-port-layout-fix.md` (pre-registration
367e6a39890), run on the implementation dfe68e4a9fc, on the trap route of run-5 (Image-0133b9e21cd,
qemu-1b, f-keep; kit-run-3 for the reuse runs, kit-run-4 for the corpora as in run-4).

## What changed

The nested layer (`ports/php/zend-alloc/sublet/zend-cap-mode.c`) no longer stores malloc's block
reference in a prefix below PHP's header, no longer rounds blocks to a 16-byte grain of at least
64 bytes, and no longer moves the object on every erealloc. It keeps a static block table in its
own .bss (2^18 slots, 4 MiB, outside every client bound, no allocation), keyed by the header
address, and finds a block there from the caller's payload address after a liveness probe of the
caller's reference. malloc receives PHP's own request whenever the child bound is below 16 KiB;
emalloc(0) asks for 49 bytes; a bound of 16 KiB or more gets the smallest slack in front of the
header that the 27-bit bounds format needs. erealloc calls realloc with that request.

## Unit test (`ports/php/zend-alloc/sublet/test-probe-layout.sh`, verdict in `unit-test-verdict.txt`)

One boot, the probe built against the new nested layer and against the layer at 5060011844e:

| Stage | New layer | Old layer (negative control) |
|---|---|---|
| geometry: request = 48 + REAL_SIZE(n) for the 2,046 sizes with a bound below 16 KiB | 2,046 of 2,046 equal; emalloc(0) asks 49 | 0 of 2,046 equal; emalloc(0) asks 80 |
| realloc: 363 steps of +8 bytes from 100 to 3,000 | 363 realloc calls with PHP's request, 346 in place, 17 moved, no malloc | no realloc call, 363 moves (malloc, copy, free) |
| stale read after an in-place realloc | in place, cause 25 | (moved) cause 25 |
| stale read after efree | cause 25 | cause 25 |
| read one past the REAL_SIZE bound | cause 28 | |
| 20,000 bytes grown to 40,000, 70,000, 140,000, shrunk to 20,000 | contents kept across slack 16, 80, 208, 16 | |
| stale read after the 70,000 realloc | cause 25 | |

probe-nested (stages 0 to 12) and probe-block-table (the renamed prefix probe) behave as declared on
the new layer: positive controls complete, stale uses fault with cause 25, stage 7 and the header
under-read with cause 28, an interior free fails by name (code 4).

## Static layout (`layout-static.txt`)

Both arms, built with their realphp flags: sizeof(zend_mem_header) 48, MEM_HEADER_PADDING 0,
offsetof(pNext) 0, offsetof(pLast) 16, sizeof(void *) 16, PLATFORM_ALIGNMENT 16,
MAX_CACHED_MEMORY 11, MAX_CACHED_ENTRIES 256. The patched zend_alloc.c asserts the 48 at compile
time.

## Dynamic layout (bench.php with the layout trace, one run per arm; `layout-trace-*.json`)

Requests are split at 48 + 16 KiB, where the child bound reaches 16 KiB.

| Count | stock | stock with PHP_CAP_MODE=6 | protected |
|---|---|---|---|
| malloc below the split | 1,067,546 | 1,067,552 | 1,067,551 |
| realloc below the split (in place / moved) | 3,811 (3,378 / 433) | 3,813 (3,374 / 439) | 3,811 (3,372 / 439) |
| malloc at or above | 21 | 21 | 21 |
| realloc at or above (in place / moved) | 17,332 (17,204 / 128) | 17,332 (17,206 / 126) | 17,334 (17,208 / 126) |
| free | 1,067,236 | 1,067,242 | 1,067,241 |

Below the split the protected arm passes exactly PHP's requests: against the stock arm run with the
same environment, the size histograms differ in two places only. One 64-byte request is missing:
the protected run printed one timing (simplecall) one digit wider, which shortens the padding
string bench.php builds for that line from 9 to 8 bytes, so it no longer needs a 64-byte block
(the 8-byte string most likely came from AG(cache), which the trace does not see). Two reallocs of 16,424 bytes moved into the upper class: REAL_SIZE 16,376 rounds
to a 16,384-byte bound, whose granule is 32 bytes, so these blocks take the slack. Above the split
the calls and the moves are the same (126) and the sizes differ by the slack and the rounding of
the bound. emalloc(0) never occurs in bench.php. The sequence hashes differ, as the two differences
above require. The block table peaked at 33,941 entries (13 % of its slots).

The variable that names the protected arm allocates by itself. PHP registers PHP_CAP_MODE in $_ENV
and $_SERVER, so the stock arm run with it makes six more mallocs and two more reallocs below the
split (requests of 56, 80, 96 and 192 bytes, two each), moves six more small blocks and two fewer
large ones, than the stock arm run without it.

## Reuse (run-5 workload and settings, `reuse.json`; run directories `run-6/reuse/php-<arm>-<rep>`)

| Boundary | Arm | issues | releases | reuses | distinct starts | median | error |
|---|---|---|---|---|---|---|---|
| zend-small | stock (1, 2) | 2,410,639 | 2,410,639 | 2,374,848 | 35,791 | 2 | 0 |
| zend-small | protected (1, 2) | 2,410,646 | 2,410,646 | 2,375,076 | 35,570 | 2 | 0 |
| zend-small | exact (1, 2) | 2,410,646 | 2,410,646 | 2,375,076 | 35,570 | 2 | 0 |
| zend-large | stock (1, 2) | 1,011,177 | 1,011,177 | 963,564 | 47,613 | 7 | 0 |
| zend-large | protected (1, 2) | 1,011,184 | 1,011,184 | 962,035 | 49,149 | 7 | 0 |
| zend-large | exact (1, 2) | 1,011,184 | 1,011,184 | 962,035 | 49,149 | 7 | 0 |

Each arm's two runs are identical in every count and bucket. Every run exits 0, and bench.php's
output is the same in every run apart from its timings.

| reusediff | zend-small | zend-large |
|---|---|---|
| stock-1 against stock-2 (noise floor) | 0 | 0 |
| protected-1 against protected-2 | 0 | 0 |
| stock against protected (rep 1, and all four pairs) | 0.0147 % | 8.16 % |
| stock against exact | 0.0147 % | 8.16 % |
| run-5, stock against protected | 0.567 % | 61.2 % |
| diagnostic: stock with PHP_CAP_MODE=6 against protected (`reuse-stock-env.json`) | 0.00008 % | 0.0012 % |

The stock image is not byte-identical to run-5's (fc4bbd81 against 6ebdb090: the images carry
their build directory in the line tables, and two stock builds of one tree in two directories also
differ), so stock ran twice again.

Wall time per run (s): stock 3,512 and 3,546; protected 4,153 and 4,167; exact 4,167 and 4,149;
stock with PHP_CAP_MODE=6 2,989 and 3,082. strcat: stock 12.5 and 12.7 s, protected 16.7 and 16.7 s,
exact 16.7 and 16.7 s (run-5: 12.2 s against 276.1 s). bench.php total: stock 3,464 and 3,499 s,
protected 4,102 and 4,116 s.

## Predictions

1. Unit test: **holds.**
2. Static layout: **holds.**
3. Dynamic layout: **holds with one boundary exception and one finding.** Below 48 + 16 KiB the
   requests and realloc outcomes equal stock run with the same environment, apart from one timing
   string. The exception: REAL_SIZE 16,376 to 16,383 rounds to a 16,384-byte bound, so the slack
   starts at a PHP request of 16,424 bytes, not 16,432. The finding: the pre-registered stock arm,
   run without PHP_CAP_MODE, makes different requests, because the variable itself allocates.
4. zend-small equal to stock: **does not hold as written**: reusediff 0.0147 % against a noise
   floor of 0, with 7 more issues in the protected arm. With the same environment in both arms,
   0.00008 % (one hand-out differs, the timing string).
5. zend-large below 1 %: **does not hold as written**: 8.16 %, median bucket equal (7). With the
   same environment in both arms, 0.0012 % (the protected arm does 2 more moves and 2 fewer
   in-place reallocs, the 16,424-byte blocks). So the variable that names the arm, not the layer,
   causes almost all of the 8.16 %, and run-5's 61.2 % included the same effect.
6. Exact equals protected: **holds**, every count and bucket at both boundaries.
7. Clean runs: **holds**: error 0, exit 0, output equal apart from timings; strcat 1.34 times stock.
8. Corpora: **holds** (`corpora-vs-run4.json`). Every verdict of the default scoring equals run-4:
   php/engine-repros asan-blind stock 5, block 12, object 25, exact 29 CAUGHT of 39; temporal-asan-
   visible 1, 5, 5, 5 of 31 (together 6, 17, 30, 34 of 43 in the denominator); plain-temporal
   virtual-nested-pools 7; zend-alloc virtual-nested-pools 20 of 20. One sensitivity reading (oracle
   crashes scored) changed on the stock control arm of plain-temporal: case 22 was killed by the
   40 s per-run watchdog in run-4 and now runs to its end (output "string(36) ...", "end"), so it
   reads not-reached instead of infra; its default verdict is out-of-denominator in both runs.

## Files

- `reuse.json`, `reuse-stock-env.json`: `reuse-hooks/php/collect.py` over the run bundles.
- `layout-trace-stock-vs-protected.json`, `layout-trace-stock-env-vs-protected.json`:
  `compare-layout.py` over the trace runs (`run-6/layout/php-trace-*`).
- `layout-static.txt`: the static check (`run-6/layout/php-static`).
- `unit-test-verdict.txt`: `test-probe-layout.sh`.
- `corpora-vs-run4.json`: `compare-corpora.py` over `run-4/` and `run-6/corpora/`.
- `provenance.txt`: images, kits, run directories and wall times.
- `drivers/`: the p13 driver scripts that built the images and ran every boot (paths of p13).
