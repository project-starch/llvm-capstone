# CheriBSD purecap as a baseline for the 31 ASAN-blind PHP defects

## The question

The PHP 5.0.0 bug corpus contains 31 memory-safety defects that **unpatched ASAN
cannot see**. The set is defined mechanically, by differencing two result files
the corpus already ships:

```
detected in  results-asan-default.json      (PHP's allocator PATCHED)
missed   in  results-asan-stockalloc.json   (PHP's allocator AS SHIPPED)
```

They are invisible without the patch because ASAN interposes at `malloc`/`free`,
and PHP's `emalloc`/`efree` mostly never reach them:

| sub-class | n | why ASAN is blind |
|---|--:|---|
| cache-hidden | 23 | `efree` pushes the block onto `AG(cache)[size>>3]` and returns. `free()` is never called, so there is nothing for ASAN to poison. All CWE-416. |
| rounding-hidden | 8 | `REAL_SIZE(n) = (n+7) & ~7`, so up to 7 bytes of slack sit *inside* the malloc'd block. An over-read into the slack never reaches a redzone. CWE-125 x6, CWE-787 x1, CWE-562 x1. |

CHERI is the standard capability baseline, so the obvious question is whether it
already solves this. **The pre-registered prediction is that it does not**, and
for the same structural reason: CHERI also interposes *below* the nested
allocator. `malloc` bounds the whole block; `(char*)p + sizeof(header)` is plain
pointer arithmetic that inherits those bounds; and revocation sweeps on `free()`,
which the cache path never calls.

If that holds, the finding is that **detector blindness is a property of where
the detector interposes, not of its strength** -- and a null result is the
result. Which is exactly why every arm here carries a positive control.

## Where, not just whether

One fact about PHP's allocator shapes the whole measurement:
`shutdown_memory_manager` (`Zend/zend_alloc.c:468`) **drains the size-class cache
and `free()`s every block**. So a block `efree`d mid-request stays malloc-live
for the rest of the request and is released only at shutdown -- strictly *after*
every use site a corpus trigger exercises.

A fault at shutdown is therefore **not a catch of the defect**: the program has
already consumed the stale data. Each row prints unbuffered phase markers
(`USE-BEGIN`, `SHUT-BEGIN`, `DONE`) and the classifier reports the last one
reached, so `BLOCKED-SYNC` (caught at the access) is never confused with
`LATE@SHUT` (caught during the drain).

## Layout

| path | what |
|---|---|
| `php-alloc/php_alloc.{h,c}` | PHP 5.0.0's non-`ZEND_MM` allocator transcribed: header prefix, `REAL_SIZE` rounding, `AG(cache)` size-class free lists, the intrusive live list, and a `shutdown_memory_manager` that drains the cache. Every macro cites its `Zend/zend_alloc.c` line. |
| `php-alloc/row.h` | phase markers, a non-eliding sink, and `STOCK_REAL_SIZE` |
| `rows/` | one `.c` per defect shape, plus controls |
| `rows.tsv` | which of the 31 each row models, and the **pre-registered** prediction per arm |
| `host-validate.sh` | validate the rows against ASAN on the host -- no CHERI needed |
| `compile-purecap.sh` | build every row purecap, in both allocator arms |
| `run-in-guest.sh` | runs in CheriBSD; sets the revocation knobs, runs every row |
| `cheri-run.py` | boots the image under CHERI-QEMU, drives the three configs |
| `classify.py` | verdicts, measured against the pre-registered predictions |
| `selftest-classify.sh` | **positive control for the classifier itself** |
| `preflight-kernel.sh` | **gate: does this CheriBSD release actually support revocation?** |
| `run-php-blind.sh` | end to end |

## The two arms and the three configs

Arms differ by one `-D`, and they are the same two patches the corpus applies in
`scripts/build-php-5.0.0-variant.sh:61-65`:

| arm | allocator | role |
|---|---|---|
| `stockalloc` | cache on, `REAL_SIZE` rounding on | the real PHP allocator. Expect CHERI to be blind. |
| `patched` | `-DPHP_PATCHED_ALLOC`: cache off, `REAL_SIZE(n)=n` | **positive control.** Expect detection. |

Configs, via `sysctl`, as in the SQLite CHERI baseline:

| config | `runtime_revocation_default` | `..._every_free_default` |
|---|:--:|:--:|
| `spatial` | 0 | 0 |
| `temporal` (realistic default) | 1 | 0 |
| `eager` (expensive upper bound) | 1 | 1 |

## Controls, and why each exists

A null result is only credible if the instrument has been shown to produce the
opposite. Four independent controls:

1. **`b3_oob_read_past_block`** -- one byte past the malloc'd block. Must fault in
   every config and **both** arms. If it does not, bounds are not enforced on
   this vehicle and no `MISS` anywhere means anything.
2. **`a6_uaf_after_shutdown`** -- a genuine post-`free()` use, in the **stock**
   arm. If revocation works at all, this must be caught. A `MISS` on `a1`-`a4`
   *together with* a catch here is the precise evidence that the blindness comes
   from `free()` never being reached, not from a misconfigured image.
3. **The `patched` arm** -- the same accesses with the corpus patches applied.
   Mirrors the corpus's own `asan-stockalloc` -> `asan-default` pair.
4. **`selftest-classify.sh`** -- a synthetic log exercising every verdict branch,
   plus negative tests that an empty log is an *error* and a wrong prediction is
   *flagged*.

5. **`preflight-kernel.sh`** -- reads the dist kernel's string table and
   confirms it exports `runtime_revocation_default`,
   `runtime_revocation_every_free_default` and `runtime_revocation_async`
   *before* an image is built. Without them the `temporal` and `eager` arms
   measure nothing and every `MISS` they report is a property of the kernel, not
   of CHERI. It carries its own positive control (three sysctl leaves that exist
   in any FreeBSD kernel), because a symbol check that cannot fire is worthless.

   Measured on the release in use -- FreeBSD 15.0-CURRENT
   `releng/26.07-88f39900c329`, `CHERI-PURECAP-QEMU`: all three present.

`z0_sanity` must run clean in both arms and all three configs, or every row is
void.

### One instrument bug worth recording

The kernel check first reported all three revocation knobs **absent** when all
three were present. Cause: `strings "$K" | grep -q NAME` under `set -o pipefail`
-- `grep -q` exits on the first match, `strings` dies of SIGPIPE (141), and
pipefail reports the *pipeline* as failed, so a successful match reads as a miss.
Every such check here now greps a materialised file or a herestring, never a
pipe into an early-exiting reader. The same pattern was present in
`selftest-classify.sh` and has been removed there too.

## Scope

These rows model the 31 defects' *mechanism*; they are not the 31 triggers. That
distinction is deliberate and follows the SQLite CHERI baseline, which had to use
a lifecycle harness because upstream SQLite would not run purecap standalone at
all. Running the real triggers needs a full PHP 5.0.0 purecap build, which is
gated separately -- see the plan. CheriBSD, being a complete FreeBSD, *can* in
principle run all 31 (including the three needing loopback, an HTTP listener and
a writable `/tmp`), which is the one thing the Capstone domain cannot do.

## Run it

```bash
./host-validate.sh        # rows vs ASAN; needs only gcc + libasan
./selftest-classify.sh    # prove the classifier can produce a catch
./run-php-blind.sh        # compile purecap, rebuild image, boot, classify
```

Measured host-ASAN results, and the two row-design errors that validation
caught, are in `HOST-VALIDATION.md`.
