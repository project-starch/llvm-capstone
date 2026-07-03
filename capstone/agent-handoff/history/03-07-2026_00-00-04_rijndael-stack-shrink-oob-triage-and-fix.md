# rijndael stack-shrink `-O0` regression: triage → genuine over-read (narrowing working), fixed

*Status: 2026-07-03. Resolves the single `-O0` regression that gated
stack-shrink default-on (task #77): rijndael faulted under
`-capstone-shrink-stack=true`. Triage verdict: a **genuine out-of-bounds read**
the narrowing correctly caught — a benchmark bug, not a too-tight bound. Fixed in
the rijndael build patch; both default and stack-shrink configs now pass.*

## The fault

From the 2026-07-02 default-on matrix (`-capstone-shrink-stack=true`, `-O0`):

```
[CAPSTONE] Cap mem access OOB: pc = 101561150, rs1 = x10, cursor = 10157f980,
           imm = 12, addr = 10157f98c, size = 8, bounds = (10157f980, 10157f990)
```

An **8-byte load at offset +12** of a **16-byte-bounded** stack object — reads
bytes [12, 20), i.e. 4 bytes past the 16-byte end.

## Root cause — false 32-bit `unsigned long` assumption (an LP64 bug)

`aes.h`:
```c
typedef unsigned long   word;    /* must be a 32-bit storage unit */
...
#define word_in(x)      *(word*)(x)     /* little-endian path (active on rv64) */
```
On rv64 `unsigned long` is **8 bytes**, but AES is defined over **32-bit** words
(the comment says so). `encrypt()` does `state_in(b0, in_blk, kp)` which expands to
`si(...,c): s(b0,c) = word_in(in_blk + 4*c) ^ kp[c]`, i.e.
`*(word*)(in_blk + 4*c)`. With an 8-byte `word`, `c == 3` reads
`*(unsigned long*)(in_blk + 12)` — **8 bytes at offset 12 of the 16-byte input
block** → exactly the observed `imm=12, size=8, bounds=16` fault. `rs1 = x10` is
`in_blk`. (`word_out(out_blk + 12, …)` is the symmetric 8-byte OOB *write*.)

Under broad (whole-domain) bounds these overlapping 8-byte-at-4-byte-stride
accesses happen to reconstruct the correct 16-byte output, so the benchmark's
self-check passed and the bug was invisible. Object-granularity **stack**
narrowing bounds the 16-byte block to 16 bytes and traps the over-read.

**Verdict: the narrowing is working.** This is the third capability-caught
benchmark bug of the same class in rijndael's port — cf. the documented
`char r[4]` (8-byte store, caught by `-capstone-shrink-globals`) and the
`unsigned long`==4 assumption already noted in the build patch. Not a too-tight
bound: the object genuinely is a 16-byte AES block and the code genuinely reads 8
bytes at offset 12.

## Fix

Make `word` an actual 32-bit type — the header's own stated intent. On rv64
`unsigned int` is 32-bit, so `word_in`/`word_out` become correct 4-byte accesses,
in-bounds and semantically-correct AES (state, tables, and round keys are all
32-bit words). Applied in `build-beebs-rijndael-capstone.sh` as a patched `aes.h`
placed in `$OUT_DIR` and shadowing the fetched header via a prepended
`-I"$OUT_DIR"` (with a verify-grep guard), matching the existing rijndael patch
style.

```sh
sed 's/typedef unsigned long   word;/typedef unsigned int    word;/' \
  "$RIJ_DIR/aes.h" > "$OUT_DIR/aes.h"
```

## Validation (this session, pristine QEMU)

- **Default config (committed baseline, no shrink):** rijndael **PASS** —
  `BEEBS rijndael correctness marker validated`, `__BEEBS_RIJNDAEL_PASSED__`.
  The `word` fix is baseline-safe (32-bit word is correct AES regardless of
  narrowing).
- **`-capstone-shrink-stack=true`, `-O0`:** rijndael **PASS**, same correctness
  marker, **no `Cap mem access` / no halt**. The prior fault is gone.
- The patch is isolated to rijndael's build script, so the other 81 BEEBS
  benchmarks are unaffected → the `-O0` stack-shrink BEEBS result becomes
  **82/82** (the 2026-07-02 matrix was 81/82 with rijndael the sole failure).

## Impact on task #77 (stack-shrink default-on)

Blocker **(a) rijndael `-O0`** is **resolved** (benchmark bug fixed; feature
validated as correctly catching it). Remaining before flipping the default:
- **(b) varargs save-area + dynamic-alloca** narrowing increments (not yet done).
- The `-O1/-O2/-O3` mass failures remain **pre-existing, non-stack-shrink** gaps
  (i128 ISel, fp128 materialize, `cscincoffset` assert; RV8 is 0/7 at `-O1+`
  independent of shrink) — not a clean signal, tracked separately.

Default stays `-capstone-shrink-stack=false` until (b) lands and a full clean
default-on matrix is green.
