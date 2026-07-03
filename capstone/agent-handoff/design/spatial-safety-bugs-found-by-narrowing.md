# Spatial-safety bugs found by object-granularity capability narrowing

*Paper artifact. Consolidates the real, pre-existing memory-safety bugs that
object-granularity capability narrowing (C1: `SHRINK` at materialization) surfaced
in **unmodified, upstream benchmark code** — bugs that whole-segment ("broad")
bounds silently allowed. This is direct evidence for the paper's spatial-safety
claim: narrowing is not just overhead, it catches bugs real programs ship with.*

## Claim

Under broad bounds (a capability covering the whole domain image), an object
over-read/over-write stays *within the domain's own capability* and does not
fault — it silently reads/clobbers adjacent objects, exactly as on a
non-capability machine. With per-object `SHRINK` narrowing, each object's
capability is bounded to `[&obj, &obj+sizeof(obj))`, so the same access **traps**
(`Cap mem access OOB`). Turning narrowing on across CoreMark / BEEBS / RV8 both
(a) kept every correct program green and (b) exposed the following genuine bugs.

Each entry is a bug in the **benchmark**, not in the compiler or the narrowing:
the object really is its declared size and the code really accesses past it.

## Findings

### 1. rijndael — 8-byte write through a 4-byte global (`char r[4]`) — caught by `-capstone-shrink-globals` (default on)

- **Code:** `aesxam.c` `fillrand()` does `*(unsigned long *)r = RAND(...)` where
  `static char r[4]`. The code assumes `sizeof(unsigned long) == 4`; on rv64 it is
  **8**, so this is an 8-byte store through a 4-byte array — 4 bytes past the end.
- **Under broad bounds:** silently clobbers the 4 adjacent `.data`/`.bss` bytes;
  only `r[0..3]` are ever read, so the RAND stream is unaffected and the benchmark
  self-check passes. Bug invisible.
- **With `-capstone-shrink-globals`:** `r`'s capability is bounded to 4 bytes; the
  8-byte store traps (`Cap mem access OOB`).
- **Fix (benchmark port):** widen `r` to `char r[8]` so the store is in-bounds
  (semantics unchanged — only `r[0..3]` are read). In
  `build-beebs-rijndael-capstone.sh`.
- **Evidence:** `design/c1-coverage-matrix-and-overhead.md` (globals row: "rijndael
  OOB found"); the patch + comment in the build script.

### 2. rijndael — 8-byte read at `in_blk+12` (`word` = `unsigned long`) — caught by `-capstone-shrink-stack` (opt-in)

- **Code:** `aes.h` has `typedef unsigned long word;` with the comment *"must be a
  32-bit storage unit"*, and `#define word_in(x) *(word*)(x)`. On rv64
  `unsigned long` is **8 bytes**. `encrypt()`'s `state_in` expands to
  `si(b0,in_blk,kp,c): *(word*)(in_blk + 4*c)`; for `c == 3` that is
  `*(unsigned long*)(in_blk + 12)` — an **8-byte load at offset 12 of the 16-byte
  AES block**, 4 bytes past the end. `word_out(out_blk+12, …)` is the symmetric
  8-byte OOB *write*.
- **Under broad bounds:** the overlapping 8-byte-at-4-byte-stride reads/writes
  happen to reconstruct the correct 16-byte output, so the benchmark self-check
  passes. Bug invisible.
- **With `-capstone-shrink-stack`:** the caller's 16-byte input block on the stack
  is bounded to 16 bytes; the 8-byte load at +12 traps
  (`Cap mem access OOB: … cursor=…f980, imm=12, size=8, bounds=(…f980,…f990)`).
- **Fix (benchmark port):** `typedef unsigned int word` (32-bit on rv64) — the
  header's stated intent and semantically-correct AES; accesses become in-bounds
  4-byte reads/writes. In `build-beebs-rijndael-capstone.sh` (patched `aes.h`
  shadowed via `-I$OUT_DIR`).
- **Evidence:**
  `history/03-07-2026_00-00-04_rijndael-stack-shrink-oob-triage-and-fix.md`.

### 3. trio (`trio-snprintf`/`trio-sscanf`) — `realloc` over-read of the old block — documented latent over-read

- **Code:** `realloc_beebs` copies from the old allocation using the *new* size,
  reading past the end of the old (smaller) block.
- **Status:** a real latent over-read that per-allocation heap narrowing **would**
  trap; it is left **un-narrowed by design** in the benchmark allocator so trio
  still runs, and the over-read is documented rather than silently fixed. This
  marks the current boundary of heap narrowing (a size-aware `realloc` is future
  work), and is itself evidence of the same bug class.
- **Evidence:** `design/granularity-provenance-discussion.md` (§ malloc / row 5;
  "`realloc_beebs` … over-reads the old block — a documented latent over-read, like
  the rijndael find").

## Why this matters for the paper

- **Two of three are in the same program (rijndael)** and are the *classic* LP64
  portability bug — 32-bit-`unsigned long` assumptions that are OOB on a 64-bit
  target. They are exactly the spatial-safety errors capabilities are meant to
  stop, and both were **latent and passing** under broad bounds.
- The bugs span **all three storage classes** the narrowing covers: a **global**
  over-write (#1), a **stack** over-read/write (#2), and a **heap** over-read (#3)
  — matching the C1 coverage matrix and showing the property is not global-only.
- The result is stated honestly: narrowing does **not** catch intra-object
  (subobject/field) overflows (see `c1-coverage-matrix-and-overhead.md` — struct
  field over-read is `no-trap-today`). The claim is *object-granularity* spatial
  safety, and these finds are its positive evidence.

## Reproduce

- ON (default globals): rijndael `r[4]` unfixed → `Cap mem access OOB` on the
  `fillrand` store. `-mllvm -capstone-shrink-globals=false` reproduces the silent
  (broad-bounds) behavior.
- Stack: build rijndael with `-mllvm -capstone-shrink-stack=true` and the `word`
  typedef unfixed → `Cap mem access OOB` at `in_blk+12`; the committed build patch
  fixes it and rijndael passes both default and `-shrink-stack`.
- Authority suite `global_oob` / `stack_oob` / `heap_oob` are the distilled
  positive controls (deliberate one-past-end accesses that must fault).
