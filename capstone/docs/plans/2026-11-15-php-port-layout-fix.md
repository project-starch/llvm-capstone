# PHP 5.0.0 lifetime port: stock layout and stock realloc, pre-registration, 2026-11-15

Committed before any measured run of the fixed port. A reading that differs from what is written
here is a finding; this file is not edited afterwards.

## The problem

Run-5 (`2026-11-15-php-reuse-distance.md`, results in
`ports/php/app/results/2026-11-15-reuse-distance/`) measured PHP's Zend allocator on bench.php
at the pinned one-tenth setting. Stock against protected: zend-small 0.567 % of reuses in another
bucket, zend-large 61.2 %, noise floor 0 at both. The exact-bounds arm measured the same as the
protected arm. strcat took 12 s on stock and 276 s protected.

The causes lie in the nested layer `ports/php/zend-alloc/sublet/zend-cap-mode.c`, not in PHP and
not in the processor design:

1. **Block size.** `zend_cap_block` does not pass PHP's request (header plus `REAL_SIZE(n)`) to
   malloc. It rounds the request to a 16-byte grain, takes at least 64 bytes, adds one
   pointer-sized prefix (16 bytes on this target) below the PHP header, and adds alignment slack
   for payloads of 4 KiB or more. The prefix holds the block's malloc reference (malloc's child
   with malloc's lifetime). `_efree` and `_erealloc` read it back through the mapping reference
   that `heap_access` returns, because the caller's child covers the payload only and cannot reach
   the header. The prefix lies outside the caller's bounds, so it does not leak authority, but it
   and the rounding change every malloc request: emalloc(8) asks for 80 bytes instead of 56.
2. **Realloc.** `zend_cap_reblock` never calls realloc. It takes a fresh block, copies, and frees
   the old one, so every `_erealloc` moves the object. Stock `_erealloc` calls realloc, which keeps
   the block in place whenever mallocng can.

## The change

The header stays PHP's own 48-byte `zend_mem_header` (pNext and pLast at 16 bytes each, size and
cached, padding to 48), as in stock. The prefix goes.

**Block table.** zend-cap-mode.c keeps a static open-addressing table in its own .bss (2^18 slots
of one reference each, 4 MiB, linear probing with backward-shift deletion, no tombstones). Each
slot holds the block reference that malloc returned (the cursor at the PHP header), keyed by the
header address. `zend_cap_block` enters the block. `_efree` and `_erealloc` find it by the
caller's payload address minus 48, after the caller's own reference passes a liveness probe and a
start-of-object check. The table never allocates and the layer no longer reads anything through
the mapping reference: its authority over a block comes from its own table, outside every client
bound. A block returned to malloc leaves the table. In mode 0 (the corpora's stock control, which
leaks every freed block by design) a leaked block stays in the table, so a stale free on that arm
behaves as before. A full table fails by name (`ZEND_CAP_FAIL code=6`).

**Block geometry.** Let `bytes` be PHP's request (48 plus `REAL_SIZE(n)`) and
`B = heap_round(max(REAL_SIZE(n), 1))`, the widest child bound any arm issues (mode 6 bounds the
child to B, mode 7 to `heap_round(n)`, never more). When B has a granule of at most 16 bytes
(B below 16 KiB), the payload at header plus 48 is exactly representable, and the layer passes
`bytes` to malloc unchanged. One exception: emalloc(0) asks for 48 bytes, which leaves no room for
a one-byte child, so the layer asks for 49, which mallocng serves from the same size class.
When B is 16 KiB or more, the 27-bit bounds format needs a payload base aligned to 32 bytes or
more, and mallocng aligns the block start, so a payload 48 bytes into the block can never carry
exact bounds. These blocks get the smallest slack in front of the header that aligns the payload
(16 bytes for B below 64 KiB, then granule minus 48) and the request becomes slack plus 48 plus B.
This is the one place where the stock layout and exact bounds conflict on this format.

**Realloc.** In every arm that returns blocks to malloc (modes 4, 6 and 7) `zend_cap_reblock`
calls realloc on the block's malloc reference with the request from the geometry above, so a
block below 16 KiB gets exactly the stock request. When realloc keeps the block, the address
stays. When the slack changes (a block crossing 64 KiB, or crossing 16 KiB), the layer moves the
header and payload inside the block (before realloc when the slack shrinks, after it when it
grows; a failed realloc leaves the old block as it was). `_erealloc` then derives the new child at
the returned address. mallocng's realloc ends the old block lifetime in place or on a move, and
with it the old child, so a load or store through the old reference faults with cause 25. Mode 0
keeps the fresh block, copy and leak, since it never returns a block.

The exact-bounds variant (mode 7) runs the same code with the same geometry; only the child bound
differs. The patch's seven substitutions in zend_alloc.c stay as they are; its comment, the
`.classes` text and zend-cap-mode.{h,c} describe the new behavior. The probes
(`probe-nested.c`, `probe-parent-prefix.c`) follow the new counters.

## Checks

**Unit test** (`ports/php/zend-alloc/sublet/probe-layout.c`, driver `test-probe-layout.sh`): links
zend-cap-mode.c with malloc, realloc and free wrapped at link time. It builds twice in one boot:
against the new zend-cap-mode.c and, as the negative control, against the file at 5060011844e.
- geometry: for every n from 0 to 16,376 in steps of 8 and for a list of larger sizes, the request
  malloc receives equals 48 + `REAL_SIZE(n)` for every n > 0 with B below 16 KiB. New: all equal.
  Old: they differ (the prefix and rounding).
- realloc: a block grown in small steps; for each step the layer calls realloc with the stock
  request and keeps the address whenever realloc keeps it. New: realloc called, some steps in
  place. Old: no realloc call, every step moves.
- stale access after an in-place realloc faults with cause 25 (both builds); stale access after
  efree faults with cause 25; a read one past the rounded bound faults with cause 28; a block of
  20,000 bytes grown to 40,000 and 70,000 and shrunk back keeps its contents across the slack
  changes.

**Static layout.** A tiny program built with the realphp flags prints sizeof(zend_mem_header),
MEM_HEADER_PADDING, offsetof(pNext) and offsetof(pLast) for the stock and the nested build. Both
print 48, 0, 0, 16. The build asserts sizeof(zend_mem_header) + MEM_HEADER_PADDING == 48.

**Dynamic layout.** A trace build of both arms (`-DZEND_CAP_LAYOUT_TRACE`, test only; stock
counts at `ZEND_DO_MALLOC/REALLOC/FREE`, protected at the layer's own malloc, realloc and free)
runs the run-5 bench.php once per arm and prints the number of malloc, realloc and free calls, a
histogram of request sizes, realloc outcomes (in place, moved) and an FNV-1a hash over the
sequence of requests, all separately for requests below and at or above 48 + 16 KiB.

## Measurement

Exactly the run-5 workload and settings: bench.php of PHP 5.1.0 at the pinned one-tenth setting
(SHA-256 6b9fc8f9...), `php -n bench.php`, the reuse hooks of `reuse-hooks/php/` unchanged, one
boot per run, kit-run-3 (Image-0133b9e21cd, qemu-1b, f-keep). Protected (PHP_CAP_MODE=6) and exact
(PHP_CAP_MODE=7) two repetitions each, from one image built by `reuse-hooks/php/build.sh
protected`. Stock is rebuilt with `build.sh stock`; if its hooked image is byte-identical to the
run-5 stock image (SHA-256 6ebdb090...), the run-5 stock repetitions stand, otherwise stock runs
twice as well. Results go to `~/asr-work/eval/run-6/reuse/php-<arm>-<rep>/` with
`provenance.txt`; `reuse-hooks/php/collect.py` computes the record.

## Predictions

1. **Unit test.** New build passes every stage. The old build fails geometry and realloc, and
   still faults with cause 25 on the stale accesses.
2. **Static layout.** Equal in both arms, as written above.
3. **Dynamic layout.** Below 48 + 16 KiB the two arms make the same malloc and realloc requests in
   the same order (equal counts, equal histograms, equal sequence hash) and realloc keeps or moves
   the same blocks, apart from emalloc(0) (49 against 48 bytes). At or above that size the counts
   of calls are equal and the sizes differ by the slack and the rounding of B.
4. **Reuse, zend-small.** Protected equals stock in issues, releases, reuses, distinct starts,
   cache_hits, fresh, moves and resizes and in every bucket: reusediff 0, the stock noise floor.
5. **Reuse, zend-large.** reusediff stock against protected below 1 % (61.2 % in run-5), with
   the same median bucket as stock. What differs comes from the blocks of 16 KiB or more.
6. **Exact equals protected** in every count and bucket at both boundaries.
7. **Clean runs.** Observer error 0, exit 0 and bench.php output equal apart from timings in every
   run. strcat on the protected arm within twice its stock time.
8. **Corpora.** Every corpus run on php-real-nested.dom builds with the patch, including the
   stock-control arm (mode 0). Rerun all of them on the new image with the run-4 runners and kit
   (kit-run-4): php/engine-repros asan-blind (run-4: stock 5, block 12, object 25, exact 29
   CAUGHT), php/engine-repros temporal-asan-visible (1, 5, 5, 5), php/plain-temporal-repros
   (virtual-nested-pools 7 CAUGHT) and php/zend-alloc-repros (virtual-nested-pools 20 of 20). Every
   protected arm catches at least as many cases as in run-4, and each changed verdict is explained.
