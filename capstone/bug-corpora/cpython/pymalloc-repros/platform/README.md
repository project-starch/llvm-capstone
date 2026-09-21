# The platform fix the CheriBSD target needs

The published PoisonCap platform poisons a block on `free` and **never takes the
poison off again**, so the next allocation that lands in that block traps on its
first write. Any program that frees and then reallocates the same size class
dies; this corpus dies before running a single case.

    platform/apply-and-build.sh /tmp/capstone/poisoncap-work
    platform/apply-and-build.sh /tmp/capstone/poisoncap-work --revert

Measured at **88 seconds** on a warm build tree: only `mrs.c` recompiles, libc
relinks, and the image is rebuilt. A platform reconstructed from scratch pays
its own build first. The published libc, image and `mrs.c` are saved under
`WORK/published-platform/` with their hashes before anything is touched, and
`--revert` puts them back, so both platforms stay available.

The rebuilt libc is bit-identical to the one these measurements were taken on
(`6726fdb0...`). The disk image is not: images are not reproducible byte for
byte, so its hash differs between builds while its contents do not.

## What is wrong

`lib/libc/stdlib/malloc/mrs/mrs.c` is the platform's PoisonCap shim. In the
published configuration `POISON_ON_FREE`, `CLEAR_ON_ALLOC_PVER_INIT` and
`PTRAP_ENABLE` are on, while `CLEAR_ON_ALLOC`, `CLEAR_ON_ALLOC_PVER` and
`CLEAR_ON_ALLOC_NWZ` are all off. So:

- `free` poisons the block -- correct, that is the protection;
- allocation calls `cclearpoisonperm()`, whose name suggests otherwise but
  which is `cheri_andperm(a, ~(1 << 12))`: it strips a permission bit from the
  returned **capability** and does not touch the memory's poison state;
- `CLEAR_ON_ALLOC` would only `memset`, and poison is out-of-band metadata --
  zeroing the payload does not remove it, and the write would itself trap;
- the two mechanisms that *would* retire the poison are compiled out:
  `CLEAR_ON_ALLOC_NWZ` (a `cclearpoison` loop) and `CLEAR_ON_ALLOC_PVER`
  (bump the version so old poison no longer matches);
- `CLEAR_ON_ALLOC_PVER_INIT`, the half that *is* on, sets the version back to
  **0** -- the same value the stale poison carries, so they match and the trap
  fires on the new owner's first write.

The binary confirms it: the published `libc.so.7` contains **2 `cpoison` and 0
`cclearpoison`** instructions. Nothing in it ever un-poisons anything.

## How it shows up here

`pym_lifetime_init` calls `feature_present("cheri_caprevoke_poison")`, which is
`asprintf`, which is `malloc(128)` (`lib/libc/stdio/vasprintf.c:52`). The kernel
reports the fault at a 128-byte object, matching exactly:

    poison exception faulting base 40a0e000, size 128 rw1
    pid 141 (program), jid 0, uid 0: exited on signal 34

A probe with no corpus code at all -- the three large allocations and then
`feature_present` -- reproduces it; the same call **before** any allocation does
not. The platform's own test programs survive because they call it first, on an
untouched heap.

## The fix

`mrs-poison-retire.patch` adds `mrs_retire_poison()` and calls it in
`mrs_malloc`, `mrs_calloc`, `mrs_posix_memalign` and `mrs_aligned_alloc`, before
anything writes into the block. `mrs_mallocx` and `mrs_realloc` delegate to
those and are covered.

## What it was measured to do

On the patched platform, with libc revocation **on**, the complete suite passes
**40/40** in one boot. The platform's own mechanism is still intact, which is
the control that matters: a direct `malloc` use-after-free is still caught
(tag 1 -> 0, `SIGPROT` at the labelled probe), and a sub-allocator's stale
access still is not. Had the patch merely disabled the protection, both would
have stopped faulting.

## Scope

This patches a research artifact's libc, so a result taken on the patched
platform is "PoisonCap with this fix" and must say so. Platform hashes differ
between the two, and `verdicts.json` records them, so the two are told apart by
their own artifacts rather than by memory. The fix has not been reported
upstream.

The platform is shared with the FFmpeg PoisonCap port. It lives here because
this corpus is what needed it; if a second consumer appears it belongs one level
up.
