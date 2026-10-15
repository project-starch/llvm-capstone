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
(tag 1 -> 0, `SIGPROT` at the labelled probe), and a nested allocator's stale
access still is not. Had the patch merely disabled the protection, both would
have stopped faulting.

## Scope

This patches a research artifact's libc, so a result taken on the patched
platform is "PoisonCap with this fix" and must say so. Platform hashes differ
between the two, and `verdicts.json` records them, so the two are told apart by
their own artifacts rather than by memory. The fix has not been reported
upstream.

The platform recipe lives in `ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/`, whose FFmpeg
integration was removed on 2026-10-10. Its image still carries the stock CheriBSD arms of the
httpd and memcached corpora, and the PostgreSQL PoisonCap arm. This fix lives here because
this corpus is what needed it; if a second consumer appears it belongs one level up.

## A second platform defect, measured 2026-10-09: posix_memalign's status under quarantine

With runtime revocation ON (the mrs shim quarantining), `posix_memalign` sets `*ptr` to a usable
block and returns a NON-ZERO status -- in every call measured the status equals the low 32 bits of
the returned pointer (`rc=1084289024` for `p=0x40a0f000`). `aligned_alloc` and `malloc` are correct,
and with revocation OFF `posix_memalign` returns 0. A probe of nine alignment/size pairs
(`posix_memalign` and `aligned_alloc` at 16, 32 and 64 bytes, sizes 48, 128 and 4096) read the same
in one boot of each mode, on the platform rebuilt with the fix above.

Consequence: a caller that tests the status, as FFmpeg's `av_malloc` does, reads every aligned
allocation as a failure -- `ffmpeg/plane-repros` could not run on PoisonCap mode 1 at all until its
libavutil was built to allocate with `malloc` (`FFPURECAP_MALLOC_ONLY=1` in its
`runners/build-libavutil-purecap.sh`). Why the status carries the pointer's low bits -- the
source returns `ret`, which is 0 on that path -- was not established; the generated code was not
read. Not reported upstream; not fixed here.
