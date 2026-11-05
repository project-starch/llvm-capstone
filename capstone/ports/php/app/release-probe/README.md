# Release-destination probe

Answers one question per case: when the defect's block is released, does it stop in Zend's
per-size-class cache, or reach libc `free()`? It is measured in the allocator, so it does not
depend on what any sanitizer reported.

`zend-alloc-release-probe.patch` adds five emit points to `Zend/zend_alloc.c`: the cache branch of
`_efree` (`CACHED`), its fall-through to `ZEND_DO_FREE` (`FREED`), `_erealloc` (`REALLOC`, which
never consults the cache), the cache drain at request shutdown (`DRAINED`) and the shutdown leak
sweep (`SWEPT`). Each line carries the payload address, the size, the cache index, the class's
occupancy and the caller's return address; the first line of a log gives the runtime address of a
function in the file, so an address can be resolved under ASLR. The probe writes unbuffered to
`$ZPROBE_OUT` and writes nothing at all when that variable is unset, so the build is the stock
allocator unless a run asks for the probe.

## Three builds

    # 1. the probe on the stock allocator
    CFLAGS='-std=gnu89 -g' ./configure --disable-all --cache-file=./config.cache && make -j

    # 2. ASan as a locator: the corpus's detector patches, so every efree reaches free()
    #    (ZEND_DISABLE_MEMORY_CACHE 0 -> 1, REAL_SIZE(size) -> (size))
    CFLAGS='-O0 -g -fno-omit-frame-pointer -std=gnu89 -fcommon -DZEND_ENABLE_FAST_CACHE=0 \
            -Wno-implicit-int -Wno-implicit-function-declaration -Wno-error -Wno-format-security \
            -fsanitize=address -fno-sanitize-address-use-after-scope' \
      LDFLAGS='-fsanitize=address' ./configure --disable-all --cache-file=./config.cache \
      && make -j EXTRA_LIBS='-lcrypt -lcrypt -lresolv -lm -ldl -lnsl -lcrypt -lcrypt'

    # 3. the decisive one: stock cache ON, probe in, ASan on, in ONE binary
    #    (the patch of build 1 with the flags of build 2)

Build 3 is what settles a case. With the cache on, ASan can only report a use-after-free on a block
that actually reached libc, and the probe says at the same address in the same run what happened to
it. Build 2 supplies the address and proves the trigger reaches the use; it classifies nothing.

Two deviations this host needed: `configure`'s `AC_PROG_LEX` probe wants a `lex` that is not
installed, so the cached answers from an already-configured tree are reused (PHP 5.0.0 ships its
generated scanners); and the generated Makefile's `EXTRA_LIBS` omits `-ldl` for the sanitised link,
so it is given on the `make` line.

## Running

    python3 run-asan-locator.py        # build 2: the free site and the block, per case
    python3 run-release-probe.py       # build 3: the decisive run

Both take their paths from constants at the top. Neither commits a capture: the per-case logs are
raw output and stay out of the repo, which is why the record keeps only `matrix.tsv`.

## Controls, and the one that is not proven

A payload of 81 bytes or more is never cached, which is what
`cache_index = ((size+7)&~7)>>3 < MAX_CACHED_MEMORY` predicts from the source; a payload of 80 or
less goes both ways, which is the 256-entry per-class cache filling; a run killed by a signal still
leaves its log; with `ZPROBE_OUT` unset nothing is written. **`SWEPT` has never fired in a control
and is unproven.** Treat a case that rests on it as unmeasured until it has one.

The result is `bug-corpora/php/engine-repros/results/2026-11-02-release-destination/`.
