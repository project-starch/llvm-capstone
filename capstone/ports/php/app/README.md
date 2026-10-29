# PHP 5.0.0 on the virtual profile

Real PHP 5.0.0 -- PHP's own `main()`, the Zend engine, `ext/standard` and the bundled regex,
145 translation units, nothing stubbed -- built as a capability application and run as a
Linux process on the virtual profile, stock or with a nested allocator layer in `Zend/zend_alloc.c`.
`port.json` is the declaration; `patches/README.md` the tree patches, the nested layer's patch and who applies them.

What it is not: `--disable-all`, so no extension beyond `ext/standard`; the CLI SAPI only; no
dynamic loading, no threads; compiled at `-O0` so a fault attributes to a line. It is PHP from
2004, not PHP 8.

## Build and run, like the other applications

```sh
export CAPSTONE_LLVM_BUILD_DIR=/path/to/compiler-build   # built from this tree
source capstone/tests/capstone-test-env.sh
CAPSTONE_VIRTUAL_LIFETIMES=0 bash capstone/ports/common/application/build-virtual.sh php "$OUT0"
CAPSTONE_VIRTUAL_LIFETIMES=1 bash capstone/ports/common/application/build-virtual.sh php "$OUT1"
```

`OUT/image/php.dom` is the image, `OUT/sdk` the SDK it shares with the entry's other images,
`OUT/source/php-5.0.0` the tree `prepare-sources.sh` prepared (the museum tarball by SHA-256,
`CFLAGS=-std=gnu89 ./configure --disable-all`, the tree patches), with `PROVENANCE.txt`.

- `0`: `zend_alloc.c` as released, every `emalloc` block from virtual mallocng.
- `1`: the nested layer (its patch applied to `zend_alloc.c`, patch and adapter in `../zend-alloc/sublet`), its arm
  baked in as mode 6 -- a lifetime per `emalloc` object, `CDERIVE` on handout and `CREVOKE` at
  every `efree`, PHP's own bounds. `PHP_REALPHP_DEFAULT_MODE=7` adds exact bounds. The image
  prints `ZEND_CAP_MODE_DEFAULTED 6` when it takes that default; `PHP_CAP_MODE` in the
  environment still overrides it, as the measurement needs.

Run it the way every virtual image runs: `capstone-vexec php.dom -n script.php` in the guest,
or through `runtime/virtual/run.py --php OUT/image/php.dom`, which adds one sort-and-file check
to the platform gate, or this port's own `run-virtual.py` for one boot with many runs.

## The measurement instrument, separately

`build-virtual.sh realphp` on its own builds the images `run-realphp.py` measures with. Two
"stock"s are in play and are not the same thing: `PHP_REALPHP_ALLOC=stock` is the UNPATCHED
image above (no nested layer at all), the honest-gap column of the spatial pair; the arm
`php-real-stock` is the NESTED image at `PHP_CAP_MODE=0` (nothing revoked, the loan leaked).
The runner uses one nested image for all four of its arms, modes 0, 4, 6 and 7
(`tools/arms.json` `php-real-*`), named per run and never defaulted: a transcript that says
`ZEND_CAP_MODE_DEFAULTED` is refused. The corpus it is scored against is external; see
`results/2026-10-26/README.md` for the recorded result and its 2026-10-27 correction, and
`results/2026-10-27-requalification/` for what ran on today's dev.

## Patch cost, measured

To run on capabilities PHP needed one function changed, `_zend_qsort_swap`, which swapped
pointers as `int`s and `char`s and so stripped their tags (patch 0002, ~20 lines), and one
compiler flag, `-fno-common` (a COMMON symbol has no section for the backend to bound).
Patch 0001 (one include) is 2004 code against any current compiler, not capabilities. Two
configure answers and one type alias are musl against glibc. Nothing in PHP's logic changed.
The nested layer, the protection being measured, is ~100 patch lines in `zend_alloc.c` plus the
adapter.

## Layout

```
build-virtual.sh        the port's own build: probes, the Phase A/B engine image, realphp
prepare-sources.sh      the PHP tree: tarball by hash, configure, tree patches, provenance
patches/5.0.0/          the tree patches: 0001 compile fix, 0002 qsort swap (the nested layer's
                        patch, a mode, is ../zend-alloc/sublet/patches/5.0.0/)
run-virtual.py          one guest boot, many runs; the transport run-realphp.py uses
run-realphp.py          the corpus instrument: arms, populations, the judge's controls
test-run-realphp.py     its offline suite; test-run-cases.py the Phase A runner's
triggers/               the 27 Phase A trigger scripts lifted from the engine's #ifdef ladder
results/                recorded runs, each with its README
```
