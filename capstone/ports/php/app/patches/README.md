# PHP 5.0.0 patches

Ordered changes to the release tarball, one directory per release: `5.0.0/NNNN-<what>.patch`,
the layout of the mruby and Perl series. Each patch opens with the reason it exists; the diff
follows.

Two kinds, told apart by who applies them. Only the tree patches live here; the nested layer's
patch is kept with the adapter it hooks PHP into:

| patch | applied by | to | why |
|---|---|---|---|
| `5.0.0/0001-zend-modules-h-includes-zend-compile-h` | `prepare-sources.sh` | the source tree, once | the release does not compile on a current compiler (113 of 145 TUs); PHP's own 5.2.0 fix |
| `5.0.0/0002-zend-qsort-swap-keeps-the-capability` | `prepare-sources.sh` | the source tree, once | `_zend_qsort_swap` strips pointer tags, so every sort faults; the PostgreSQL port's 0013 rule |
| `../../zend-alloc/sublet/patches/5.0.0/0001-zend-alloc-sublet-hooks` (+ `.classes`) | `build-virtual.sh` | a copy of `Zend/zend_alloc.c` in the build's mirror, in **nested** mode only | the nested allocator layer the three protected arms run on (system allocator = virtual-malloc, nested = virtual-nested-pools, nested exact = virtual-nested-exact; the PHP-side keys php-real-block/object/exact in arms.json): seven substitutions, no added statement; the caller's pointer becomes a child of the block (`zend-cap-mode.c` -> the libc's `__capstone_malloc_derive`), and `_efree` ends it and gets the block back (`__capstone_malloc_reclaim`); inert without `-DZEND_CAP_SUBLET`, which the `inert` build case asserts with `cmp` |

So the **tree patches** (0001, 0002 here) are in every image and every arm, stock included:
they make PHP compile and make PHP's own sort work on a capability machine, and change
nothing a measurement depends on. The **nested patch** is a mode: the stock arm's
`zend_alloc.c` is the release's byte for byte, the three protected arms' (system allocator, nested, nested exact) is this patch applied to it.
`prepare-sources.sh` applies every patch of `5.0.0/` and nothing else, and `build-virtual.sh`
asserts that the tree it is given carries each of them (each applies in reverse) before
building anything. Until 2026-10-29 all three lived here as `php-5.0.0-000[123]-*.patch` and
both scripts skipped the nested patch by name; results and plans from before then cite those names.

The nested patch's `.classes` says what each of its hunks does and why; it travels with the patch.

Adding a patch: a tree patch changes how PHP reaches memory, never what it computes, and
says in its header what measurement showed the need. It goes in `5.0.0/` with the next
number. A new mode is not a tree patch: it goes with its adapter, with its own `.classes`.
