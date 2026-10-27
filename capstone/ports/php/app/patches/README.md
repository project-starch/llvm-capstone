# PHP 5.0.0 patches

Ordered, versioned changes to the release tarball, named `php-5.0.0-NNNN-<what>.patch`
like the other ports' series. Each patch opens with the reason it exists; the diff follows.

Two kinds, told apart by who applies them:

| patch | applied by | to | why |
|---|---|---|---|
| `0001-zend-modules-h-includes-zend-compile-h` | `prepare-sources.sh` | the source tree, once | the release does not compile on a current compiler (113 of 145 TUs); PHP's own 5.2.0 fix |
| `0003-zend-alloc-sublet-hooks` (+ `.classes`) | `build-virtual.sh` | a copy of `Zend/zend_alloc.c` in the build's mirror, in **seam** mode only | the allocator seam the block/object/exact arms run on; inert without `-DZEND_CAP_SUBLET`, which the `inert` build case asserts with `cmp` |

So the **tree patches** (0001) are in every image and every arm, stock included: they make
PHP compile on a current compiler and change nothing a measurement depends on. The **seam patch** (0003) is a mode: the stock arm's
`zend_alloc.c` is the release's byte for byte, the seam arms' is this patch applied to it.
`prepare-sources.sh` skips 0003 by name and `build-virtual.sh` asserts that the tree it is
given carries every tree patch (each applies in reverse) before building anything.

`0003.classes` says what each hunk of the seam patch does and why; it travels with the patch.

Adding a patch: a tree patch changes how PHP reaches memory, never what it computes, and
says in its header what measurement showed the need. A new mode gets its own number and its
own `.classes` file.
