# E.1: does a toolchain from dev build the measured SPEC images byte for byte?

Programme item E.1's acceptance: "a compiler built from dev reproduces the paper's SPEC images byte for byte, or the
differences are listed and explained." The paper's Full rows come from arm FRCDKMSa3 (11 images of 8 benchmarks),
as the post-submission programme records at 384adcc2552b; +Spatial is arm FRCDKMSspa3, which this note does not
check. Neither the docs nor the paper cite the images' hashes.

Result:
- The compiler half holds: dev's compiler and linker reproduce all 11 images, the one difference being the compiler
  version string gcc embeds.
- The runtime half does not: the measured SDK was built from branch `alloc-lean`, which is not on dev, and from
  none of the eleven commits the programme listed.

## The measured build's recipe

- Compile and link: the recorded `make.out` lines of each benchmark's `build_base_capstone-lifetimes.0000`, with
  the production driver replaced by the arm's wrapper and these flags prepended to every compile line:
  `-mllvm -capstone-shrinkto-globals -mllvm -capstone-shrinkto-end-adjust=1 -mllvm -capstone-direct-calls
  -fomit-frame-pointer -mllvm -capstone-drop-redundant-shrink -mllvm -capstone-drop-redundant-shrink-declarations
  -fjump-tables -mllvm -capstone-fold-global-lo -mllvm -capstone-disable-cost-per-use -mllvm
  -capstone-memcpy-align-dispatch`. Twelve compiles in parallel, then the link lines; images stripped with
  `--strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq`.
- Compiler and linker: clang 29027126f39b.
- Wrapper: `capstone/runtime/host/capstone_vm/compiler.py` as of 5069028a65b0 (dev's has since added the post-link
  size-assumption check).
- SDK manifest codegen: `-capstone-gp-free -capstone-image-gp`, features `+f +d +movc-keeps-integer-source +relax +c
  +cap-base-rvc`, ABI lp64.
- Runtime, libc, builtins: built by `capstone/runtime/virtual/build-sdk.sh` from branch `alloc-lean` at 9950c04d384f,
  with clang 98337e20741b.
  - Options: `-DCAPSTONE_HEAP_LIFETIMES=ON -DCAPSTONE_HEAP_OBJECT_BOUNDS=ON -DCAPSTONE_HEAP_SCRUB_FREE=OFF
    -DCAPSTONE_HEAP_MAPPING_REUSE=ON -DCAPSTONE_MEMSET_CAP_ZERO=ON -DCAPSTONE_MUSL_NATIVE_IO=ON
    -DCAPSTONE_DOMAIN_FAULT_RECOVERY=OFF -DCAPSTONE_APPLICATION_ARENA_BYTES=67108864
    -DCAPSTONE_APPLICATION_CONTEXTS=15 -DCAPSTONE_APPLICATION_CONTEXT_BYTES=65536
    -DCAPSTONE_APPLICATION_DATA_BYTES=2097152 -DCAPSTONE_APPLICATION_EXCHANGE_BYTES=262144
    -DCAPSTONE_APPLICATION_GRANT_BYTES=0 -DCAPSTONE_APPLICATION_STACK_BYTES=1048576`.
  - Environment: `MUSL_CAPSTONE_EXTRA_CFLAGS=-fomit-frame-pointer`, `CAPSTONE_VIRTUAL_MOVC_KEEP` unset.
  - Its musl patches 0011 to 0013 are grow-mmapped-chunks-in-place, one-call-malloc and get-meta-inline.
  - alloc-lean sits on the native route and contains none of the eleven listed commits by hash; its memset switch is
    a cherry-pick (7166a346d516) of b68bbde23c79.

## Arm devcc: dev's compiler, the measured runtime

Compiler and linker: clang 4e3bae5d83fc with dev's lld (a3fae7063a56). Runtime, libc, builtins, link script and
wrapper are the measured arm's. Everything else is the recipe above.

- 10 of 11 images are byte-identical to the measured ones.
- 502.gcc_r differs (61968d8bf6bdc844 against 58124bcc766ceb17, same size, 48,085 bytes). Of its 387 objects, the
  one with a difference beyond `.comment` is `toplev.o`, in `.rodata.str1.1`. gcc embeds clang's `__VERSION__`,
  which carries the compiler's git hash, and the string's merge position moves the rest of the section.
- Control: both occurrences of the hash in `toplev.o` set back to 29027126f39b and the image relinked: 58124bcc766ceb17,
  the measured image. dev's linker change has no effect on these images.

## Arm devall: everything from dev

Compiler and linker as devcc. SDK built from dev 3cd698e2745f by its own build-sdk.sh, with `CAPSTONE_MEMSET_CAP_ZERO=ON`,
heap lifetimes on and scrub off, compiled by clang 4e3bae5d83fc. dev's link script and wrapper.

All 11 images differ. Archive members against the measured SDK, comparing sections other than `.comment`,
`.symtab` and `.strtab`:
- runtime: 28 against 27 members. The measured one has `native-syscall.S` (the native route), and 20 others differ.
- libc: 944 of 1,349 members differ.
- builtins: 12 of 55 members differ.
These follow from a different source branch, compiler and option set, listed above.

## What it would take

The runtime half needs alloc-lean's runtime on dev: the native route, then alloc-lean's mallocng series, the
mapping-reuse option, and the measured option set as a recorded recipe. The native route is waiting for the
lead's decision. Until then the measured images are reproducible with dev's compiler and the measured SDK, as arm
devcc shows.
