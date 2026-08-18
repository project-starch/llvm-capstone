# Building a 2024 MicroPython as a Capstone domain

Used to measure `MPY-T02` / CVE-2024-8947 (and `MPY-T05`, the same defect) inside a
pure-capability domain rather than only on the host. The defect is fixed at our
pin, so the only way to run it in the domain is to build the fix commit's parent,
`ce491ab0d1`, with our port.

## What actually blocks it, in order

**Portability patches.** 17 of the 20 in `../../patches/` apply directly, `0003`
applies with `git apply -3`. Two do not:

- `0010-gc-range-test-scanned-words-as-integers` had to be hand-applied. Both its
  hunks are character-identical in the 2024 tree, only at different line numbers,
  so this is a context conflict rather than a semantic one. It matters: without it
  the conservative root scan compares arbitrary stack words as pointers and the
  collector faults on the first non-pointer it sees.
- `0012-stream-preserve-ioctl-pointer-arguments` was dropped. It is only needed for
  stream ioctl, which the bytearray trigger does not touch.

A third pointer comparison in `gc_get_ptr_area` was deliberately NOT changed: the
pinned tree has the same code unpatched and works, so it is not on the scan path.
Checked rather than assumed.

**`mp_int_t` moved into core.** Between 2024 and the pin, the `mp_int_t` and
`mp_uint_t` typedefs moved into `py/mpconfig.h`. Before that they were the port's
job. Our `mpconfigport.h` targets the pin and correctly omits them, which leaves
the 2024 tree with no definition at all. The symptom is a cascade of
"function cannot return function type 'mp_int_t' (aka 'int (int *)')".
`mpy2024-compat.h` supplies them, plus `SEEK_SET`/`SEEK_CUR`/`SEEK_END` which the
2024 `py/stream.c` expects and the freestanding shim does not provide.

**`py/cstack.h` did not exist.** Our `mpy_domain.c` calls
`mp_cstack_init_with_sp_here()`, an API introduced after 2024. `shim2024/py/cstack.h`
maps it onto the 2024 `mp_stack_ctrl_init()` plus `mp_stack_set_limit()` from
`py/stackctrl.h`, which the port already includes. Same two actions.

## Recipe

    git -C $CAPSTONE_TMP_ROOT/micropython worktree add /tmp/capstone/mpy-t02dom 4bed614e707c^
    # apply patches: all but 0010 and 0012 with `git apply -3`, 0010 by hand
    MPY_SRC_DIR=/tmp/capstone/mpy-t02dom \
    MPY_TESTS=all MPY_TEST_BASE_DIR=capstone-temporal MPY_TEST_INCLUDE_UNSUPPORTED=1 \
    DOM_NAME=t02dom \
    DOMAIN_EXTRA_DEFS="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES \
                       -I<this dir>/shim2024 -include <this dir>/mpy2024-compat.h" \
    bash build-micropython-silicon.sh

Note the feature profile. At CORE_FEATURES `MICROPY_PY_ARRAY_SLICE_ASSIGN` is off,
so the fix's second case, assigning to a slice from itself, is not reachable. The
domain repro reports `-1` for it rather than failing, so the primary case still
produces a verdict.

## A compiler crash found on the way, and NOT caused by this branch

Before the `mp_int_t` shim existed, the malformed tree made our clang assert:

    APInt::getZExtValue(): Assertion `getActiveBits() <= 64' failed
    #19 PointerExprEvaluator::VisitCastExpr  clang/lib/AST/ExprConstant.cpp:9832

That is the FRONTEND constant evaluator, a file `capstone-codegen-cap-constants`
does not touch, and it is reached only after 21 errors, i.e. on already-invalid
input. Valid pointer-comparison constant expressions compile cleanly on
`capstone64`. Recorded as a pre-existing, low-severity crash-on-invalid-input in
this fork: 128-bit pointers reaching a `getZExtValue()` in `ExprConstant.cpp`.
