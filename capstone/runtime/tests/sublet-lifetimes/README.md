# C Sublet wrappers

Run `run.sh` after sourcing `capstone/tests/capstone-test-env.sh` and setting
`CAPSTONE_LLVM_BIN` to an installed Capstone compiler. The QEMU submodule must
contain the matching Sublet instructions and be built. No compiler source
changes or new assembler mnemonics are required.

The gate compiles the public `capstone_cap_derive`, `capstone_cap_release`
and `capstone_cap_renew` wrappers with
`-O1` and `-O2`, then executes their emitted instructions on flat and paged
node tables. The small test has no globals, external dependencies or
relocations; the runner refuses a relocated object before embedding its code
in the qualified bare-metal bootstrap. `-capstone-gp-free` selects the
existing virtual calling convention.

Twenty-eight checks (seven variants, `-O1`/`-O2`, flat/paged) cover a nonzero
base-relative offset, inherited data rights and no other right on the child,
a release through the object alone, fresh allocation over the same bytes
without zeroing, a renewal that keeps bytes, bounds and place, and five
failures, each at its exact fault PC with its exact node count:
- an overflowing offset is refused at the zero-size `CDERIVE` (29);
- a linear parent is refused at that same instruction (26), before the `CSSCC`
  that would move it out of its register;
- `capstone_cap_revoke_child`, the two-operand `CREVOKE` of the capstone-sublet
  bitstream, is illegal on this model, at the `CREVOKE` (2);
- the child, released through an offset copy, faults when dereferenced (25);
- the child, after a renewal, faults when dereferenced (25).

A renewal takes no identity on the flat table, whose ids carry a generation,
and one on the paged table, so the two layouts expect different node counts.

Besides the compiler it needs host `clang` with `ld.lld` on `PATH` and `rg`.
Case 36 of the QEMU gate's numbering is reserved for this test's success case.

The QEMU submodule's `tests/sublet-lifetimes` gate covers the instruction-level edge
cases; `tests/virtual-capstone-model` covers the independent lifetime model.

The wrappers are the allocator-facing interface. mallocng reaches the same
instructions through `heap_sublet.h`. This test does not requalify any
allocator port or its benchmarks.
