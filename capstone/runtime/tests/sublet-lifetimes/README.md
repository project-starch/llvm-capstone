# C Sublet wrappers

Run `run.sh` after sourcing `capstone/tests/capstone-test-env.sh` and setting
`CAPSTONE_LLVM_BIN` to an installed Capstone compiler. The QEMU submodule must
contain the matching Sublet instructions and be built. No compiler source
changes or new assembler mnemonics are required.

The gate compiles the public `capstone_cap_derive`,
`capstone_cap_revoke_child` and `capstone_cap_without_manage` wrappers with
`-O1` and `-O2`, then executes their emitted instructions on flat and paged
node tables. The small test has no globals, external dependencies or
relocations; the runner refuses a relocated object before embedding its code
in the qualified bare-metal bootstrap. `-capstone-gp-free` selects the
existing virtual calling convention.

Twenty checks (five variants, `-O1`/`-O2`, flat/paged) cover a nonzero
base-relative offset, inherited data rights, removing MANAGE, freeing through a
client copy, fresh allocation over the same bytes without zeroing, and four
failures, each at its exact fault PC with its exact node count:
- an overflowing offset is refused at the zero-size `CDERIVE` (29);
- a linear parent is refused at that same instruction (26), before the `CSSCC`
  that would move it out of its register;
- a reference without MANAGE cannot end a child, at the `CREVOKE` (27);
- the child, released through its client alias, faults when dereferenced (25).

Besides the compiler it needs host `clang` with `ld.lld` on `PATH` and `rg`.
Case 36 of the QEMU gate's numbering is reserved for this test's success case.

The QEMU submodule's `tests/sublet-lifetimes` gate covers the instruction-level edge
cases; `tests/virtual-capstone-model` covers the independent lifetime model.

The wrappers prepare the allocator-facing interface. Existing mallocng,
pymalloc and other Sublet adapters continue to use their existing lifetime
protocol until ported separately. Their benchmarks are not requalified by
this test.
