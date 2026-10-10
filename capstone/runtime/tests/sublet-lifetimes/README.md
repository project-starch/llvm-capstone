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

Eight checks cover a nonzero base-relative offset, inherited data rights,
removing MANAGE, freeing through a client copy, fresh allocation over the
same bytes without zeroing, and rejection of an overflowing offset. Failure
cases check the exact fault PC and that no node was allocated. The QEMU
submodule's `tests/sublet-lifetimes` gate covers the instruction-level edge
cases; `tests/virtual-capstone-model` covers the independent lifetime model.

The wrappers prepare the allocator-facing interface. Existing mallocng,
pymalloc and other Sublet adapters continue to use their existing lifetime
protocol until ported separately. Their benchmarks are not requalified by
this test.
