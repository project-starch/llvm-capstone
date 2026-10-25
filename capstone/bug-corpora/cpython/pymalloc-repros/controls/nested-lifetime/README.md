# Nested-lifetime controls: parent, sibling, successor

CPython's row of `capstone/docs/plans/2026-10-11-nested-lifetime-controls.md`. Not cases and not
the corpus's `00_control_uaf_block`: these run in the **whole interpreter** with patch 0014
(`CPY_SUBLET=1`) as the protected arm, the configuration the security table measures, not in the
standalone pymalloc replay with patch 0003 that the twenty cases use. The directory name does not
match `[0-9][0-9]_*`, so `shared/build-cases.sh` and `runners/virtual/run-virtual.py` never pick
it up.

A Python script cannot hold a raw block pointer, so pymalloc is driven from C: `_pymctl.c` is a
built-in module, compiled into both arms from the same file. `build-arm.sh` is
`CPY_TEST_CAPI=1 [CPY_SUBLET=1] build-virtual.sh cpython` with one added step: after configure it
copies `_pymctl.c` into the unpacked source's `Modules/` and appends it to `Setup.local`'s
`*static*` block, the way `prepare-cpython-capstone.sh` adds the test C-API modules. Nothing
under `capstone/ports` is changed.

| entry point | sequence | protected | release |
|---|---|---|---|
| `_pymctl.parent()` | A = `PyObject_Malloc(48)`, `PyObject_Free(A)`, then read the pool's free-list head, A's free-list link and A's last byte, write that byte and read it back, through the **pool header pointer obmalloc uses** | completes | completes |
| `_pymctl.sibling()` | A and B from the same pool (same size class, same `POOL_ADDR`), free A, read all 48 bytes of B and check them, write and read back B's last byte | completes, B intact | completes, B intact |
| `_pymctl.successor()` | free A, `PyObject_Malloc(48)` until the address equals A's (LIFO: normally the first try), write and read every byte through C | C == A, completes | C == A, completes |
| `_pymctl.stale()` | `successor()`'s sequence, then one read through the old A at `pymctl_stale_read` | fault, cause 25, at the label | completes, reads C's 0xc3 |

Each entry point is its own interpreter process
(`cpython.dom -S -c "import _pymctl; _pymctl.<name>()"`), so the stale read is a separate run.
The old A is held through memory (`pymctl_held`), as the corpus's `held` is.

## The parent reference

obmalloc keeps, per arena, the capability the system allocator (mallocng) returned for the 1 MiB
arena: `allarenas[i].arena_ptr` (patch 0009, read through `_Py_ARENA_PTR`). In the protected build
patch 0014 derives `arena_life` from it with `CDERIVE`, and every client block from `arena_life`
with `CDERIVE` again, bounded to the request. `pymalloc_free` revokes the block's child through
`arena_life` (`CREVOKE`), then carries on with `arena_inside(ao, p)` =
`_Py_ARENA_PTR(ao) + (p - ao->address)` and `pool = POOL_ADDR(that)`. That pool pointer is what
writes the free-list link into the freed block.

Control 1 takes the same path: it finds A's arena by address in obmalloc's own table
(`interp->obmalloc->mgmt.arenas`), forms `_Py_ARENA_PTR(ao) + offset`, applies `POOL_ADDR`, and
reaches A's former range as the pool plus A's offset in it. So the reference is `arena_ptr`, the
grandparent of A's revoked child, not A and not `arena_life`. The release build has the same field
(0009) and takes the same path, although there A's own pointer would reach the pool too.

## Probes

Every access to the memory under test goes through a labelled probe, so a fault's pc names the
access. Lines are in `_pymctl.c`.

| probe | instruction | used by |
|---|---|---|
| `pymctl_read` (line 54) | `lbu` | parent 174, 179, 182; sibling 227, 233; successor/stale 288 |
| `pymctl_write` (line 63) | `sb` | parent 178, 180; sibling 232; successor/stale 284 |
| `pymctl_stale_read` (line 72) | `lbu`, alone in `pymctl_stale_probe` | stale 325 |
| `pymctl_link_probe` (line 82) | pointer-sized C load | parent 164, 170 (pool header), 173 (A's link) |

The parent control also reads `pool->szidx` and `pool->ref.count` directly; both are in the pool
header, not in A's range.

## Build and run (on the build host)

    build-arm.sh protected OUT/protected
    build-arm.sh release   OUT/release
    run-controls.py RESULTS --kit <kit-sublet> --llvm-bin <toolchain>/bin \
        --native OUT/release/source/build-python/python \
        --arm protected=OUT/protected --arm release=OUT/release --repeats N

`run-controls.py` refuses a kit whose QEMU, `capstone-vexec` or `capstone_vm.ko` is not the
registered one (af37cc32, 3975314a, a1c6cb6b) and a build whose manifest is not the arm's. It boots
each arm and repeat on its own staged disk through `capstone/runtime/virtual/run-staged.py` (one
hart, exact bounds) with its own QEMU lock, and writes `RESULTS/results.json`: result lines, fault
lines resolved against the image's symbols and the probe labels, and the per-control verdict
against the plan's prediction. Serial captures stay on the build host.
