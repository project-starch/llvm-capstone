# pymalloc under Sublet inside the running interpreter

*Branch `cpython/9-sublet-interpreter`, stacked on `cpython/8-reintegration`
(PR #107). Written 2026-09-25. The question: the twenty CPython pymalloc defects
are stopped 40/40 by Sublet in the component port, where the consumer is a C
model; the interpreter now runs in a domain, where the consumer is Python. This
plan joins the two halves.*

## The two halves, and why they do not currently touch

| | runs in a domain | pymalloc under Sublet | catches the twenty |
|---|---|---|---|
| `ports/cpython/interpreter` (#107) | yes, full 3.13.7 | **no** | no |
| `ports/cpython/pymalloc` | allocator only | yes | yes, 40/40 |

`git grep -il sublet` over `ports/cpython/interpreter/` is empty. The
interpreter's thirteen patches are address width, alignment and provenance;
`0009-pymalloc-arena-pointer` keeps the arena's *pointer* so pools can be
derived from it, which is provenance, not revocation. A freed block is handed
back out with no revoke, so a stale read after reuse reads the new occupant —
the defect, unabated.

## What exists to build with

`ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c` (285 lines) is
the adapter, over the shared `<sublet/sublet.h>` primitives — the same
`sublet_take`/`sublet_give`/`sublet_take_linear` the SQLite and PostgreSQL ports
use. Its seam is already the right shape, and it is what the interpreter build
has to call:

    pym_lifetime_init(region)         the metadata region
    pym_set_mode(0|1)                 spatial | sublet
    pym_arena_alloc/_free/_address/_pointer
    pym_pool_create/_pointer/_reclass
    pym_block_pointer(header, offset)
    pym_issue(ptr, requested)         hand-out: mrev + delin
    pym_release(ptr)                  free: revoke
    pym_resize / pym_requested / pym_validate
    pym_user_raw_malloc/_free/_realloc   the >512-byte fallback

`patches/cpython-3.13.7-0003-block-lifetime-hooks.patch` already routes
upstream obmalloc through those calls, in 13 hunks against
`Objects/obmalloc.c` and `Include/internal/pycore_obmalloc.h`.

## Step 1 — reconcile 0003 with the interpreter's 0009, as a new patch

0003 cannot be applied on top of the interpreter port: **both patches rewrite the
same lines.** 0009 replaces `arenaobj->address = (uintptr_t)address` and
`pool_address` with an `arena_ptr` field read through `_Py_ARENA_PTR()`; 0003
replaces the same statements with `pym_arena_address(address)` and the adapter's
arena token. They are two answers to the same provenance problem.

0003 also `Depends-On` 0002 (capability provenance), which depends on 0001, the
*extraction* patch — it compiles obmalloc.c standalone behind
`pymalloc-compat.h` and cuts interpreter statistics, the debug allocator and
lifecycle code. The interpreter build must not have that.

### Corrected 2026-09-25, after reading the adapter: 0014 REPLACES 0009, it does not sit on it

This plan first said "keep 0009's provenance mechanism, add only the lifetime
transitions". That is wrong, and the adapter says why in its own comment.
`pym_arena_alloc` returns `a`, the adapter's `struct arena`, with:

    /* An opaque token, not an alias spanning the arena's children. */

The arena's capability stays in the adapter's slot and every pool is carved out
of it; no capability spans the whole arena while its children are out. That is
what makes per-block revocation mean anything — an arena-wide alias would still
reach a revoked block's memory.

0009 keeps exactly such an alias: `arenaobj->arena_ptr = (pymem_block *)address`,
read through `_Py_ARENA_PTR()`, with `pool_address` derived from it as a pointer.
The two models are mutually exclusive inside `obmalloc.c`.

So **0014 supersedes 0009 there**: `arenaobj->address` comes from
`pym_arena_address(token)`, `pool_address` is an address again, and every pool
and block pointer is handed out by `pym_pool_create` / `pym_block_pointer` —
address in, freshly derived capability out, which is why the adapter needs only
addresses as keys.

*Corrected 2026-09-25, after the twenty ran.* This paragraph went on to say that 0009
"keeps" everything outside obmalloc's arena derivation: the alignment builtins in
`pymacro.h`, `_PyMem_FreeDelayed` and the statistics walks. It cannot keep them. The
build skips 0009 whole in the Sublet arm, so that arm had none of them. Two things
followed from that:

- every interpreter exit faulted in the first statistics walk, which cast a pool
  address to `poolp` (`_PyInterpreterState_FinalizeAllocatedBlocks`, cases 4, 7, 13,
  17 and 18);
- stringlib's fastsearch would read through an address on a one-character `find` in
  a UCS-2 string.

0014 now carries 0009's `pymacro.h` and delayed-free hunks verbatim, with the two
walks written in the adapter's vocabulary (`pym_pool_pointer`).
`prepare-cpython-capstone.sh` refuses to build either arm once the two copies of the
`pymacro.h` hunk differ.

**The build therefore selects, it does not stack:** the plain interpreter applies
0001-0013 as today; the Sublet interpreter applies 0001-0008 and 0010-0013 with
**0014 in place of 0009**. Both stay coherent, and the plain port's recorded
boots are untouched.

0003 remains the map of where the transition sites are. In the tree the port
produces they are `new_arena` (1825), `pymalloc_pool_extend` (2040),
`allocate_from_new_pool` (2062, pool carve at 2134), `pymalloc_alloc` (2198),
`pymalloc_free` (2470) and `pymalloc_realloc` (2554).

One thing 0014 does **not** need a patch for: the arena allocator itself.
`PyObject_SetArenaAllocator()` can install `pym_arena_alloc`/`pym_arena_free`
from `domain_entry.c` at startup, so the hook does not have to be cut into
obmalloc at all.

## Step 2 — link and initialise the adapter in the domain image

`prepare-cpython-capstone.sh` and `toolchain/capstone-cc` build the image;
`toolchain/domain_entry.c` is where `pym_lifetime_init()` and `pym_set_mode()`
must run, before the first allocation of the interpreter's own startup. The
adapter needs a metadata region of its own; the domain already takes a 128 MiB
CMA block (`run-cpython-domain.sh`), so the region is a partition of what the
run script already asks for, not a new monitor feature.

## Step 3 — run the thirteen Python cases, spatial against sublet

`bug-corpora/cpython/pymalloc-repros/interpreter/` holds one Python script per
defect and the native measurement: **14 of 20 reproduce a heap-use-after-free on
a native 3.13.7**, and **13 are candidates in a domain** — the others need
`_bz2`/`_lzma`/`zlib`, `_ssl`, `_curses`, or a `fork`/`exec` a domain has not
got. `run-cpython-domain.sh` takes the script as an argument and batches several
runs per boot (`CPY_DOM`, `CPY_ENV`), which is what a thirteen-case pair needs:
a boot is a minute and a run about ten seconds.

The pair is the same shape as the component port's: spatial must complete,
sublet must fault at the stale access. Note what the interpreter arm does NOT
inherit — the C corpus publishes a labelled probe address and the runner checks
the fault PC against it. A Python-level trigger has no such probe; the oracle
has to be built, and "it faulted somewhere" is not the same claim.

## What the first boots said (2026-09-25)

The arm builds, links and runs Python. Three faults, and how each was placed.

**Placing a fault at all.** The loader puts ELF vaddr `0x10000` at the load base,
so the ELF address is `pc - base + 0x10000`. Taking it as `pc - base` named the
wrong function in the first write-up (retracted). The check that settles it is the
**instruction encoding**: the fault line gives the register and offset, and the
disassembly at the computed address must match. Both attributions below were
confirmed that way, and any further one must be.

| arm | script | where | what it is |
|---|---|---|---|
| spatial | hello.py | `_PyInterpreterState_FinalizeAllocatedBlocks +0xfc`, `lwu t1, 0(t1)` | the shutdown walk over arenas and pools, striding an address and loading through it. `t1` held a bare integer. A strided integer walk has no authority where each pool body is its own capability — and it is the one place hello.py fails, after running its whole body. It is the walk 0009 rewrites and 0014 did not carry (corrected under Step 1) |
| spatial | case 0 | adapter code 516, `pym_validate` | the pointer CPython hands `_PyObject_Realloc` is not bit-identical to the lease the adapter issued. A SELF-CHECK of the adapter, which the component port's replay never trips because it only reallocs its own leases |
| sublet | case 0 | `_PyEval_Vector +0x64`, `ldc a6, 0(a6)` then `lw a7, 0(a6)`, `addiw a7, a7, 1` | a capability loaded out of an array, then a refcount increment through it. It came back UNTAGGED, and a REV handle in the same dump covers the faulting address: an object whose block was REVOKED, on the STARTUP path, before the case's own marker |

A stale object pointer at an `INCREF` is the shape the corpus looks for — but on
the startup path the interpreter is not supposed to have a use-after-free, so the
suspicion is the adapter revoking a block the interpreter still holds.

**First hypothesis, from the primitives rather than the run.** `sublet_carve`
gives the carved block the region's ORIGINAL node and leaves the remainder under
a FRESH one, and revoke walks the junior run of nodes. Carving front to back that
way, freeing an early block in a pool could revoke the blocks carved after it.
The component port's security suite claims to cover this ("live siblings"), so
either the carve order is handled and the cause is elsewhere, or that case does
not reach the shape an interpreter produces. Settle it with a directed probe over
the adapter, not another interpreter boot: a boot costs minutes and answers one
bit.

**And the emulator's node pool.** `CAP_REV_TREE_SIZE` is 65536 and this emulator
reclaims none, which a real interpreter under per-block revocation exhausts:
QEMU asserted in `_cap_rev_tree_dup_node_before`. `CAPSTONE_REV_NODES=8388608`
clears it. The `REV-NODE WATERMARK -- cumulative allocation #1022` line is the
silicon half of the same fact and is not fixed by an environment variable.

## Measured constraints to design against

From `block-lifetimes.c` and the port's README, read 2026-09-25:

- **`ARENA_COUNT 32` × `ARENA_SIZE` 1 MiB = a 32 MiB ceiling** on small-object
  space, `POOL_SIZE` 16 KiB, `POOL_COUNT` 64 per arena, `LARGE_COUNT` 4096 raw
  records. The interpreter's recorded boots run with a 48 MiB heap arena, and
  whether 32 MiB of pymalloc arenas carries even startup is **not measured**.
  Raising the constants is cheap; knowing the real high-water mark is not, and
  that measurement should come before the constants are guessed at.
- **`arena_for()` is a linear scan over the 32 arenas**, and `pool_for()` calls
  it. That sits on the path of every free. Upstream uses a radix tree for
  exactly this lookup. It is a cost, not a correctness problem, but it is on the
  hot path of a real workload rather than of a bounded replay.
- **One revocation node per live block, against QEMU's tag map of 2^20
  capabilities** (`docs/plans/cpython-interpreter-port.md`). 32 MiB of 16-byte
  blocks is 2M, so the tag map can be reached before the arena ceiling is. Also
  unmeasured.
- **Three layers, and only the middle one is ported.** Ten per-type free lists
  sit above pymalloc, so a freed object may never reach the allocator. In the C
  corpus the model consumer calls `pym_free` directly and the question does not
  arise; through the interpreter it decides per case whether Sublet can see the
  defect at all. `docs/ref/cpython-pymalloc-defects.md` calls its own layer
  column a proxy and flags case 2 as likely free-list.

## Order, and what is already true

1. ~~Which defects have a Python-level trigger at all.~~ Measured: 14 of 20 natively. All 20 were run
   in the domain, because case 16 turned out to be reachable too.
2. ~~Patch 0014.~~ It compiles, links and runs. On 2026-09-25 it also gained 0009's remainder: the
   alignment builtins, delayed free, and the statistics walks in the adapter's form.
3. ~~The pair over the cases.~~ It is plain against sublet, not spatial against sublet, because the
   plain port is the natural control. It needed C-66 first: the compiler trap that ended six
   control arms.
   - Result: `bug-corpora/cpython/pymalloc-repros/interpreter/results/20260925-qemu-interpreter/`.
   - Sublet catches all 14 native use-after-frees at their use, and 11 of them are caught only by
     Sublet. The pass rule is a halt after `CPY-CASE-ARMED`, in the defect's own function.
4. **Open:** per case, which allocator layer the freed object actually came from.
5. **Open:** `_elementtree.c`'s JOIN flag strips the tag from `text`/`tail`. The negative control
   found it in both arms. It needs its own port patch, and then the control's remaining modules have
   to be rerun.
