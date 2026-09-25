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

So the deliverable is a **new interpreter patch 0014**, not a port of 0003:
keep 0009's provenance mechanism, add only the lifetime transitions — issue at
every hand-out, release at every free, reissue at every successful realloc
including in place — at the sites 0003 identifies. 0003 is the map of where
those sites are; `_Py_ARENA_PTR` stays the way a pointer into the arena is
derived.

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

1. ~~Which defects have a Python-level trigger at all~~ — measured: 14 of 20,
   13 of them runnable in our domain build.
2. Patch 0014 (step 1) and the adapter linked in (step 2), then boot `hello.py`:
   **does the interpreter still start with a revoke on every pymalloc free?**
   Nothing else matters until it does, and the answer also settles the arena
   ceiling and the tag-map question above.
3. Spatial/sublet pair over the thirteen, batched per boot, with an oracle that
   is more than "it faulted".
4. Per case, which allocator layer the freed object actually came from — the
   proxy in `case.json` turned into a measurement.

**The toolchain is the gate.** This host has no built Capstone LLVM
(`find` over `/home`, `/tmp`, `/opt` to depth 8 finds neither `clang` nor
`ld.lld`), so step 2 cannot start until one is built. A Release+Asserts build
is under way at `llvm-capstone-compiler` (`a378789289cd` — the revision the
port's own survey and boots used), configured with the `Capstone` target.
