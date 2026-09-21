# Wiring: what a level's `.inc` has to say

One file per level, named `<level-id>.inc`, included into the allocator's own
translation unit. The build system does not change, which is the reason the
existing Apache wiring is an include and not a source of its own.

The counting is not here. It is `a1core.inc` in the manuscript's
`experiments/a1/hook`, the same core every survey target has used. A wiring
file is the four or five call sites and the two names, nothing more.

## The two shapes, and which levels need which

`a1hook.h` offers level 0 in two shapes, and the choice is not cosmetic.

**Interposed.** `A1_INTERPOSE_MALLOC`, the wiring forwards `a1_block_malloc`
and friends to libc, and level 0 is the whole process. Fifteen of the
seventeen levels take this shape, because their level 0 really is the system
allocator.

**Recorded.** The target makes the call itself and hands the result over with
`a1_block_record` and `a1_block_forget`. This is the shape for a level whose
level 0 is *another custom allocator*, which `a1hook.h` describes as "what a
target needs whose level 0 is its own named wrapper".

Two levels are nested this way, and they are why this branch exists:

| level | level 0 | what the recorded shape captures |
|---|---|---|
| `lookaside` | `memsys5` | a slot released to the connection free list while memsys5 still owns the block |
| `apr_bucket` | `apr_pool` | a bucket node returned to the bucket allocator's freelist while the pool still owns the node |

For these two, `A1_L0_NAME` must be the name of the allocator beneath, not
`libc`, and the wiring must call `a1_block_record` where that allocator hands
storage to this one and `a1_block_forget` where it takes it back. Interposing
malloc instead would measure the wrong boundary and the row would say nothing
the existing survey does not already say.

## One custom level per process

`a1core.inc` carries exactly two levels. A program with several nested
allocators therefore needs one run per allocator, which is why `levels.json`
lists seventeen levels and not seven, and why the four PostgreSQL managers are
four runs of the same pgbench workload rather than one run with four tags.

## What a wiring file must state in its own comment

The existing `aprhook.inc` is the worked example, and it states four things
that a reader cannot recover from the code. A new wiring states the same four:

1. **Which deaths are counted at this level**, and why that is the set no
   defense at level 0 can see. APR's wiring says a pool object has no free at
   all and that its deaths are exactly the invisible ones.
2. **What sits between the two levels.** APR's wiring names the allocator's
   node cache, because the pool hides objects behind nodes and the cache hides
   nodes behind itself.
3. **What is not measured here**, in plain words. APR's wiring already said
   `apr_bucket_alloc` was not measured, and that counting one of a program's
   two allocators can only understate the share of invisible frees. That
   sentence is what this branch is closing.
4. **Why it is an include**, if the build would otherwise have to change.

## Seam confidence

`levels.json` marks each seam `port-patch`, `port-document` or `to-confirm`.
A wiring may only be written against a seam that is at least `port-document`,
and the first build is what turns either into a fact. If the build shows the
seam is elsewhere, the map is corrected before the wiring is, so the two never
disagree.
