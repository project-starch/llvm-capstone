# WAMR EMS, PR 2279: the column is set up, the oracle is not finished

WAMR is the first serious runtime whose Capstone column already runs, which is why
its CHERI column is being built next. This directory is that column's material for
case 1 of the EMS set.

## What is ready

* **One tree, two arms.** `patches/0005-make-the-PR-2279-pinuse-fix-a-knob.patch`
  in `benchmarks/wamr/` turns the upstream fix at `ems_alloc.c:837` into
  `WAMR_EMS_PINUSE_BUG`, so the vulnerable and the fixed build differ in exactly
  that one line. The knob previously existed only as an uncommitted edit in the
  fetched source under `$CAPSTONE_TMP_ROOT` and would have been lost on the next
  fetch; it is a patch now.
* **`ems_pinuse.c`**, the trigger, which is upstream's own verified reproducer
  (`samples/mem-allocator/main.c`) with a question added.
* **`ems_stubs.c`**, the seven platform symbols EMS needs so the allocator can be
  driven without the whole runtime. None is on the path under test: the mutexes are
  no-ops because the reproducer is single-threaded and logging is discarded. EMS's
  own sources are compiled unmodified apart from the knob.
* **A host control that separates the arms 7 times out of 7**, at second-allocation
  sizes 8, 16, 32, 64, 128, 192 and 256.

## The oracle is finished, and the answer is that this case is not a blind spot

The oracle works and is positively controlled: the overlap predicate reports 1 on
two ranges that do overlap, and the two arms separate 7 times out of 7. What it
reports about the defect is this:

| | fixed | vulnerable |
|---|---|---|
| second allocation, sizes 8..256 | succeeds | **NULL** |
| 16 further allocations of 32 B | all succeed, none near the live block | **none succeed** |
| any allocation landing inside the live block | no | **no** |
| the live block's payload after the corrupting free | unchanged | **unchanged** |
| forged-chunk sizes 0x20..0x100 x asks 16..128, 24 combinations | -- | **0 overlaps, 0 clobbers** |

Without the forged header the two arms are byte-for-byte identical, so the forgery
is part of the trigger rather than decoration; it stands for data a wasm module
writes into its own block.

**So the consequence of PR 2279, as reproduced here, is DENIAL: the backward
coalesce destroys the free list and the allocator stops serving. It is not a
memory-safety violation.** No access goes out of any bounds, nothing is handed out
twice, and the live object is never written. There is therefore nothing for CHERI
to miss, and running it under CheriBSD would produce a true statement about
nothing.

That is a result, not a failure to reproduce: **EMS case 1 is a robustness bug, not
a blind-spot case.** The inventory lists it as "[V] a self-contained C reproducer is
already in our tree", which is accurate -- the reproducer reproduces. What it
produces is not what this study measures.

## Where to go instead, within the same allocator

Case 3 of the EMS set (PR 428, `remove_tree_node` missing its `hmu_is_in_heap`
check) is described as a **write-anywhere primitive** from a corrupted free-tree
node. A write-anywhere that stays inside the pool is exactly the blind spot, and
unlike case 1 it has a memory-safety consequence by construction. Case 4
(`gc_migrate` walking the heap on an attacker-influenced size) is the second
candidate for the same reason.

The apparatus here carries over unchanged: the knob patch, the stubs, the
build recipe and the overlap oracle all apply to any EMS case.

## What was ready before that



The separation is real and deterministic, but the consequence it shows is the
**allocator failing**, not the defect's documented one:

```
fixed        n=8..256   second allocation succeeds
vulnerable   n=8..256   second allocation returns NULL
```

`inside_live` and `clobber` are 0 at every size. So the free list is destroyed and
the allocator refuses, rather than handing the same memory out twice.

**That is not yet a blind-spot measurement.** Where no access goes out of bounds
there is nothing for CHERI to miss, and reporting "CHERI does not catch it" on this
oracle would be hollow -- true, and about nothing. The case's documented mechanism
is "a later free coalesces BACKWARDS into a live object and the allocator hands the
same memory out twice", and the trigger sequence here does not reach it.

## What would finish it

Reach the double hand-out. The sequence stops before upstream's final `free(p)`,
and the backward coalesce that matters is the one that merges the freed neighbour
into the still-live block. Vary which block is freed and in what order until a
later allocation lands INSIDE the live one, then the oracle is
`inside_live`, which is already implemented and already returns 0 honestly.

Only then is the CHERI run worth spending: everything happens inside `store[]`, one
static array EMS carves internally, so purecap sees in-bounds accesses through a
capability derived from `store` and revocation has nothing to revoke -- but that
argument is only worth making once there is an access to argue about.
