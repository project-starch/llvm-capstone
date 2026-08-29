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

## What is NOT ready, and why it must not be measured yet

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
