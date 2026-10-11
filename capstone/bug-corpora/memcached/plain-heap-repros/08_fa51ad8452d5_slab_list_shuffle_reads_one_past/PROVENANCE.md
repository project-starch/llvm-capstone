# fa51ad8452d5 — fix off by one in slab shuffling

## The defect

The backwards page-list shuffle ran `x` from 0 to `slabs - 1` and read `slab_list[x + 1]`, so its last read is at index `slabs`. `grow_slab_list` doubles the array only when `slabs == list_size`, which makes a completely full list a normal steady state — and in that state the read is one element past the allocation. The count was decremented only after the loop.

## Upstream defect

- **Fix:** `fa51ad8452d5`, *"fix off by one in slab shuffling"*, `slabs.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin decrements before the loop.

## The vulnerable code, quoted from the fix's parent

```c
    for (x = 0; x < s_cls->slabs; x++) {
        s_cls->slab_list[x] = s_cls->slab_list[x+1];
    }
    s_cls->slabs--;
```

## The fix

```c
    s_cls->slabs--;
    for (x = 0; x < s_cls->slabs; x++) {
        s_cls->slab_list[x] = s_cls->slab_list[x+1];
    }
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. Only the shuffle loop and its premise are reproduced: a pointer array that is exactly full, which is the state `grow_slab_list` leaves. No page mover, no rebalance thread, no slab classes. This matters because the surrounding subsystem is out of reach, while the loop alone is not.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
