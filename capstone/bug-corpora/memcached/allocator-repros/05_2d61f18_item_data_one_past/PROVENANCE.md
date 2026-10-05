# 2d61f18 — three bytes of a two-byte terminator, one byte past the item

## The defect

The incr/decr path rewrites an item's value and then appends the protocol terminator. The item is
sized for exactly `res + 2` bytes, and the copy was told to write **3**:

```c
memcpy(ITEM_data(new_it), buf, res);
memcpy(ITEM_data(new_it) + res, "\r\n", 3);     /* two bytes of string, three written */
```

`"\r\n"` is two characters plus a NUL, so a length of 3 writes that NUL **one byte past the item's
data**. Upstream's subject calls it what it is: *"Fix heap corruption when copying too much data
onto an item."*

## Upstream defect

- **Introduced by** `c47ee89` (2007-04-12, *"fix potential bug with memcpy(), use in two more
  places"* — which changed the length from 2 to 3), **fixed by** `2d61f18` (2008-06-18). Having both
  halves in history is how the shape was confirmed rather than inferred.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO** — a fix-reversal, like four of the five temporal rows. The site
  survives verbatim with the fix applied: `1.6.45:memcached.c:2316-2317` reads
  `memcpy(ITEM_data(new_it), buf, res);` then `memcpy(ITEM_data(new_it) + res, "\r\n", 2);`.

## Why this crosses the CHUNK bound and not just the item's data bound

A slab class rounds its chunk size up, so an item's data usually ends **below** the chunk end and a
one-byte overflow would land in the class's own rounding slack. The case removes that accident: it
takes two chunks from the real allocator, measures the **stride** between them, and sizes the item's
data to end exactly at it. The one-byte overflow then crosses into the next chunk of the page. The
stride is derived, never assumed from a class size.

**A process note worth keeping.** The first version of this case asserted that the second chunk sat
above the first. The free list handed them out the other way, and the control refused the run with
`CONTROL-FAILED 753` rather than quietly measuring nothing. The case now orders the two chunks and
says so.

## What is real here, and what is reduced

**Real:** the allocator. `slabs.c` from memcached 1.6.45 as the port builds it, so the chunk and
page geometry the case turns on is a property of memcached's slab allocator, not of this driver.

**Reduced:** no server, no hash table, no LRU, no protocol parse. The item is laid out in a real
slab chunk by hand and the consumer is reduced to the access that crosses.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential — the buggy
arm's crossing happens and the case's `damage` flag is set; the fixed arm's bound keeps it inside.

**Does not establish** any Capstone or CheriBSD reading. Those arms are **declared predictions**
and were not measured: this run made no domain build. Each `case.json` says which way it predicts
and why, so the reading settles it rather than confirming an assumption.

**Does not establish** upstream reachability.
