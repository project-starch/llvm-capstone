# 78eb770 — the flags are copied into suffix space that was never allocated

## The defect

`do_item_alloc` fills the item header and copies the flags into the suffix field:

```c
memcpy(ITEM_suffix(it), &flags, sizeof(flags));
```

**When `nsuffix` is 0 the item has no suffix space at all**, and `ITEM_suffix(it)` is then the same
address as `ITEM_data(it)` — so the four-byte copy lands in the **value's** storage. With a
zero-length value (`nbytes = 2`, just the CRLF) it overruns the value entirely. Upstream's subject
states the cause: *"When nsuffix is 0 space for flags hasn't been allocated so don't memcpy them."*

## Upstream defect

- **Fix:** `78eb770` (2019-05-24). It wraps the copy in `if (nsuffix > 0)`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** `1.6.45:items.c:330-332` carries the guard verbatim. A
  fix-reversal.

## This one is a SUB-OBJECT crossing, and that is the interesting part

The overrun stays **inside the chunk**: it overwrites the value's storage and the bytes after it that
belong to the chunk but not to the value. So a **chunk-granular bound cannot see it** — and the slab
port's bound *is* the chunk. That is the same shape FFmpeg's `subobject-repros` corpus is built
around, reached here from the other end: a field boundary inside one allocation.

The case asserts the premise rather than assuming it — that with `nsuffix = 0`, `ITEM_suffix` and
`ITEM_data` really are the same address — and places a sentinel in the two bytes past the value.

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
