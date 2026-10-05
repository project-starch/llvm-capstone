# ecdb011 — an unterminated item key handed to a string formatter

## The defect

`do_item_cachedump` passed `ITEM_key(it)` straight to a `%s`-style formatter. **An item's key is not
NUL-terminated in the struct**, so the read runs forward from the key into the item's later fields
and on through the chunk, stopping wherever the next zero byte happens to be.

## Upstream defect

- **Fix:** `ecdb011` (2009-02-12), *"Fix memory corruption error in stats cachedump."* It copies a
  bounded `nkey` into a local buffer and terminates it there. A follow-up, `bd2f3ab`, tightened that
  copy from `nkey + 1` to `nkey`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** `1.6.45:items.c:669-670` has the later, tightened form —
  `strncpy(key_temp, ITEM_key(it), it->nkey);` then `key_temp[it->nkey] = 0x00; /* terminate */` —
  and `:672` formats `key_temp`. A fix-reversal.

## The only row here whose crossing has no fixed extent

How far the read goes depends on where the next zero byte is, which is **why the fix bounds the copy
rather than adding a terminator at the source**. The case caps its own scan at 64 bytes so a run
cannot wander, which means it measures **that** the key bound was crossed, not how far — and the
`damage` flag is exactly that: the scan passed `nkey`.

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
