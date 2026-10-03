# Class 3 (reuse-not-free) on memcached's real bipbuffer: every arm is blind, measured (2026-10-03)

**Question.** Hooking memcached's small core allocators began with reading the bipbuffer — the
logger's per-watcher output buffer, which also carries `items.c`'s `lru_bump_entry` records. It turned
out not to be the class expected. **Does anything we currently run catch a lifetime that ends with no
allocator event at all?**

**Pre-registration.** Fixture 19 and all five predictions were pushed in `4a30b3f34dbc` **before** the
build, and they register a predicted **blindness** rather than a catch.

## Verdict

**Five of five cells exactly as pre-registered: every arm returns. The revoking arms are blind too.**

| arm | fixture 19 — class 3 | fixture 18 — class 1, **same boot, after it** |
|---|---|---|
| `level0` | RETURN `130015b` | RETURN `12001ee` |
| `shrink` | RETURN `130015b` | RETURN `12001ee` |
| `sublet` | **RETURN `130015b`** | **FAULT** cause 24 |
| `slabsublet0` | **RETURN `130015b`** | **FAULT** cause 24 |
| `slabsublet1` | **RETURN `130015b`** | **FAULT** cause 24 |

Exit status corroborates every RETURN: `91 = 0x5b = 130015b & 255`.

## Why this is class 3 and not a use-after-free

Read from the port's own source tree, not inferred:

    bipbuf_new      malloc(sizeof(bipbuf_t) + size)                 -- ONE allocation, flexible data[]
    bipbuf_request  return (unsigned char *)me->data + me->a_end    -- a pointer INSIDE it
    bipbuf_poll     void *end = me->data + me->a_start;
                    me->a_start += size;
                    ... me->a_start = me->a_end = 0;  return end;   -- CURSORS ONLY

`bipbuf_poll` performs **no free of any kind**. It advances cursors and, when the buffer empties,
resets them to zero — so the next `bipbuf_request` hands out the very same bytes. A consumer holding a
polled pointer names another record's data through a pointer that **was never freed, is still tagged,
and is still in bounds of the single `malloc`**. Only the data's identity changed.

That is class 3 in [`docs/design/sharing-bug-taxonomy-and-novelty.md`](../../../../../docs/design/sharing-bug-taxonomy-and-novelty.md):
*"owner recycles the buffer in place; no `free()` ever happens, pointer stays tagged and in-bounds,
only the data's identity changes"* — the row where ASan, GC, Rust, CHERI spatial, CHERI async **and
CHERI eager** are all listed blind, and which
[`docs/ref/paper-bug-inventory.md`](../../../../../docs/ref/paper-bug-inventory.md) records as having
**one** row of evidence, with its two built specimens marked *"Capstone/CHERI columns TODO"*.

**The condition was created, not assumed.** `frees-performed=0` is printed beside
`polled-same=1 reissued-in-place=1`, and the fixture exits `0xE0019` rather than a mark if the buffer
fails to recycle in place.

## The blindness is visible in the capability itself

On `sublet`, the arm with the narrowest bounds:

    record1  cursor=c86cc020  bounds=[c86cc000,c86cd020)  len-from-cursor=4096
    record2  cursor=c86cc020  bounds=[c86cc000,c86cd020)  len-from-cursor=4096

Identical cursor **and identical bounds**. Both records are sub-allocations inside one `malloc`
(`data[]` sits at `+0x20` after the port's 16-byte alignment, so `len-from-cursor` is the whole
4096-byte data area). **Per-object bounds on the outer allocation cannot separate the records** —
which is precisely what "nested allocator" means, made visible rather than argued.

## The in-boot positive control, which is the load-bearing part

A clean RETURN on a protection arm is exactly what a dead arm, a mis-built image or a boot that never
ran the fixture also looks like. So fixture 18 ran **in the same boot, on the same arm, immediately
after 19**: the same stale-pointer-then-reuse shape, but with a release that *does* reach the runtime
allocator.

On `sublet`, `slabsublet0` and `slabsublet1` it **faulted**, cause 24, in that same boot. Fixture 1
also ran first in all five boots and returned `100001`. So the arms demonstrably fault, there and
then, on the class whose lifetime ends at the allocator. **The only difference between the two
fixtures is where the lifetime ends.** That is what makes this a measurement of the bipbuffer rather
than a silence.

This is also the control the tshark bundle was audited for lacking, applied here by construction
rather than retrofitted.

## What this does and does not say

- **It is the first Capstone-column measurement of a class-3 specimen**, and it is a **negative** one.
  The inventory's two existing row-3 specimens are host-verified with ASan silent and a positive
  control firing; their Capstone column was TODO. This fills it for a third specimen — in an allocator
  we actually port and run — and the answer is that **none of the five arms catches it**.
- **Consequence for a claim that is not mine to edit.** The blindness map gives class 3 an unqualified
  **✓** in the Capstone column. This measurement says that ✓ is **not supported by the heap arms as
  they stand** — `level0`, `shrink`, `sublet`, `slabsublet0` and `slabsublet1` all return. It would
  require the nested allocator itself to be hooked, so that `bipbuf_poll` becomes a revocation event.
  That is a scoping qualification on a paper claim, reported here and deliberately **not** written
  into the paper.
- **It is not a defect report.** `bipbuffer.c` has only **two** commits in upstream's entire history —
  the initial import and a Solaris compiler-warning fix — so there is no upstream defect *in* it. What
  this fixture reproduces is the allocator's **documented, intended** recycling contract, which is
  what makes class 3 underreported: nothing is logged because nothing is wrong, locally.
- **ASan was not run here.** Its blindness to this shape is structural (no free, no realloc, one
  malloc) and is asserted from the taxonomy, not measured in this bundle.
- **QEMU only. N = 1 per cell** for fixture 19. Fixture 18's faults, however, now reproduce on a
  **second, different image** (`sublet` `f61c16a447dcd794` here against `1e3df2b1edefeb31` in
  `../2026-10-03-qemu-plain-heap-contrast/`), so that result is N = 2 across images.
- **The port's own oracle did not run** — `run-safety.py` needs `capstone-vm`, which this rootfs
  cannot support. The guest command is that runner's, replicated over serial/9p with `mc-harness`
  cross-built for the guest, and the comparison against the expect file was done by hand.

## What the next step would be, and what it costs

Hooking the bipbuffer means giving `bipbuf_request`/`bipbuf_push` a per-record sublet region and making
`bipbuf_poll` **revoke** it. The existing slab adapter is the pattern and the yardstick:
`src/slab-sublet/` is **137 lines** across three files with 14 hook functions, plus one patch
(`0006-slab-sublet-hooks`). A bipbuffer adapter is a smaller surface — three call sites, one buffer
kind, no page/chunk split. Measured build cost for the memcached port as it stands: libevent 199 s
once, arm SDKs 14 s, safety images 9 s, slab image 19 s.

Also recorded while reading, as the residual of a defect already built: `e779381` (fixture 18) took
**three** commits to fix — `e779381`, then `3bc58f6` *"prior fix missing continue, still broke"* and
`3a66ebe` *"fix null watcher more"*, all ancestors of our pin. The final form's loop condition is
`while (!w->failed_flush && (skip_scr = bipbuf_request(w->buf, ...)) == NULL)`, so the unfixed version
reads the freed watcher twice more — including `w->buf` handed **straight into the bipbuffer
allocator**. Fixture 18 reduces only the write half. That stale-pointer-into-the-nested-allocator
half is a second shape, available and not yet built.

Files: `result-lines.txt` (every line above).
