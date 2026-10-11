# Case 0 — `19c51d27b9`, the NetScaler page-buffer overrun

## Upstream

`19c51d27b9`, subject **"Don't go past the end of a page in a NetScaler file."**, one file,
`wiretap/netscaler.c`, 69 insertions and 16 deletions.

The fix adds three guards to the `PACKET_DESCRIBE` macro, none of which the parent has:

```c
/* Make sure the record header is entirely contained in the page */
if ((nstrace_buflen - nstrace_buf_offset) < sizeof *type) { ... return FALSE; }
/* Check sanity of record size */
if ((phdr)->caplen < sizeof *type) { ... return FALSE; }
/* Make sure the record is entirely contained in the page */
if ((nstrace_buflen - nstrace_buf_offset) < (phdr)->caplen) { ... return FALSE; }
```

It also widens `nstrace_buf_offset` and `nstrace_buflen` from `gint32` to `guint32`, which is what
makes the subtraction above safe.

## The defect

| | `wiretap/netscaler.c` | text |
|---|---:|---|
| the page buffer | `:682` (parent) | `nstrace_buf = (gchar *)g_malloc(NSPR_PAGESIZE);` |
| its size | `:52` | `#define NSPR_PAGESIZE 8192` — unchanged at the pin |
| the length | `:957` (parent) | `(phdr)->caplen = pletoh16(&pp->nsprRecordSize);` |

`caplen` is a **16-bit field read straight out of the file**, so 0..65535, and the pre-fix macro
relates it to nothing. The copy is

```c
memcpy(ws_buffer_start_ptr(wth->frame_buffer), type, (phdr)->caplen);
```

with `type = (…*) &nstrace_buf[nstrace_buf_offset]`. A crafted file supplying the maximum size while
the reader is near the end of a page therefore reads tens of kilobytes past an 8192-byte
`g_malloc` — **65471 bytes past it** in this case's arrangement.

No container is involved. `g_malloc` is `malloc` plus abort-on-failure; there is no capacity field,
no doubling and no slack, unlike `ws_buffer`, `GArray` and `GByteArray`, which is what disqualified
several other candidates in the same triage.

## What is reduced, and what is not

- **The allocator is real in shape**: the allocation is the upstream size and the length is the
  upstream field's full range. GLib is not linked — `g_malloc` is `malloc` plus abort-on-failure,
  and `g_free` is `free`, which is all this case uses of it.
- **The arms differ by one guard**, the third of the three above. The allocation, the offset and the
  record size are identical in both; the fixed arm **refuses the record** rather than shortening it,
  which is why `cap`, `touched` and `extent` read the same on both arms.
- **Reduced to the first crossing byte.** The unreduced `memcpy` spans 65471 bytes past the
  allocation; walking all of it would be an unbounded read through whatever follows, which is a
  property of the defect and not something a reduction should perform. The magnitude is reported as
  `extent=` on both arms so it is never lost.
- **Left out:** the two other guards (the record *header* containment check and the
  `caplen < sizeof *type` sanity check), the `gint32` → `guint32` widening, and the v10/v20 macro
  expansions. Each bounds a different part of the same read; this case is the record-body crossing.

## Liveness

`live_in_pin: false`. The fix is an ancestor of `v4.6.8`, and the pin carries its guards —
`v4.6.8:wiretap/netscaler.c` contains **2** occurrences of `"record crosses page boundary"` while
the page buffer is still `g_malloc(NSPR_PAGESIZE)` at `:731`. The parent has the allocation at
`:682` and no guard at all.

Reconstructing a pre-fix consumer shape against the shipped allocator is this tree's normal
practice, and the convention is stated at `../../memcached/allocator-repros/README.md:132-135`.

## Measured

`results/20261006-native-plain-heap/`:

```
fixed  cap=8192 touched=73663 extent=65471 crossed=0 damage=0   VERDICT FIXED
buggy  cap=8192 touched=73663 extent=65471 crossed=1 damage=1   VERDICT DEFECT-REPRODUCED
asan   buggy: heap-buffer-overflow, READ of size 1, 0 bytes after 8192-byte region,
       at read_probe (shared/corpus.h:64), allocation at case.c:28
       fixed: silent, exit 0
```

**ASan reports this one.** The crossing leaves the `malloc` bound, so a redzone sits exactly where it
lands. `../../wmem-repros` and `../../ffmpeg/subobject-repros` record ASan *blind* for the opposite
reason, and the contrast is the project's own axis, measured rather than argued.

## The arm that is here to say "nothing changes"

`sublet-chunks` is a required arm of this corpus and its prediction is that **the chunk port makes no
difference**: the object is not a wmem chunk but a direct `g_malloc`, so narrowing wmem chunks
cannot touch it. That is the row's point — it is the one tshark case the inner-allocator port is
irrelevant to, and it belongs beside the five wmem cases the port *does* discriminate.
