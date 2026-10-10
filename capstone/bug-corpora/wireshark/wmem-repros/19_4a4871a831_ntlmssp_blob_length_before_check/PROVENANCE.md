# 4a4871a831 — ntlmssp: swap bounds check and length for memcpy

## The defect

The metadata overstates the allocation: `result->length = blob_length` runs unconditionally, then `if (blob_length < max_blob_size)` guards the allocation, so for an oversized blob the struct claims a length its buffer does not have and a later memcpy at offset 32 reads past it.

## Upstream defect

- **Fix:** `4a4871a831` ("ntlmssp: swap bounds check and length for memcpy").
- **CVE:** none assigned.
- **Live at the `v4.6.8` pin: NO** — the fix is already in, so this is a **fix-reversal.**

## The vulnerable code, quoted from upstream

The fix's own diff, `epan/dissectors/packet-ntlmssp.c`:

```diff
   if (result != NULL) {
-    result->length = blob_length;
     if (blob_length < MAX_BLOB_SIZE)
     {
-      result->contents = (guint8 *)wmem_alloc(wmem_file_scope(), blob_length);
-      tvb_memcpy(tvb, result->contents, blob_offset, blob_length);
+      result->length = blob_length;
+      result->contents = (guint8 *)tvb_memdup(wmem_file_scope(), tvb, blob_offset, blob_length);
```

and the consumer that trusts the length, `:1649`:

```c
  if (conv_ntlmssp_info != NULL && conv_ntlmssp_info->ntlm_response.length > 24) {
    memcpy(conv_ntlmssp_info->client_challenge, conv_ntlmssp_info->ntlm_response.contents+32, 8);
```

**Liveness**, read from the pinned source rather than from ancestry — the latter has called
backported fixes live before — and keyed on an identifier that exists on exactly one side:

read from the PINNED source: v4.6.8:epan/dissectors/packet-ntlmssp.c contains the fixed `tvb_memdup(wmem_file_scope(), tvb, blob_offset, blob_length)` 1 time and the pre-fix unconditional `result->length = blob_length;` before the size check 0 times. Two-sided. A fix-reversal.

**The metadata is written before the allocation it describes is known to have happened.**
`result->length` is assigned unconditionally; the allocation is guarded. For a blob at or above
`MAX_BLOB_SIZE` the struct therefore claims a length its buffer does not have, and the consumer at
`:1649` reads eight bytes at offset 32 of a buffer that is stale from an earlier, smaller blob. The
fix moves the one assignment inside the check.

## Why this is a NESTED row

wmem, a chunk the BLOCK or BLOCK_FAST allocator carved from a block g_malloc handed out. An inner allocator carved the crossed region, so under this inventory's axis -- WHO ALLOCATED THE OBJECT -- this row IS NESTED. A malloc-granular bound cannot see the crossing: the block wmem carved it
from is one `g_malloc`, and the access stays inside that block.

## What is real here, and what is reduced

**Real:** the allocator. Upstream's own wmem, through this corpus's seam, with the chunks carved
consecutively from the same block so the successor's position can be asserted.

**Reduced:** MAX_BLOB_SIZE is 256 and the earlier, smaller blob is 24 bytes -- the reachable case, since the struct is per-conversation and reused. The case asserts that the recorded length exceeds what was allocated, that the consumer's `> 24` threshold admits it, and that the read at offset 32 is past the chunk yet inside the block.

## What the run establishes, and what it does not

**Establishes:** the crossing is created — the case's own `CHECK` assertions must hold for it to exit
0, so a reduction whose arithmetic missed fails rather than reporting a verdict about nothing — and
**stock CheriBSD does not catch it**, measured 2026-10-07 with a revocation control faulting in the
same boot.

**Does NOT measure** the Capstone, PoisonCap or native arms. Those are declared: these four cases
have not had a Capstone domain build. Note also what the corpus's own `corpus.json` says and which
applies here: every arm of this harness narrows a wmem allocation to its request via `wm_narrow()`,
so a spatial crossing faults on *all* arms and these rows do not discriminate the chunk port.

**N = 1 per cell.**
