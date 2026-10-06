# ed20250c13 — proto.c: protect against buffer overflow in proto_find_undecoded_data()

## The defect

`for (i = fi->start; i < fi->start + fi->length; i++) { decoded[i/8] |= 1 << (i%8); }` consults the field extent only, so a field whose start plus length reaches past the frame drives the byte index past the bitmap.

## Upstream defect

- **Fix:** `ed20250c13` ("proto.c: protect against buffer overflow in proto_find_undecoded_data()").
- **CVE:** none assigned.
- **Live at the `v4.6.8` pin: NO** — the fix is already in, so this is a **fix-reversal.**

## The vulnerable code, quoted from upstream

The allocation, `ed20250c13^:epan/proto.c:9709`:

```c
	gchar* decoded = (gchar*)wmem_alloc0(wmem_packet_scope(), length / 8 + 1);
```

and the fix, which threads the bitmap's own length through a struct so the loop can be bounded by it:

```diff
+typedef struct {
+	gint length;
+	gchar *buf;
+} decoded_data_t;
...
-		for (i = fi->start; i < fi->start + fi->length; i++) {
+		for (i = fi->start; i < fi->start + fi->length && i < decoded->length; i++) {
 			byte = i / 8;
 			bit = i % 8;
```

**Liveness**, read from the pinned source rather than from ancestry — the latter has called
backported fixes live before — and keyed on an identifier that exists on exactly one side:

read from the PINNED source: v4.6.8:epan/proto.c contains the fix's `decoded_data_t` struct 4 times and the pre-fix bare `gchar* decoded = (gchar*)wmem_alloc0` 0 times. Two-sided. A fix-reversal.

**The loop is bounded by the field's extent and by nothing else.** A field whose `start + length`
reaches past the frame — which a malformed or truncated-after-reassembly capture produces — drives
`byte` past `length / 8` and the `|=` writes into the next chunk. The index comes from the protocol
tree's *own* field extents, so the loop is trusting a sibling subsystem's output rather than an
input; accordingly the fix does not clamp the field, it bounds the write.

## Why this is a NESTED row

wmem, a chunk the BLOCK or BLOCK_FAST allocator carved from a block g_malloc handed out. An inner allocator carved the crossed region, so under this inventory's axis -- WHO ALLOCATED THE OBJECT -- this row IS NESTED. A malloc-granular bound cannot see the crossing: the block wmem carved it
from is one `g_malloc`, and the access stays inside that block.

## What is real here, and what is reduced

**Real:** the allocator. Upstream's own wmem, through this corpus's seam, with the chunks carved
consecutively from the same block so the successor's position can be asserted.

**Reduced:** the frame is 64 bytes, so the bitmap is 9. The field claims start 56 and length 32, reaching index 87 and byte 10 -- past the bitmap and inside the block, both asserted.

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
