# fb862976df — guards its loop with a counter the loop never updates, so the guard cannot fire

## The defect

The uniform-tile-spacing loop fills `col_width_val[i]` while `remaining_size > 0`, and
guards itself with `if (current->num_tile_columns > VVC_MAX_TILE_COLUMNS)`. But
`num_tile_columns` is assigned from `i` only **after** the loop, so throughout the loop it holds a
stale value and the check cannot fire. The real counter, `i`, is tested nowhere.

## Upstream defect

- **Fix:** `fb862976df`. It tests `i == VVC_MAX_TILE_COLUMNS` inside the loop and deletes the now-redundant post-loop copy of the check.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavcodec/cbs_h266_syntax_template.c`:

```diff
         while (remaining_size > 0) {
-            if (current->num_tile_columns > VVC_MAX_TILE_COLUMNS) {
+            if (i == VVC_MAX_TILE_COLUMNS) {
                 av_log(ctx->log_ctx, AV_LOG_ERROR,
-                       "NumTileColumns(%d) > than VVC_MAX_TILE_COLUMNS(%d)\n",
-                       current->num_tile_columns, VVC_MAX_TILE_COLUMNS);
+                       "Exceeded maximum tile columns (%d) (remaining size: %u)\n",
+                       VVC_MAX_TILE_COLUMNS, remaining_size);
                 return AVERROR_INVALIDDATA;
             }
             unified_size = FFMIN(remaining_size, unified_size);
             current->col_width_val[i] = unified_size;
             remaining_size -= unified_size;
             i++;
         }
         current->num_tile_columns = i;
-        if (current->num_tile_columns > VVC_MAX_TILE_COLUMNS) {
-            ...
-            return AVERROR_INVALIDDATA;
-        }
```

The adjacent members, `n9.0.1:libavcodec/cbs_h266.h:594-595`:

```c
    uint16_t col_width_val[VVC_MAX_TILE_COLUMNS];           ///< ColWidthVal
    uint16_t row_height_val[VVC_MAX_TILE_ROWS];             ///< RowHeightVal
```

**Liveness: a FIX-REVERSAL.** The fix is already in at the pin, read from the pinned
source rather than from ancestry — `git merge-base --is-ancestor` has called backported fixes live
before, so it is not used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavcodec/cbs_h266_syntax_template.c` contains the fixed `i == VVC_MAX_TILE_COLUMNS` — **1 occurrence**
- the same file contains the pre-fix `num_tile_columns > VVC_MAX_TILE_COLUMNS` — **0 occurrences**

The case reconstructs the **pre-fix consumer shape against the shipped allocator**, which is this
tree's convention and not a weaker kind of case: it is stated at
`../../memcached/allocator-repros/README.md:132-135`, and most cases in this tree are fix-reversals.
Liveness is **recorded, never required**; requiring it is what left this cell nearly empty, and that
inference was retracted on `dev`.

**Why this row is worth having beyond the crossing itself.** This is the
**gate-that-cannot-fire** shape in upstream code: a guard that is present, reads correct, and is
keyed to a variable the loop does not update. The two arms therefore differ by the guard's **key**,
not by the presence of a guard — the buggy arm keeps the stale check and prints its value, so the
reading is about the defect rather than about adding a check. No other case in this corpus has that
shape.

The crossing itself is member-to-member: `col_width_val` is followed directly by `row_height_val` in
one allocation, which the case asserts.

**Reachability note:** the H.266/VVC coded-bitstream reader, reached through the
`vvc_metadata` and `trace_headers` BSFs as well as the VVC decoder's parameter-set parse.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** `VVC_MAX_TILE_COLUMNS` and `VVC_MAX_TILE_ROWS` are cut to **20** each. The
defect is that the bound check is keyed to a stale counter, so the arrays' real lengths change only
how many iterations it takes to leave. The stale guard is **reproduced** in the buggy arm rather than
deleted.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control (fixed) arm run first.

**Does not establish** reachability upstream, anything about silicon, or the Capstone, PoisonCap,
CheriBSD and ASan readings. Those arms are **declared predictions** in `case.json` and are *not*
measured for this case. Measuring the Capstone arm needs a probe case in
`ports/ffmpeg/buffer-pool/security-tests` — the seam cases 0-2 use — and measuring ASan needs
`results/20261005-native-subobject/asan-probe.c` extended, with its positive control still firing.

**An arena caveat that bears on the Capstone and CheriBSD predictions.** This corpus's driver hands
the port's `av_malloc` **one** arena (`shared/driver.c` calls `ff2_memory_init`), and
`ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26` carves it by bumping a cursor,
returning a raw interior pointer and rounding every request up to 64 bytes;
`__builtin_capstone_cap_shrink` appears only in `src/capstone-domain/payload-capabilities.c:57`, on
pool *payload* blocks. So on this harness there is **no per-allocation bound on the struct at all**,
and a completion would be weaker evidence than "the bound is the whole allocation". The sub-object
claim does not rest on that — the crossing is interior by construction, which the case asserts on
the offsets — but the arm must not be read as a measured per-allocation bound.

**N = 1 per cell.**
