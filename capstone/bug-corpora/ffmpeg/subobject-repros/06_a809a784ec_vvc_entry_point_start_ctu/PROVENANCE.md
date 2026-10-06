# a809a784ec — walks off `entry_point_start_ctu` because its write index is bounded by nothing

## The defect

`sh_entry_points` appends an entry for every tile or entropy-sync boundary in the slice,
incrementing `j` each time, and compares `j` against nothing. A slice whose every CTU starts a
boundary makes `j` advance on every iteration, so the writes run past the array for as long as the
CTU count lasts.

## Upstream defect

- **Fix:** `a809a784ec`. It returns `AVERROR_INVALIDDATA` once `j` reaches `VVC_MAX_ENTRY_POINTS`, and changes the function's return type to carry that out — which is why the diff touches `sh_derive` too.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: YES.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavcodec/vvc/ps.c`:

```diff
-static void sh_entry_points(VVCSH *sh, const H266RawSPS *sps, const VVCPPS *pps)
+static int sh_entry_points(VVCSH *sh, const H266RawSPS *sps, const VVCPPS *pps)
 {
     if (sps->sps_entry_point_offsets_present_flag) {
         for (int i = 1, j = 0; i < sh->num_ctus_in_curr_slice; i++) {
             ...
             if (pps->ctb_to_row_bd[ctb_addr_y] != pps->ctb_to_row_bd[pre_ctb_addr_y] || ...) {
+                if (j >= VVC_MAX_ENTRY_POINTS)
+                    return AVERROR_INVALIDDATA;
                 sh->entry_point_start_ctu[j++] = i;
             }
         }
     }
+
+    return 0;
 }
```

The array is **VVCSH's last member**, `n9.0.1:libavcodec/vvc/ps.h:264-266`:

```c
    // entries
    uint32_t entry_point_start_ctu[VVC_MAX_ENTRY_POINTS];   ///< entry point start in ctu_addr
} VVCSH;
```

and the bound, `n9.0.1:libavcodec/vvc.h:153`:

```c
    VVC_MAX_ENTRY_POINTS = VVC_MAX_TILE_COLUMNS * 135,
```

**Liveness: LIVE AT THE `n9.0.1` PIN**, read from the pinned source rather than from
ancestry — `git merge-base --is-ancestor` has called backported fixes live before, so it is not
used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavcodec/vvc/ps.c` contains the pre-fix `static void sh_entry_points` — **1 occurrence**
- the same file contains the fixed `j >= VVC_MAX_ENTRY_POINTS` — **0 occurrences**

**Why this is still a sub-object crossing, although the array is last in its struct.** A
`VVCSH` is itself a member of the slice context — every consumer reaches it as `&lc->sc->sh`
(`libavcodec/vvc/ctu.c:532`, and eight other sites in that file) — so a write past VVCSH's last
member lands on the **enclosing** struct's next member, inside the same allocation. The case asserts
that adjacency, rather than assuming it, because this is the one row in the corpus where the
containment claim rests on the enclosing struct and not on the struct the array is declared in.

**The magnitude is unbounded**, unlike the single-step crossings of cases 3, 4 and 5. The reduction
keeps the overrun inside the allocation so that what is measured is the sub-object claim.

**Reachability note:** the VVC decoder's slice-header parse, for a stream with
`sps_entry_point_offsets_present_flag` set and tiles or entropy sync enabled.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** `VVC_MAX_ENTRY_POINTS` is cut from `VVC_MAX_TILE_COLUMNS * 135` to **64**,
and the boundary condition to "every CTU starts a boundary" — which is the reachable worst case the
constant's own comment describes. The defect is that `j` is bounded by nothing, so the array's real
length changes how many iterations it takes to leave, not whether it leaves. Both reductions are
stated in `case.c` as well, not only here.

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
