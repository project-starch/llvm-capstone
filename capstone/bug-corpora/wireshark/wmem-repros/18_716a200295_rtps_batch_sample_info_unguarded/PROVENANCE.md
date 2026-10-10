# 716a200295 — RTPS: Fix OOB write in DATA_BATCH sample info list

## The defect

A gate disabled by its own default: `rtps_max_batch_samples_dissected > 0 && count >= rtps_max_batch_samples_dissected` cannot fire when the preference is 0, which is the shipped default, so nothing bounds the counter while the arrays stay sample_info_max long.

## Upstream defect

- **Fix:** `716a200295` ("RTPS: Fix OOB write in DATA_BATCH sample info list").
- **CVE:** none assigned.
- **Live at the `v4.6.8` pin: YES** — the vulnerable guard is still there and the fix's marker is absent.**

## The vulnerable code, quoted from upstream

The guard, `716a200295^:epan/dissectors/packet-rtps.c:16734`:

```c
      if (rtps_max_batch_samples_dissected > 0 && (unsigned)sample_info_count >= rtps_max_batch_samples_dissected) {
        expert_add_info(pinfo, list_item, &ei_rtps_more_samples_available);
        offset = sampleListOffset;
        break;
      }
```

and the fix, with its own explanation of why the preference was the wrong thing to guard on:

```diff
-      if (rtps_max_batch_samples_dissected > 0 && (unsigned)sample_info_count >= rtps_max_batch_samples_dissected) {
+      /* Never write past the sample_info_flags/sample_info_length arrays,
+       * which are allocated with sample_info_max entries. When the
+       * preference is 0, sample_info_max is 1024 (see above); otherwise it
+       * equals the preference. Guarding on sample_info_max here (rather than
+       * on the preference) keeps the 0 case bounded too. */
+      if ((unsigned)sample_info_count >= (unsigned)sample_info_max) {
```

**Liveness**, read from the pinned source rather than from ancestry — the latter has called
backported fixes live before — and keyed on an identifier that exists on exactly one side:

read from the PINNED source, keyed on the identifier: v4.6.8:epan/dissectors/packet-rtps.c contains the pre-fix guard `rtps_max_batch_samples_dissected > 0 &&` 1 time and the fixed `(unsigned)sample_info_max` 0 times. Two-sided. LIVE AT THE PIN.

**A gate its own default disables.** `rtps_max_batch_samples_dissected` is a user preference whose
default is **0**, meaning "no limit". With it 0 the first clause is false, the comparison is never
reached, and nothing bounds `sample_info_count` — while the arrays are still only `sample_info_max`
entries long. This project has paid for this shape repeatedly in its own code; here it is upstream's,
and the row is worth having for the shape independently of the crossing.

## Why this is a NESTED row

wmem, a chunk the BLOCK or BLOCK_FAST allocator carved from a block g_malloc handed out. An inner allocator carved the crossed region, so under this inventory's axis -- WHO ALLOCATED THE OBJECT -- this row IS NESTED. A malloc-granular bound cannot see the crossing: the block wmem carved it
from is one `g_malloc`, and the access stays inside that block.

## What is real here, and what is reduced

**Real:** the allocator. Upstream's own wmem, through this corpus's seam, with the chunks carved
consecutively from the same block so the successor's position can be asserted.

**Reduced:** SAMPLE_INFO_MAX is reduced from 1024 to 8. The defect is that NOTHING bounds the counter, so the array's length changes how many iterations it takes to leave, not whether it leaves. The preference is kept at its real default of 0, because that value IS the defect, and the case asserts both that the pin's guard does not fire and that the fix's guard would.

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
