# The native fix-differential arm on all four cases, case 3 included — and the instrument defect it exposed (2026-10-03)

**Question.** Case 3 (`03_5c66a3ab51_*`, the corpus's first `AVRefStructPool` case) was declared on
2026-10-03 with its predictions registered and **no arm run**. Does it reproduce? And do cases 0-2 still
reproduce after `shared/driver.c` and `shared/corpus.h` were extended to create the refstruct pool,
which recompiles them?

**Verdict: yes to both. `runners/run-native.sh` exits 0 — four `DEFECT-REPRODUCED`, four `FIXED`.**

| case | allocator | fixed arm | buggy arm |
|---|---|---|---|
| 00 `af_join_dedup_bound` | AVBufferPool | `FIXED` output holds a reference | `DEFECT-REPRODUCED` `reuse_same_address=1 stale_read=0xCC` |
| 01 `h264_refs_partial_clear` | AVBufferPool | `FIXED` reset covers the whole list | `DEFECT-REPRODUCED` `survivors_past_len=2 reuse_same_address=1` |
| 02 `vidstab_parked_plane_pointer` | AVBufferPool | `FIXED` nothing carried across frames | `DEFECT-REPRODUCED` `wrote_through_stale=1` |
| **03 `vvc_nonref_output_releases_tabs`** | **AVRefStructPool** | `FIXED` `tables_released=0 stale_read=0xB0` | **`DEFECT-REPRODUCED`** `tables_released=1 reuse_same_address=1 reissued_is_rpl_tab=1 stale_read=0xCC` |

Cases 0-2 are unchanged from their previous native runs, so extending the shared driver did not disturb
them — which is the reason they were re-run rather than assumed.

## The part worth keeping: case 3 failed first, and the allocator was right

The first run returned **`CONTROL-FAILED 03_… buggy arm rc=1`** with
`tables_released=1 reuse_same_address=0 stale_read=0xA0`. The release had happened; the pool had simply
not handed back *that* entry.

**Cause, read from the allocator rather than guessed.** The free list is **LIFO**:

    pool_return_entry()      ref->opaque.nc = pool->available_entries;
                             pool->available_entries = ref;          /* push onto the HEAD */
    refstruct_pool_get_ext() RefCount *ref = pool->available_entries; /* pop the HEAD */
                             pool->available_entries = ref->opaque.nc;

`ff_vvc_unref_frame` releases `tab_dmvr_mvf` **first** and `rpl_tab` **second**, so the entry the next
`av_refstruct_pool_get` returns is **`rpl_tab`**. The reduction tracked only `tab_dmvr_mvf`, so it
compared the reissued entry against the wrong one of the two it had released.

**That is an instrument defect, not a refutation.** Nothing about the case's premise changed: both side
tables are the decoder's, both are released by the same branch, and either one coming back proves the
reuse. The fix tracks both and names which came back (`reissued_is_rpl_tab=1`), so the oracle can no
longer be satisfied or defeated by release order.

**A second, smaller instrument fix went in with it.** Once `stale` could point at `rpl_tab`, the fixed
arm printed `stale_read=0x00` — because only `tab_dmvr_mvf` had been filled. A zero there cannot be told
apart from uninitialised memory, and reads like a finding. `rpl_tab` now gets its own `0xB0` fill with a
live control (`CHECK(frame.rpl_tab[0] == 0xB0, 609)`), so the fixed arm reports a known value and the
buggy arm's `0xCC` is contrasted against one for **both** tables. No verdict changed, and the fault
lines moved again, `99,100,103` → `104,105,108`.

**Why it matters beyond this case.** `reuse_same_address=0` is exactly what a *correct* allocator and a
*nonexistent* defect both look like. Had case 3 been built only for the capability arm, this would have
surfaced as a clean, monotone, entirely void domain result — a pooled-reuse case that never created its
triggering condition, which is the failure CLAUDE.md names as the most expensive class here. The native
arm cost one minute and caught it.

**Re-registration, with the prediction untouched.** `runners/sublet-port-expect.txt` now reads
`poolsublet 46 FAULT 104,105,108` instead of `84,85,88`. Only the **line table** moved; the predicted
outcomes (`poolsublet` faults on 46, `poolstock` completes, both complete on 47) are unchanged, and the
reason is recorded in the expect file itself rather than only here.

## What this does and does not say

- **It does** give case 3 its first measured arm, and it is the arm `case.json` declares as
  `native-fix-differential`.
- **It does** confirm the reuse happens inside the pool: no `free()` is involved, which is what makes the
  class invisible to a free-keyed tool — measured separately the same day in
  [`../2026-10-03-native-asan-pool-blindness/`](../2026-10-03-native-asan-pool-blindness/README.md).
- **It does not** measure any capability arm. ~~The `poolsublet`/`poolstock` rows (fixtures 46/47) remain
  predictions. They need the FFmpeg **app** port, whose SDK gate **correctly refuses both toolchains
  available on this host** … That is a tried-and-recorded blocker, not an assumption.~~
  **CORRECTED 2026-10-04 (`5208789e4b9e`): the blocker claim is withdrawn.** Two toolchains do fail the
  gate, but a third, qualified one is present — the `llvm-capstone-cc` sibling worktree's
  `build-release`, commit `7d01722aab88` — and `check_toolchain` accepts it end to end. All sixteen
  corpus images (40-47, both arms) were built through that gate with it on 2026-10-04. So 46/47 remain
  predictions **only because nobody had run them**, and the emphatic "not an assumption" framing was the
  worst part of the original wording: it invited trust in a conclusion drawn from a partial enumeration.
  Case 3's own capability pair was measured the next day as probe case 39
  (`../../../../ports/ffmpeg/buffer-pool/results/measurements/20261004-vvc-case3-probe39/`).
- **It does not** make case 3 live at the pin. `5c66a3ab51` is an ancestor of `n9.0.1`, so the case
  re-introduces the reverse of the fix, as all four in this corpus do.
- **N = 1 per arm.** These are deterministic: no timing, no concurrency, no allocator nondeterminism.

Files: `result-lines.txt` — every line above, plus the build provenance.
