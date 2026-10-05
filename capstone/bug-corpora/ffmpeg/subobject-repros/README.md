# FFmpeg sub-object spatial defects: a bound crossed BETWEEN TWO MEMBERS of one allocation

A sibling of [`../pool-repros/`](../pool-repros/README.md), deliberately separate. That corpus's
boundary is storage an `AVBufferPool` handed out; these three defects cross a bound **inside a
single allocation**, between adjacent members of a struct, which is a different boundary and would
have been misdescribed by widening the other corpus's.

**All three are LIVE at the `n9.0.1` pin.** Triage:
[`docs/ref/ffmpeg-spatial-defect-triage.md`](../../../docs/ref/ffmpeg-spatial-defect-triage.md).

## What makes this class worth a corpus of its own

| shape | cases | caught by any arm we have | oracle used |
|---|---|:--:|---|
| array member written past its end into the next member of the same allocation | 0, 1, 2 | **0 / 3** | the upstream fix |

**Nothing we have catches any of them, and measuring that is the point.** Every per-allocation
bound — `shrink`, `sublet`, the pool arms, CHERI — is *in bounds* for a member-to-member crossing.
The authority that would have to be narrowed is **per struct member**, which means the compiler or
the allocation site, not an allocator. That is the taxonomy's `partial²` cell, where
`table6-cheri-vs-capstone-explained.md` already gives CHERI and Capstone the same verdict.

So the only oracle available is the **upstream fix**: the buggy arm's write crosses into the
neighbouring member and a sentinel there changes; the fixed arm's bound keeps every write inside its
own member. That is `native-fix-differential`, and it is the arm these rows are measured on.

## The three rows

| case | upstream | the crossing |
|---|---|---|
| **0** | `8864fd0aec` cbs_h265 pic_timing | `uint16_t num_nalus_in_du_minus1[600]` index 600 → the low half of `uint32_t du_cpb_removal_delay_increment_minus1[0]`. Offset 1200, 4-aligned. **Fully contained** |
| **1** | `68845e26f7` Vulkan HEVC ref sets | 8 bytes past `RefPicSetStCurrBefore[8]` → wholly into `RefPicSetStCurrAfter[8]`. **Fully contained: no magnitude escapes the allocation** |
| **2** | `e058af88ab` Vulkan HEVC DPB | `ref_src[16]` → `h265_refs[0]`. Contained **for the first crossing only**; far past it the walk leaves the struct, which is why the case is reduced to that first write |

## Measured, 2026-10-05

`runners/run-native.sh` **exit 0**: all three fixed arms print `VERDICT FIXED` and all three buggy
arms print `VERDICT DEFECT-REPRODUCED`. The runner runs the **control arm first** and treats its
failure as infrastructure rather than data, which is what caught the first version of case 2 — it
wrote all sixteen extra entries the `DPB[32]` walk permits, ran 128 bytes into a 64-byte member and
so left the allocation, and the containment check refused it. The reduction to the first crossing is
recorded in that case's `fidelity` rather than left implicit.

**ASan is blind to this class, measured two-sided.** `results/20261005-native-subobject/` holds a
probe with two arms over the same struct: the member-to-member crossing draws **no report** (exit 0,
and the neighbour is observably clobbered), while a write one `uint16_t` **past the allocation** —
the positive control — draws `heap-buffer-overflow` and exit 1. The first build of that probe had
*both* arms silent, because at `-O1` the dead stores were optimised away; that is why the control is
there at all, and why the arm is only believable with it.

The Capstone and CheriBSD arms are **declared completions, not measurements**: measuring them needs
a domain build this corpus does not have. Each `case.json` says so.

Files per case: `case.c`, `case.json`, `PROVENANCE.md`. Seam: `shared/` (a verbatim copy of
`pool-repros/shared/`, because each corpus owns a private copy and the `FF2_*` infrastructure macros
come from the port's `replay-config`).
