# FFmpeg sub-object spatial defects: a bound crossed INSIDE ONE ALLOCATION

A sibling of [`../pool-repros/`](../pool-repros/README.md), deliberately separate. That corpus's
boundary is storage an `AVBufferPool` handed out; these ten defects cross a bound **inside a
single allocation** — between adjacent members of a struct, or between two slices carved from one
block by arithmetic — which is a different boundary and would have been misdescribed by widening the
other corpus's.

**Cases 0, 1, 2, 5 and 6 are LIVE at the `n9.0.1` pin; 3, 4, 7, 8 and 9 are fix-reversals.**
Liveness is **recorded, never required** — the convention is at
[`../../memcached/allocator-repros/README.md:132-135`](../../memcached/allocator-repros/README.md),
and treating it as a requirement is what kept this corpus at three cases until 2026-10-06. Every
liveness reading here was taken from the **pinned source**, two-sided (vulnerable construct present,
fix marker absent, or the reverse), not from `git merge-base --is-ancestor`, which has called
backported fixes live before. Triage and the full candidate disposition:
[`docs/ref/ffmpeg-spatial-defect-triage.md`](../../../docs/ref/ffmpeg-spatial-defect-triage.md).

**Under this inventory's axis — *who allocated the object* — every row here is NOT NESTED.** The
object is one allocation and nothing sub-allocated the crossed region. Case 9 is the row most likely
to be mis-filed: its bound is a slice carved by `base + i*stride`, which is a *sub-object*, not an
inner allocator's sub-allocation. The tally is per corpus, so that sentence is load-bearing.

## What makes this class worth a corpus of its own

| shape | cases | caught by any arm we have | oracle used |
|---|---|:--:|---|
| array member written one element past its end into the next member | 0, 1, 2, 5 | **0 / 4** | the upstream fix |
| array member **read** one element past its end into the next member | 3 | **0 / 1** | the upstream fix |
| index **underflows** out of the START of a member into the one before it | 4 | **0 / 1** | the upstream fix |
| an **unbounded loop** walks off a member's end into the next member | 6, 8 | **0 / 2** | the upstream fix |
| a copy sized by its **source** length overruns the destination member | 7 | **0 / 1** | the upstream fix |
| a **carved sub-slice** crossed by a data-controlled index | 9 | **0 / 1** | the upstream fix |

**Nothing we have catches any of them, and measuring that is the point.** Every per-allocation
bound — `shrink`, `sublet`, the pool arms, CHERI — is *in bounds* for a crossing interior to one
allocation. The authority that would have to be narrowed is **per struct member**, or per carved
slice, which means the compiler or the allocation site, not an allocator. That is the taxonomy's
`partial²` cell, where `table6-cheri-vs-capstone-explained.md` already gives CHERI and Capstone the
same verdict.

So the only oracle available is the **upstream fix**: the buggy arm's access crosses into the
neighbouring member and a sentinel there changes; the fixed arm's bound keeps every access inside its
own member. That is `native-fix-differential`, and it is the arm these rows are measured on.

**What is measured, and what is declared.** Cases 0-2 carry a Capstone reading from probe cases
40-42 of `ports/ffmpeg/buffer-pool/security-tests`, run under QEMU on modes 0 and 2. Cases 3-9 are
measured on `native-fix-differential` and `native-detect`; their Capstone and PoisonCap arms are
**declared predictions**, recorded before the runs so a refutation stays visible.

**`cheribsd-revocation` is MEASURED for all ten, 2026-10-06** (`results/20261006-cheribsd/`):
**0 of 10 caught**, 22 of 22 arms, runner exit 0, with `cheribsd-abi` (reporting
`CHERI_ABI pointer_bytes=16 runtime_revocation=1`) and `cheribsd-bounds` both firing in the same boot. Each case prints its own verdict, so a completion is a reading rather than a silent
pass. Set beside the sibling `../plain-heap-repros/`, where **3 of 4 were caught**, this is the whole
contrast measured on one platform in one day: a crossing that leaves the usable allocation is caught,
one interior to it is not.

**Two things that reading must carry.** First, the reason is **weaker** than "CHERI bounds the
allocation and the crossing stays inside" — see the arena caveat below; nothing bounded anything at
this granularity on that harness. Second, **`-O0` is load-bearing**: a first suite built `-O1`
returned `VERDICT INCONCLUSIVE` for case 0 while the other nineteen arms passed and both controls
fired, because case 0's index is a compile-time constant one past its member and clang folded the
store away. Rebuilt at `-O0` with nothing else changed it reads `DEFECT-REPRODUCED`.

**This is compiler-specific, not platform-specific, and the distinction matters for anyone
re-running the native arms.** gcc at `-O1` keeps the store — the committed native readings in
`results/20261006-native-subobject-7/result-lines.txt` were captured with gcc at `-O1` and are
correct as they stand. clang at `-O1` folds it. So a native re-run with `CC=clang` at `-O1` will
disagree with the committed file on case 0, and that disagreement is the compiler, not a
regression. The purecap builds use clang because that is the only CHERI compiler, which is why
this surfaced on the CheriBSD arm first.

**One caveat the Capstone and CheriBSD predictions must carry.** This corpus's driver hands the
port's `av_malloc` *one* arena, and `ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26`
carves it by bumping a cursor, rounding every request up to 64 bytes;
`__builtin_capstone_cap_shrink` appears only in `src/capstone-domain/payload-capabilities.c:57`, on
pool *payload* blocks. So on this harness there is **no per-allocation bound on the struct at all**,
and a completion is weaker evidence than "the bound is the whole allocation" would be. The
sub-object claim does not rest on it — each case asserts its offsets — but the arm must not be read
as a measured per-allocation bound.

## The original three rows

Cases 3-9 were added on 2026-10-06; their shapes are in the table above and their
measurements in `results/20261006-native-subobject-7/`. The three below are the corpus's first,
kept with their own detail because the reductions they record are referenced elsewhere.


| case | upstream | the crossing |
|---|---|---|
| **0** | `8864fd0aec` cbs_h265 pic_timing | `uint16_t num_nalus_in_du_minus1[600]` index 600 → the low half of `uint32_t du_cpb_removal_delay_increment_minus1[0]`. Offset 1200, 4-aligned. **Fully contained** |
| **1** | `68845e26f7` Vulkan HEVC ref sets | 8 bytes past `RefPicSetStCurrBefore[8]` → wholly into `RefPicSetStCurrAfter[8]`. **Fully contained: no magnitude escapes the allocation** |
| **2** | `e058af88ab` Vulkan HEVC DPB | `ref_src[16]` → `h265_refs[0]`. Contained **for the first crossing only**; far past it the walk leaves the struct, which is why the case is reduced to that first write |

## Measured, 2026-10-05 — the original three

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
