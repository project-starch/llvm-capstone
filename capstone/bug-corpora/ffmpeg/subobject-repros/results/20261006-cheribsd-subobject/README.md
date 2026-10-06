# The subobject corpus on CheriBSD purecap (2026-10-06)

**What this replaces.** All three rows carried `cheribsd-revocation` as a **prediction**,
justified by *"UNMEASURABLE ON THIS HOST, checked rather than assumed:
ports/common/cmake/toolchains/cheribsd.cmake:4-9 requires CHERI_SDK and CHERI_SYSROOT"*.
That justification is now **out of date**: the host has both, the buffer-pool port's
`cheribsd` preset configures and builds, and the fixtures cross-build against the purecap
`libffmpeg-pool.a`. So the prediction was measurable after all, and here it is measured.

Every number below comes from `result-lines.txt`, taken with **no pipe between the program
and `$?`** — a `cmd | tail` reports `tail`'s status, which is how the first attempt at this
read `RC=0` from a run that had faulted.

## With revocation off: the prediction holds, and is now a measurement

| case | control (`fixed`) | defect (`buggy`) |
|---|---|---|
| `01_68845e26f7_vulkan_hevc_refpicset_member` | VERDICT FIXED | **VERDICT DEFECT-REPRODUCED** — 8 bytes past `RefPicSetStCurrBefore[8]` into `RefPicSetStCurrAfter`, inside one refstruct allocation |
| `02_e058af88ab_vulkan_hevc_dpb_member` | VERDICT FIXED | **VERDICT DEFECT-REPRODUCED** — the DPB walk wrote `ref_src[16]`, which is `h265_refs[0]`, inside one refstruct allocation |

The control holds and the defect reproduces **on a purecap machine, with CHERI bounding
every allocation**. That is the row's claim, measured rather than predicted: a crossing
between two members of ONE allocation is in bounds for a per-allocation capability, so
CHERI does not catch it. Revocation is irrelevant here — no lifetime ends.

## `00_8864fd0aec_cbs_h265_pic_timing_member` does not reproduce on purecap

`revoff buggy RC=1 VERDICT INCONCLUSIVE`, with its control holding. The reduction is
ABI-dependent by construction: its `distinguishing` field turns on offset 1200 being
4-aligned and the two-byte write landing on the **low half** of the next member. Under
`l64pc128d` the struct does not lay out that way, so the assertion the case makes about
which half survives no longer describes what happens. **Not a defence result** — the case
does not exercise its defect on this ABI, and it is outside this arm's denominator.

## With revocation on, these cells are NOT scoreable

Every `revon` row is `RC=162` with no verdict line — **including the `fixed` control arm**.
A control that does not hold cannot carry a verdict, so no catch and no miss is recorded.

**It is the fixture, not the platform**, and that is attributed rather than assumed:
[`attribution-probe.c`](attribution-probe.c) — `malloc`, `memset`, `realloc`, `free`, built
with the **same** SDK, target, ABI and link mode and run in the same guest — returns
`ATTRIB-OK` and `RC=0` under both knob settings. So running under revocation is not what
breaks; something in the fixture or in `libffmpeg-pool` is. Diagnosing that is open work and
belongs to the port, not to this corpus.

## Setup

- Platform: the CheriBSD purecap image whose loader accepts dynamically linked purecap
  binaries; `security.cheri.runtime_revocation_default` is left at 0 and each run selects
  its own mode with `_RUNTIME_REVOCATION_DISABLE=1` / `_RUNTIME_REVOCATION_ENABLE=1`,
  because with the system default at 1 this guest's own `sshd` dies during a file copy.
- `libffmpeg-pool.a` built from the port's own `cheribsd` preset with the matched SDK; each
  fixture is `case.c` + `shared/driver.c` unchanged, linked against it.
- The `fixed` arm runs **first** for every cell, so an infrastructure failure shows up as a
  failed control rather than as a defect that quietly did not reproduce.
