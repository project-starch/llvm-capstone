# Case 13, the corpus's first SPATIAL row, measured — and it REFUTED its own predictions

**Build: `WM_CHUNKS=ON`** (chunk-granular), judged against the `sublet-chunks` arm. The sibling
bundle `../20261005-qemu-spatial-case13-OFF/` is the same two cases on the `WM_CHUNKS=OFF`
(region-granular) build.

**Verdict: 4 of 4 arms as predicted, runner exit 0 — after the predictions were corrected by the
run itself.**

| case | mode | arm | expected | passed | cause | pc |
|---|---|---|---|---|---|---|
| 11 `http-header-map` (temporal control) | spatial | `spatial` | complete | ✅ | — | — |
| 11 | sublet | `sublet-chunks` | fault | ✅ | **24** | `0x101902258` |
| **13 `http-range-cursor-past-chunk`** | spatial | `spatial` | fault | ✅ | **5** | `0x10190222c` |
| **13** | sublet | `sublet-chunks` | fault | ✅ | **5** | `0x10190222c` |

Case 11 is in the run as a **regression control**, not for its own sake: it is a temporal row, and
it still reads `spatial` completing and `sublet-chunks` faulting with **cause 24**, unchanged by the
two runner fixes below. Case 13 reads **cause 5** — a bounds fault — on both arms, at the labelled
read probe, with the pc equal to the probe address the run resolved from the image.

## What the run refuted

Case 13 was filed predicting that `spatial` and `sublet` would **complete** and only
`sublet-chunks` would fault, i.e. that it discriminated the chunk port. **Both predictions were
wrong, and the reason is one line of the port:** `src/shared/wmem-port-hooks.h:11-15`

```c
static inline void *wm_narrow(void *p, size_t n) {
#if defined(WM_DOMAIN)
  uintptr_t base = (uintptr_t)p;
  return __builtin_capstone_cap_shrink(p, base, base + n);
```

narrows **every** wmem allocation to its request under `WM_DOMAIN`, unconditionally and whatever
`WM_CHUNKS` is set to. So **no arm of this harness has malloc-granular bounds**, the 6-byte object
is bounded to 6 bytes everywhere, and a read at `+8` faults on all of them. The first register dump
said so plainly before the source did: `x10 = C(e1a00268 [e1a00260,e1a00266) type 1)` — a 6-byte
bound with the cursor two bytes past it.

**Consequence, stated so it is not mis-cited:** this case proves the defect is real and that
per-allocation bounds catch it. It does **not** demonstrate "a nested allocator hides the extent
from `malloc`". That contrast needs an arm whose bounds really are malloc-granular, and it lives in
the tshark **app** port, whose measured fx12 length ladder is `level0` 41 908 912, `shrink`
8 388 560, `sublet` 1 048 528, `chunks` 64.

## Two instrument defects were fixed first, both negative-tested

Neither was about this case; both would have reported a correct reading as a failure.

1. **`run-defects.py` hardcoded `cause in (24, 25)`** — the revoked-authority pair. Right for a
   temporal row, wrong for a spatial one: case 13 faults with cause **5**, so the gate would have
   failed it. The expected causes now come from the case's own arm (`cause` key), defaulting to the
   temporal pair when absent. Checked against every measured row in `results/`: all 14 rows
   carrying a cause are still accepted, so the tightening regresses nothing.
2. **`classify()` assumed the unprotected `spatial` mode always completes.** For a spatial defect it
   faults. Both modes are now judged by their declared arm (`arm_name()`), and the regression check
   is in this bundle: cases 0-12's `spatial` arms are still judged as completions and their
   protected arms as faults.

## The negative control, run and passed

`--negative-control` on the same two cases and both modes: **4/4 oracles reported FAIL as they
must**, exit 0. So the 4/4 above is not vacuous — every oracle in this run, including both of case
13's new ones, is proven able to fail.

## Provenance

- **Corpus:** `bug-corpora/wireshark/wmem-repros`, case 13 added 2026-10-05, triaged in
  `docs/ref/wireshark-spatial-defect-triage.md`.
- **Upstream:** `0261fd7da6`, *"http: Fix buffer overflow, use after free in HTTP Range"*. Only the
  buffer overflow is reduced. **Not live at the `v4.6.8` pin** — a fix-reversal, like every
  memcached and FFmpeg corpus case.
- QEMU only. **N = 1 per cell.** No board run; none is needed for this claim, and cause 5 under
  capstone-qemu says nothing about the deployed silicon.

Files: `matrix.tsv`, `inputs.json`.
