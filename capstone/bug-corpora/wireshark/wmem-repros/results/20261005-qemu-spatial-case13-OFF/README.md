# Case 13 on the region-granular build — the control that showed the harness has no malloc-granular arm

**Build: `WM_CHUNKS=OFF`**, judged against the `sublet` arm. The primary bundle, with the full
account of what this run refuted and the two runner fixes it needed, is
[`../20261005-qemu-spatial-case13-ON/`](../20261005-qemu-spatial-case13-ON/README.md) — read that
first; this README only records the control's own numbers.

**Verdict: 4 of 4 arms as predicted, runner exit 0.**

| case | mode | arm | expected | passed | cause | pc |
|---|---|---|---|---|---|---|
| 11 `11-http-header-map` | spatial | `spatial` | complete | ✅ | — | — |
| 11 `11-http-header-map` | sublet | `sublet` | fault | ✅ | 24 | `0x101901424` |
| 13 `13-http-range-cursor-past-chunk` | spatial | `spatial` | fault | ✅ | 5 | `0x1019013f8` |
| 13 `13-http-range-cursor-past-chunk` | sublet | `sublet` | fault | ✅ | 5 | `0x1019013f8` |

**Why this build was run at all.** Case 13 was filed claiming the chunk port was its discriminator,
so the region-granular build was supposed to be the arm that *completes*. It faults too, with the
same cause 5 at the same labelled probe — which is what established that
`src/shared/wmem-port-hooks.h:11-15` (`wm_narrow()`) bounds every wmem allocation regardless of
`WM_CHUNKS`, and therefore that **this harness has no malloc-granular arm to contrast against.**
A control that refutes the experiment's premise is the most useful kind, and it is why this bundle
is kept rather than discarded.

Case 11 is the temporal regression control: `spatial` completes and the protected arm faults with
**cause 24**, unchanged.

QEMU only. **N = 1 per cell.** Files: `matrix.tsv`, `inputs.json`.
