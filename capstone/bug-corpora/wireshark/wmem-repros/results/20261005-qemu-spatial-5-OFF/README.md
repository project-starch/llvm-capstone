# The same five spatial rows on the region-granular build

**Build: `WM_CHUNKS=OFF`**, judged against the `sublet` arm. Read
[`../20261005-qemu-spatial-5-ON/`](../20261005-qemu-spatial-5-ON/README.md) first — it carries the
full account, including the refuted load-versus-store prediction and why both builds fault.

**Verdict: 12 of 12 arms as predicted, runner exit 0.** `matrix.tsv` has the rows.

The point of running this build at all: it is the arm that was *supposed* to complete. It faults
identically, which is what established that `wm_narrow()` bounds every wmem allocation regardless of
`WM_CHUNKS`, and therefore that **this harness has no malloc-granular arm to contrast against.** A
control that refutes the experiment's premise is the most useful kind.

Case 11 is the temporal regression control: `spatial` completes, the protected arm faults with
**cause 24**, unchanged by the two runner fixes that 59c0f94b3a01 landed (the hardcoded cause pair, and the assumption that the unprotected mode always completes). This round needed no further runner change.

QEMU only. **N = 1 per cell.** Files: `matrix.tsv`, `inputs.json`.
