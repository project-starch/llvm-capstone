# CPython 3.13.7: four-arm inner reuse

All 12 processes pass the complete `objects.py 8 3 0` JSON/GC oracle:
three fresh CheriBSD processes per mode, plus the six previously archived
Capstone spatial/Sublet processes. These are application executions, not replay.
This input qualifies the full interpreter; it is not a pyperformance score.

| Arm | Processes | Issues/process | Observed reuse share | Median observed gap |
|---|---:|---:|---:|---:|
| Capstone spatial | 3/3 | 77,005 | 63.48% | 2–3 |
| Capstone + Sublet | 3/3 | 77,005 | 63.48% | 2–3 |
| CheriBSD spatial adapter | 3/3 | 80,194 | 65.13% | 2–3 |
| CheriBSD PoisonCap adapter | 3/3 | 80,272 | 55.77% | 4,096–8,191 |

Compare protection against its own platform's control. The two CheriBSD arms
use the same fresh `-O1` binary and kernel, with corrected libc and verified
process revocation **enabled**. Guest setup services retain a disabled default.
The prior outer-disabled CheriBSD control is superseded for this comparison.
The Capstone pair retains its declared 262,144-node capacity. All raw repeat
variation is preserved; histograms are pooled only when drawing the figure.

Two fixes made the protected arm executable:

- The [kernel probe patch](../../patches/cheribsd-poison-object-probe.patch)
  avoids re-entering user `vm_fault()` under the revoker's VM-map lock. It
  probes held physical pages or resolves their VM backing objects. Unsupported
  objects, partial pages and pager errors stop the run; they are never counted
  as an absence of poison. The separate regression checks resident poison,
  poison hidden by `mprotect(PROT_NONE)`, untouched zero-fill and an unmapped
  address. Both poisoned capabilities lose authority and the zero-fill one
  remains usable. Swap-pressure and general concurrent-VM stress remain untested.
- The CPython adapter now defers free-list publication until revocation
  completes. Pool occupancy prevents reclassification or release meanwhile.
  It transfers the published SQLite thresholds: 4,096 entries, or held spans
  at least 16 MiB with at least one quarter quarantined. Held spans are live
  rounded blocks plus quarantined blocks. Full queues revoke before draining.
  Reallocation can move a block instead of forcing an immediate sweep.
  Teardown drains precede destruction of callback owners.

Every protected process reports 19 sweeps: 18 capacity drains and one teardown,
zero percentage-triggered or compatibility-forced sweeps. Peak quarantined
payload is 1,864,144 bytes; the fixed queue itself occupies 196,608 bytes in
both modes and is included in reported adapter metadata. These component
counts do **not** establish a total-memory advantage. The existing `copied_bytes`
counter covers adapter snapshots, not all interpreter realloc copies.

The figure measures the conditional distribution of **observed same-start
reuses**, indexed by successful new lifetimes. It includes startup and shutdown.
It does not measure every failed allocation attempt, censored retirement
cohorts, physical working set, elapsed time, or general fragmentation.

Run `python3 validate.py` to reconstruct every histogram and check the output,
phase sequence, runtime policy, queue policy, final drainage and manifest
identity. The six new transcripts, runner and VM command/serial output are in
`cheribsd-raw.tar.gz`; private VM keys are excluded. The six Capstone transcripts
and build identity remain in `../cpython-reuse-three-arm-20260928/`.
`kernel-platform.patch.gz` preserves the complete tracked platform delta against
`kernel-source-identities.json`'s source revision, including pre-existing artifact
repairs. `kernel-config` records the configuration; RISC-V is the tested target.
`runtime-identities.json` records the kernel, SDK and corrected libc hashes.
