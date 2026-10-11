# In-process node reclamation in the supervised VM

At the original 65,536-node capacity, the updated QEMU passes all 27 application
repeats from the default-CheriBSD comparison, including its six failing mruby
attempts. Nine extended application runs also pass. Application binaries, inputs,
output oracles, launcher, kernel and firmware are unchanged. The original
comparison data remains a separate historical configuration; CheriBSD was not
rerun for this emulator fix.

## Cause and change

Revocation nodes are metadata identities, not application heap bytes and not
one-to-one with malloc calls. A freed allocation can leave invalid identities
that cannot be reused until old tagged references are removed. The existing
collector ran at process teardown. Long-running applications could therefore
consume the node pool despite freeing their heap allocations.

QEMU now pauses a supervised application before a node-allocating instruction
when only the 256-node cleanup reserve remains. It restores the trusted monitor
context, sweeps stale tags in memory, registers and paused continuations, recycles
invalid unpinned identities, then unwinds the helper. Resuming the application
retries the same instruction using its saved architectural state. It requires
no application cooperation or new monitor/driver ABI. If collection cannot
recover space, genuine exhaustion still produces a resource fault and cleanup
remains possible.

This is a **one-hart QEMU software tag sweep**. These results do not establish
FPGA reclamation behavior, scan-free collection, hardware latency or total-memory
superiority over CheriBSD. Live and pinned metadata must still fit in the pool.

## Checked outcomes

| Workload | Pass / attempts | Node allocations per process, including setup and teardown |
|---|---:|---:|
| FFmpeg, 30 frames, 1 / 4 / 16 streams | 9 / 9 | 1,279–1,280 / 5,071 / 20,239 |
| FFmpeg, 150 / 600 frames, 1 stream | 6 / 6 | 3,639 / 12,954 |
| mruby, 32 / 128 records, 8 batches, retained graph 256 | 6 / 6 | 9,812–9,813 / 26,016 |
| mruby, 512 records, 8 batches, retained graph 256 | 3 / 3, previously 0 / 3 | 95,295 |
| mruby, 128 records, 8 batches, retained graph 4,096 | 3 / 3, previously 0 / 3 | 76,409 |
| mruby, 512 records, 128 batches, retained graph 256 | 3 / 3 | 1,092,495 |
| mruby, 128 records, 128 batches, retained graph 4,096 | 3 / 3 | 169,769 |
| FFmpeg, 30 frames, 64 streams | 3 / 3 | 80,911 |

Each mruby run includes a fourfold burst halfway through its batches. Each
FFmpeg stream is independently decoded and checked against its full frame-hash
oracle. Both matrices and the lifecycle gates share one unchanged Linux boot.
The node high-water mark remains 65,285, including monitor cleanup allocations.
All 36 application attempts end with zero live domains, regions, bytes, retired
nodes and poisoned blocks. Observed requested bytes also return to zero at exit.
Cached physical storage remains retained by the monitor; this is not returned
Linux RAM. Full reservations and measurement limits remain in the
[comparison contract](../../../../../experiments/applications/comparison.md).

The old emulator fails the new 200,000-cycle churn fixture by SIGSEGV after
65,241 additional node allocations. The fixed emulator passes that fixture and
a separate post-reuse stale-reference control, each with 200,064 additional
allocations. Live data remains intact across collection. A distinct fixture
creates valid ancestors that cannot be collected: it faults after 65,222
allocations, and a subsequent healthy process passes. The gate checks allocation
progress so an earlier unrelated fault cannot satisfy these controls.

The complete lifecycle gate passes **1,008 mixed starts** after these controls,
with stable cached bytes, live nodes and tag pages. A shorter rerun adds the
explicit allocation-progress assertions. Four native runtime tests and twelve
host Python tests pass. These are runtime correctness checks, not measurements
of security strength.

## Evidence and reproduction

- [Platform and binary identities](provenance.json), including the old failing control.
- [Application results](applications.json): all 36 attempts, counters and exit metrics.
- [Full lifecycle gate](lifecycle.json) and [allocation-progress gate](lifecycle-progress.json).
- [External raw artifact identity](archive.json): logs, per-phase measurements,
  workload matrices, source changes, fixtures and the tested emulator. Guest
  credentials are excluded. Host paths in raw captures are not committed.

LLVM branch: `runtime-node-reuse`, based on `65a70550cdb2`.
QEMU branch: `runtime-node-reuse`, commit `22aec7ee0fe1`, based on `6550f19489e0`.
Use the parent's pinned QEMU with the existing managed application platform.
Build the Sublet fixture and Linux supervisor as described in
[application verification](../../../../applications.md#verification), then run:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/runtime/tests/application/run.py \
  --state "$CAPSTONE_TMP_ROOT/dev-vm" \
  --sublet-image /mnt/host/contract-sublet.dom --repeat 200 \
  --report "$CAPSTONE_TMP_ROOT/node-reuse-acceptance.json"
```

Application reruns use the existing `experiments/applications/run.py`, three
repetitions, the original workload matrix and a separate extended matrix. Both
point files and raw manifests are in the artifact; adjust their image paths to
the local shared directory. The extended matrix uses a 300-second timeout.
