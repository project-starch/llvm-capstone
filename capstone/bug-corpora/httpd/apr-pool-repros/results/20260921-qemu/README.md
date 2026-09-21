# The httpd/APR case, paired arms, QEMU, 2026-09-21

**2/2 arms passed.** The one case this corpus has completes in the spatial arm
and faults at the labelled stale read in the Sublet arm, against one
`defect-00.dom` built through the port's `APRP_CORPUS_SRC` seam.

    matrix.tsv    the two result lines
    inputs.json   the sha256 of the binary, the loader, QEMU and the compiler,
                  the negative control, and the repetition matrix

`matrix.tsv`, not the serial logs. A capture is contaminated by construction —
kernel and driver banners carry account names — and about forty of its lines
matter.

## What was run

The arm is chosen at **runtime** from the loader's mode argument, so the two
arms differ in exactly one thing:

| mode | what the adapter does on APR's free-list transitions | expected | got |
|---|---|---|---|
| spatial (0) | a node keeps the alias it was carved with | completes | `completed=1`, `nodes=3`, `node_reuses=1`, no fault |
| sublet (1) | release and issue each `sublet_give` then `sublet_take` | faults at `apr_defect_read` | `cause=24`, `pc=0x101866a8c` = the published probe |

`node_reuses=1` in the spatial report is the mechanism itself, counted by the
adapter: the destroyed pool's node left the free list a second time, as the
other pool.

## The oracle, and that it can fail

The Sublet arm requires a fault at the **published** probe address, not merely
a fault. The domain prints its case marker (`0xcf1a…|case`) and both probe
addresses at the moment the stale read is about to happen; the runner reads the
expected PC out of that boot's own output and compares. The setup `CHECK` that
the same node came back as the other pool runs before the marker.

`--negative-control` corrupts the fixture so the case's `CHECK(700)` refuses it
before any pool is created. Both oracles fired: the spatial one because no
report completed, the sublet one because no fault occurred.

## Repetition, and two boots that produced nothing

Each of the four arms — positive and negative, spatial and sublet — was then
run eight times, 32 boots. `inputs.json` has the per-arm counts. 31 returned
`rc=0` with a result. The 32nd, a positive sublet arm, stalled before the
login prompt: the guest-boot flake recorded the same day against the pymalloc
corpus, one boot in roughly forty, before anything the corpus does is reached.
Separately, the first negative-sublet attempt produced no result with a
different signature — the guest went silent after issuing `run.sh` — and the
same arm then returned nine times of nine. Both are recorded as infrastructure
no-results (`exit 75`, no verdict), not as measurements, and neither
correlates with mode. Their cause is not established.

## What this does not show

Nothing about httpd: this corpus pins the allocator, not httpd's tree, and
`live_in_pin` stays `null` with the reason. Nothing about PoisonCap or CheriBSD:
there is no such build of APR, and those arms stay `"not written"`. One case,
one shape — a stale allocator *handle* rebound to a live pool, which the other
corpora do not have — not a survey.
