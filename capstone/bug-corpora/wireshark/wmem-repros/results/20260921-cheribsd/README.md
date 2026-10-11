# Thirteen cases on CheriBSD: the guest's own revocation, and PoisonCap, 2026-09-21

Three arms on the PoisonCap platform, guest libc revocation **on** for all
three (the ABI probe reported `runtime_revocation=1` in both boots), with the
local libc fix the CPython corpus documents (`libc 6726fdb0…`).

| arm | result |
|---|---|
| CheriBSD default, plain build | **0 / 13 caught**: every case completed |
| PoisonCap mode 0 | 13 / 13 completed: the matched control |
| PoisonCap mode 1 | **13 / 13 SIGPROT at the labelled read probe**, including case 12 |

    matrix.tsv    one line per arm: verdict, exit, fault PC, probe address, at-probe, adapter counters
    inputs.json   the plain and PoisonCap suite records: binary hashes, platform hashes, revocation settings

Every protected fault is signal 34, code 2, at the address `supervise`
resolved for `wm_defect_probe` from the child's own map and ELF. The
`counters` column is the adapter's own line, printed at the ready marker
(the protected arm does not survive the access after it), and it says which
hook retired the stale object:

| protected arms | counters at the marker | the hook |
|---|---|---|
| cases 0–11 | `sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0` | the packet pool's reset poisoned its 2 MiB block |
| case 12 | `sweeps=1 poison_bytes=16 epochs=0 released_chunks=1` | the recycler's individual `wmem_free` poisoned the 16-byte chunk |

Case 12 is the difference to Sublet in one number: no epoch ended, one chunk
was released, and the registry's stale name was dead by the time the next
packet read it. Every mode-0 arm reports `sweeps=0`.

This is the second PoisonCap run of the day. The first produced the same 26
verdicts, but its driver printed the counters only at exit, so the protected
arms had none; the driver now prints them at the marker, and the first run is
retained in the raw archive. The plain arm is the day's only run.

## How it was produced

    host/cheribsd/poisoncap/build.sh BUILD-plain --corpus <this corpus>
    host/cheribsd/poisoncap/build.sh BUILD-poison --poisoncap --corpus <this corpus>
    host/cheribsd/poisoncap/run.py BUILD-plain  OUT-plain  --modes 0   --runtime-revocation on ...
    host/cheribsd/poisoncap/run.py BUILD-poison OUT-poison --modes 0,1 --runtime-revocation on ...

Raw guest logs are archived outside the repository at
`~/artifacts/wireshark/20260921-wmem-corpus/raw-campaign.tar.gz`.

## What this does and does not establish

The plain arm's completions are informative only because the same programs
fault under PoisonCap mode 1 and the bounds control faulted in the same boot:
the probes reach the stale reads, and the guest can observe a fault. What
libc's quarantine and revoker never see is an event, because wmem hands
storage back to libc neither at a reset nor at an individual free.

PoisonCap's 13 of 13 rests on the adapter's synchronous sweep per
invalidation, a deliberately conservative policy measured in an emulator; the
counters in `matrix.tsv` are adapter measurements, not a PoisonCap overhead
figure, and must not be quoted as one.
