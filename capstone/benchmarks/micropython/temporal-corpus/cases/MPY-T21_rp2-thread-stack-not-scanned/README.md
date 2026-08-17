# MPY-T21: rp2: Script using _thread crashes without explicit GC call
Source: #8550, https://github.com/micropython/micropython/issues/8550  
Upstream state: closed, first seen 2022-04-14

**BLOCKED. Upstream status unresolved, see below.**

## The defect

thread stack memory collected because it is not a scanned root.

Class `premature-free`, CWE-416, in `ports/rp2 + py/gc.c`. Scope `port-heap`, so a second allocator is involved.

## What Capstone does about it

`traps_unmodified` = **unclear**. A second allocator is involved and its behaviour has not been examined.

`traps_if_gc_cap_aware` = trapped, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. Closed upstream as no-longer-reproducible rather than fixed, and it needs the rp2 port and threads.

## Reproducing

Not reproducible with the current setup: closed upstream as no-longer-reproducible rather than fixed, and it needs the rp2 port and threads.
