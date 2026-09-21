<!-- The operating manual for the Capstone domain target. The corpus README
describes the corpus; this describes how to run it here. -->

# Running the corpus in a Capstone domain

One defect per boot, one program per defect. `build-cases.sh` invokes the
port's one-source seam once per case and leaves the programs as
`<out>/bin/defect-NN.dom`; the runner stages the one its case needs. A
capability fault ends the domain, so a case that provokes one cannot also
report results beside it.

    shared/build-cases.sh capstone-domain <out>
    python3 runners/capstone-domain/run-defects.py <results> \
      --domain-build <out> --linux-build <port>/build/linux-guest

and the control that makes the result mean something, which must exit 0:

    python3 runners/capstone-domain/run-defects.py <results> --negative-control

`--cases` takes a diagnostic subset and `--modes spatial,sublet` a single arm.
The port is `ports/apr/pools`; its `linux-guest` preset builds the loader.

## Two modes, one binary

Each case runs twice against the same `defect-NN.dom`, which picks its arm
from the mode argument:

| mode | authority | required outcome |
|---|---|---|
| `spatial` | a node keeps the alias it was carved with across APR's free list | the sequence COMPLETES |
| `sublet` | release and issue each `sublet_give` then `sublet_take` the node | the stale access FAULTS at the labelled probe |

The hooks sit on APR's own free-list transitions, not on `free()`: under the
default `APR_ALLOCATOR_MAX_FREE_UNLIMITED` upstream never calls `free()` on
the path that matters, so a hook there would never fire. See the port's
`patches/apr-1.7.4-0002-node-lifetime-hooks.patch`.

**The expected PC is not hardcoded.** The domain publishes both probe addresses
through its marker, and the oracle compares the fault PC against what that boot
printed, so a relink cannot silently turn the check into a tautology.

Two emulator behaviours are both accepted and recorded: without local trap
delivery the fault HALTS the domain and QEMU exits; with it the fault is
DELIVERED, the launcher dies by SIGSEGV and the VM survives. A surviving VM is
not treated as the weaker result.

A boot that produced no capture, or no result at all, exits 75 with no verdict
rather than recording a failure that would read like the defect not
reproducing.
