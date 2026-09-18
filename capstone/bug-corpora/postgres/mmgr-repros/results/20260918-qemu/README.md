# Eight PostgreSQL memory-context defects under Capstone, paired arms

`matrix.tsv`: **16/16 arms passed.** Eight upstream consumer defects, each live
in the pinned 17.0 release, run twice against the same logic.

- **spatial** — this port's unprotected baseline. Chunks are offsets inside one
  arena capability, so the stale access succeeds. All eight **complete**.
- **sublet** — every chunk is its own capability, revoked on free. All eight
  **fault**, cause 24.

For cases 1–7 the runner requires the fault PC to equal the address the domain
published for its labelled probe instruction, so the result is "faulted at *this*
access", not "crashed somewhere". Case 0 is the exception and declares it: its
stale access is a second `pfree`, so the manager faults reading a revoked chunk
header inside `GetMemoryChunkMethodID` before any bookkeeping runs. Its oracle
accepts a fault anywhere and is the only one that does.

The claim these pairs support: **bounds and provenance alone catch none of the
eight; revocation catches all eight.**

The spatial arm is this port's unprotected baseline, **not** a CHERI model —
CHERI narrows bounds per allocation and this arm does not. That difference is
irrelevant to these eight, because a reused chunk stays tagged and in bounds
under either, but the argument about CHERI is made in prose and not by this arm.

## What is real and what is reduced

Real: PostgreSQL 17.0's `aset.c`, `mcxt.c` and `slab.c`, compiled unmodified but
for the port's capability-ABI and Sublet patches; the contexts created with the
upstream parameters; the allocation and free calls themselves.

Reduced: the consumers, to the allocator call sequence each upstream defect
makes, in the same order. Reaching the originals needs a backend, a planner, a
walsender or a concurrently dropped partition, none of which changes what the
allocator does. Each case's `PROVENANCE.md` states the split line by line.

## Cost

The whole matrix runs in about **70 seconds** — roughly 3.5 s per arm, measured.
One fault per boot is not a harness choice: a capability fault in a domain
terminates the emulator, because there is no receiver for it. `defects.c`
therefore selects its case externally through `selection.bin`, as the port's own
fixtures do. Fault delivery (PR #52) changes that property, but at this size it
would save seconds, not minutes; its value is the containment claim, not speed.

## Reproduce

Build the domains through the port's seam, then run the suite:

    cmake --preset capstone-domain \
      -DPG_CORPUS_SRC=<repo>/capstone/bug-corpora/postgres/mmgr-repros/shared/defects.c
    cmake --build --preset capstone-domain --target defects-spatial defects-sublet
    python3 shared/run-defects.py <out> --domain-build <...> --linux-build <...>

`inputs.json` fingerprints the domain, loader, compiler, emulator and region
geometry. Raw serial captures stay local; only result lines are committed.
