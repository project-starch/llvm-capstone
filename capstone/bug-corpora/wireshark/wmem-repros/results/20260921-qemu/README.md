# Thirteen cases, paired arms, 2026-09-21

**26/26 arms passed** in one run: every `spatial` arm completed, twelve
`sublet` arms faulted with cause 24 at the labelled read probe, and the
thirteenth — case 12, the recorded non-detection — completed as its oracle
requires.

    matrix.tsv    one line per arm: verdict, cause, faulting PC, expected PC, site
    inputs.json   the image hash of every case, the loader, emulator and compiler

`../20260921-qemu-negative-control/` is the same suite with the input record
corrupted: all four selected oracles reported FAIL, as they must.

## How it was produced

    cmake --preset capstone-domain -B .../corpus-domain -DWM_CORPUS_DIR=<this corpus>
    run-defects.py OUT --domain-build .../corpus-domain --linux-build .../linux-guest

QEMU `ce93cb32…`, revocation node pool 65,536 (the emulator's compiled
`CAP_REV_TREE_SIZE`), the Capstone clang built from `f7b50f08…`; the binary
hashes of emulator, compiler, loader and every image are in `inputs.json`.
Raw serial logs of every attempt are archived outside the repository at
`~/artifacts/wireshark/20260921-wmem-corpus/raw-campaign.tar.gz`.

## The attempts before this one

This record is the third run of the suite; the two before it are retained in
the raw archive.

The first passed 22 of 26 arms. Three `sublet` arms — cases 0, 9 and 11 —
faulted with cause 24 *before* their ready marker: each computed the report's
field offset (`+56`, `+1`, `+16`) from the stale pointer after the reset, and
on Capstone the capability arithmetic (`cincoffsetimm`) on revoked authority
faults before any load. The cases now read through the pointer the holder
returns, and their provenance says so. The fourth failure, case 10's
`spatial` arm, exited 75 with a serial log that never reached the guest's
login prompt: a boot-side infrastructure failure, not a measurement.

The second passed 26 of 26 with those corrections. The port's upstream patch
then gained one more guarded hook — the recycler's individual free now
retires the chunk's storage through `wm_release`, a no-op for Capstone and
the point where the PoisonCap backend acts — so every image changed and the
suite was run a third time. Same 26 of 26; this is that run.

## What this does and does not establish

Twelve reported Wireshark use-after-free defects, reduced to the allocator
calls they make against the real `wmem` allocators, are revoked reads under
Sublet and silent reads without it. The thirteenth shows the mechanism's
boundary on a reported defect: an individual free into the recycler ends no
epoch. Nothing here runs a dissector, and every liveness claim is in the
case's own `case.json`.
