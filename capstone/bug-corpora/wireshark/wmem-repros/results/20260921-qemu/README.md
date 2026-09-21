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
hashes of emulator, compiler, loader and every image are in `inputs.json`. Raw serial logs of both attempts are archived outside the
repository at `~/artifacts/wireshark/20260921-wmem-corpus/raw-campaign.tar.gz`.

## The attempt before this one

A first run of the same thirteen images passed 22 of 26 arms. Three `sublet`
arms — cases 0, 9 and 11 — faulted with cause 24 *before* their ready marker:
each computed the report's field offset (`+56`, `+1`, `+16`) from the stale
pointer after the reset, and on Capstone the capability arithmetic
(`cincoffsetimm`) on revoked authority faults before any load. The cases now
read through the pointer the holder returns, and their provenance says so.
The fourth failure, case 10's `spatial` arm, exited 75 with a serial log that
never reached the guest's login prompt: a boot-side infrastructure failure,
not a measurement, and it passed unchanged in this run. The first attempt is
retained in the raw archive.

## What this does and does not establish

Twelve reported Wireshark use-after-free defects, reduced to the allocator
calls they make against the real `wmem` allocators, are revoked reads under
Sublet and silent reads without it. The thirteenth shows the mechanism's
boundary on a reported defect: an individual free into the recycler ends no
epoch. Nothing here runs a dissector, and every liveness claim is in the
case's own `case.json`.
