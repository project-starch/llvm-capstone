# A1: aliases retained across ancestor revocation

This synthetic fixture uses the replay harness and Capstone runtime. It does
not replay a decoder recording or port an additional allocator. A parent owns
4,096 bytes and delegates a 64-byte child region. A separate 1,024-byte sibling
region belongs to neither the parent nor the child subtree.

Before revocation the child alias is copied into a global, a heap object,
a linked-list tail, sibling-region storage and a register. Every copy is
readable before the transition. The parent then revokes its senior handle;
the child returns nothing and does not clear or revoke its copies or handle.
Heap/list holders come from the separate metadata arena; globals live in the
domain image. Canary and sibling accesses check unaffected storage.

The matrix tests every storage location with a read and a write:

* Immediately after ancestor revocation, before initialization or reuse.
* After reuse at the child's exact old address. Memory-holder cases initialize
  the recovered parent and issue a new 64-byte child grant. The leaf assembly
  fixture reads and writes through recovered parent authority at that address,
  keeping the old child alias continuously in a register.
* A matched no-revoke arm runs the same access sites. Its reuse phase overwrites
  the old address through still-valid authority, so stale accesses succeed.
  This is an oracle control, not a CHERI emulation.

There are 20 protected stale-access cases, 20 no-revoke counterparts, and four
valid-control executions (two phases, two modes). Each runs in a fresh QEMU
instance. Expected faults require a completed setup marker, a capability-fault
cause and the exact labelled instruction PC. A register-only marker additionally
proves that its transition and sibling/fresh-access checks completed.
Timeouts, arbitrary crashes and setup assertion failures are not successes.

## Running

Build the domain and Linux loader as described in the component README.
Source `capstone/tests/capstone-test-env.sh` and set the build-directory overrides
if using custom builds. From this component directory:

```sh
bash security-tests/qemu/run.sh /tmp/capstone/a1-new-run \
  --cases 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --modes 0,2 --rounds 1
```

Case 14 is the valid reuse control; 35 is the valid immediate control.
For other cases, `15 + phase*10 + location*2 + write` selects phase 0/1,
location global/heap/list/sibling/register (0–4), and read/write (0/1).
Mode 0 skips revocation; mode 2 performs parent revocation. Mode 1 is rejected.
Default pool-lifetime tests remain separate; A1 is explicitly selected.

Export a completed matrix, including explicitly retained failed attempts:

```sh
python3 security-tests/capstone/export.py /tmp/capstone/a1-results.json \
  /tmp/capstone/a1-new-run
```

Multiple campaign directories allow an interrupted matrix to be completed in
fresh directories. Export requires exactly one accepted execution per matrix
cell, checks raw-file hashes and rechecks the fault/completion oracles.

`alias-scatter-register.S` is a leaf with no calls or stack accesses.
The old alias remains in `a0` until the labelled load/store. Inspect the
**built binary**, including both labels, when changing its assembly or compiler.
No callback, pointer census or alias-holder rewrite occurs on the revoke path.
The reuse path may zero the recovered 4 KiB region; that initialization is
not an address-space alias sweep.

## What this can establish

A passing matrix demonstrates ancestor invalidation for these five alias
locations in the pinned QEMU, including rejection after address reuse, while
unaffected and new authority remains usable. It is functional evidence about
authority and reuse, not a cycle measurement, memory-overhead estimate, proof
over all storage locations, or isolation of a hostile manager in another domain.

PoisonCap and PICASSO are not measured by this matrix. In particular, alias
scattering alone must not be described as a missing PoisonCap property.
A cross-system hierarchy claim needs matched adapters and equivalent parent,
child and reuse semantics. Keep this result separate from the previous trusted
PICASSO per-lease experiment.

The fixture carves synthetic grants directly and does not update FFmpeg's pool
accounting. Its normal-return `payload=0` line therefore does not measure
these grants; use the explicit region sizes above.
