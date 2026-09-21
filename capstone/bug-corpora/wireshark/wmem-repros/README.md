# Wireshark wmem defects

Upstream use-after-free defects in Wireshark's dissection loop, reduced to
programs that run against Wireshark's **own** memory manager: `wmem`'s core
and its four allocators from the pinned 4.6.8 release, compiled unmodified but
for the guarded authority hooks the port applies. The consumers are reduced,
the allocator is not.

The layout is the contract in
[`../../cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md):
one directory per case, `NN_<upstream-fix>_<slug>/`, holding a `case.c` that is
a complete translation unit, a `case.json` of machine-readable claims, and a
`PROVENANCE.md`. Case numbers are dense from zero. One program per case,
because a capability fault ends the run and a case that provokes one cannot
also report results beside it.

Every case turns on the same event: the per-dissection packet pool is reset
(`wmem_free_all`), which ends every packet-scoped object at once while the
pool's first 2 MiB block stays allocated. The pointer that survives the reset
is what differs — a C global, a column, an address, a structure the file scope
owns — and so does whether the next packet reoccupies the storage before the
stale read. The seam (`shared/corpus.h`) exposes that pool as `wm_packet` and
the reset as `wm_next_packet()`; where the report's pool was the separate
`wmem_packet_scope()` of older releases, the seam's pool stands in for it: the
same allocator type with the same reset.

## The cases

Seventeen reports collapse to thirteen defects: three CMS captures, two USBLL
captures and two HTTP captures each reached one line, and each pair or triple
is one case with the other reports named as siblings.

| # | upstream fix | reports | shape | reoccupied before the read |
|---|---|---|---|---|
| 0 | `3c8be14c82` | #18852, #18910 | stale pointer held by a global across packets | yes, asserted |
| 1 | `c14d731e45` | #17800, #17809, #17835, #17935 | stale pointer held by a global across packets | yes, asserted |
| 2 | `99da8c2cdc` | #21261 | packet-scope object held by a column or address past the scope's end | no |
| 3 | `6eab9f83ab` | #20587 | packet-scope object held by a column or address past the scope's end | no |
| 4 | `b48759e4a4` | #19960 | packet-scope object held by a column or address past the scope's end | no |
| 5 | `5a109265a6` | #17367, #17368 | packet-scope object held by a column or address past the scope's end | no |
| 6 | `a8b16d74e1` | #18622 | stale pointer held by a global across packets | not asserted |
| 7 | `fb504bc76c` | #19045 | packet-scope object kept by file-scope state across packets | not asserted |
| 8 | `31ab1a0a17` | #18735 | packet-scope object kept by file-scope state across packets | not asserted |
| 9 | `693dc40936` | #18779 | packet-scope object kept by file-scope state across packets | not asserted |
| 10 | `6fd3af5e99` | #19695 | packet-scope object kept by file-scope state across packets | not asserted |
| 11 | `3a5f82dfb5` | #20702, #20703 | packet-scope object kept by file-scope state across packets | yes, asserted |
| 12 | `90bb3a5c9e` | #20664 | stale pointer after an individual recycler free | yes, asserted |

"Reoccupied before the read" is asserted only where the report evidences it:
the next packet's own first allocation is checked to land on the same address
before the ready marker. Where a report shows only the stale read, the case
performs only that, and the unprotected arm returns the old bytes.

Case 12 is the corpus's recorded **non-detection**. Its lifetime ends by an
individual `wmem_free` into the block allocator's recycler, not by a pool
reset; Sublet lends whole regions, a chunk inside a live block has no epoch of
its own, and the protected arm is expected — and checked — to complete. It is
kept because hiding it would misstate what the mechanism covers.

Every `case.json` says how liveness at the 4.6.8 pin was established, and
each `PROVENANCE.md` quotes the pre-fix code from the fix's parent by line.
Where a report's pool was the separate `wmem_packet_scope()` of older
releases, the seam's packet pool stands in for it: the same allocator type
with the same reset.

## Shapes

`case.json`'s `shape` must be one of these, so that two cases sharing a
reduction class are visibly siblings rather than accidentally similar.

| shape | what makes it its own class |
|---|---|
| stale pointer held by a global across packets | a file-scope C global or static keeps the pointer; a later packet's dissection runs before the read |
| packet-scope object held by a column or address past the scope's end | a column or an address keeps the pointer; the scope is torn down at the end of dissection and the print step reads it in the same packet, before anything reoccupies the storage |
| packet-scope object kept by file-scope state across packets | a conversation record, proto data or a reassembly table in file scope keeps the pointer, and a later packet reads it through that state |
| stale pointer after an individual recycler free | the lifetime ends by wmem_free into the block allocator's free list, inside a live block; no pool reset is involved |

## Arms

| arm | target | what it establishes |
|---|---|---|
| `spatial` | Capstone domain | the sequence completes without protection |
| `sublet` | Capstone domain | fault at the labelled probe the oracle names |
| `native-detect` | host | declared, not written |

Both arms run the same program; the loader picks the arm at run time. The
sublet oracle names an **instruction**: the run publishes the probe addresses
and the fault must land on the one the case's oracle names — the read probe,
the write probe, or the allocator's own probe when the stale pointer is handed
back to `wmem`.

## What Sublet does, measured 2026-09-21

| shape | cases | caught |
|---|---|:--:|
| stale pointer held by a global across packets | 0, 1, 6 | **3 / 3**, cause 24 at the read |
| packet-scope object held by a column or address past the scope's end | 2, 3, 4, 5 | **4 / 4**, cause 24 at the read |
| packet-scope object kept by file-scope state across packets | 7, 8, 9, 10, 11 | **5 / 5**, cause 24 at the read |
| stale pointer after an individual recycler free | 12 | **0 / 1** — recorded, not hidden |

Twelve of thirteen. Every `spatial` arm completed: the unprotected allocator
returns either the old bytes (the column cases, where nothing intervenes) or
another object's bytes (the cases that assert reoccupation). Every fault
landed on the labelled instruction, compared against the address the run
resolved from the image. The record is `results/20260921-qemu/`, its negative
control beside it; both attempts, including the one whose three cases derived
an interior pointer after the reset and faulted at the arithmetic instead, are
described there.

The one miss is the one predicted before the first run: case 12's lifetime
ends by an individual `wmem_free` into the block allocator's recycler, inside
a live block, and Sublet's epoch is the block's. A mechanism that acts per
chunk would see it; this one does not, and the row says so.

## Building and running

The port builds the cases; case material does not live inside a port. Pass the
corpus root and the port builds one program per case:

    cmake --preset capstone-domain -DWM_CORPUS_DIR=<repo>/capstone/bug-corpora/wireshark/wmem-repros
    cmake --preset native          -DWM_CORPUS_DIR=<repo>/capstone/bug-corpora/wireshark/wmem-repros

Programs are named as the contract names run artifacts, `02-mdb-address-column`,
with `.dom` for the domain. The hosted build runs the unprotected sequence
natively (`<program> 0 <case>`) and refuses any other mode or case with exit 75.

    /tmp/capstone/venv/bin/python3 shared/run-defects.py OUT --domain-build BUILD_DOMAIN --linux-build BUILD_LINUX
    /tmp/capstone/venv/bin/python3 shared/run-defects.py OUT-nc ... --negative-control
    python3 shared/summarize-run.py OUT results/<stamp>

The runner needs the same environment the port documents (`CAPSTONE_QEMU_BINARY`,
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, the venv with `pexpect`).
A run that produced no `serial.log` did not run; check before believing a
domain result. `--negative-control` corrupts the input record so the program
refuses it before any case runs; every arm must then FAIL, and the flag makes
the exit status 0 only if every one did. `tests/check-corpus.py --self-test`
enforces the contract and first proves it can reject four corruptions.
