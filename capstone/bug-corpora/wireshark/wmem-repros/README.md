# Wireshark wmem defects

Upstream use-after-free defects in Wireshark's dissection loop, reduced to
programs that run against Wireshark's **own** memory manager: `wmem`'s core
and its four allocators from the pinned 4.6.8 release, compiled unmodified but
for the guarded authority hooks the port applies. The consumers are reduced,
the allocator is not.

> **Looking for more cases?** [`../LIVE-CANDIDATES.md`](../LIVE-CANDIDATES.md) lists **92
> defects live at the 4.6.8 pin**, checked by content rather than by ancestry, with the two
> sharpest verified by hand against the pinned source. The reason they were not found earlier
> is the population: the previous triage searched `v4.6.8..release-4.6`, 83 commits, which by
> construction cannot see a defect whose fix was **never backported**. `v4.6.8..master` is
> 4,401. The checker is committed beside the list as `../check-liveness-at-pin.py`.


The layout is the contract in
[`../../SCHEMA.md`](../../SCHEMA.md):
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
| 22 | `c702b44a01` | #16818 | double free into the block allocator's free list | no: the second free is itself the access |

"Reoccupied before the read" is asserted only where the report evidences it:
the next packet's own first allocation is checked to land on the same address
before the ready marker. Where a report shows only the stale read, the case
performs only that, and the unprotected arm returns the old bytes.

Case 12 is the corpus's recorded **non-detection**. Its lifetime ends by an
individual `wmem_free` into the block allocator's recycler, not by a pool
reset; Sublet lends whole regions, a chunk inside a live block has no epoch of
its own, and the protected arm is expected — and checked — to complete. It is
kept because hiding it would misstate what the mechanism covers.

**Since the chunk port (2026-09-29), case 12 is caught too.**

- The port's default build (`WM_CHUNKS=ON`, `a34caaedb1bc`) gives every chunk of the block
  allocator a region of its own, so the individual free is a revoke. Case 12 then faults at its
  read probe.
- The region-granular build (`WM_CHUNKS=OFF`) still completes it. The two builds have separate
  oracles, `sublet-chunks` and `sublet`, and the runner picks one from the build.
- Result: [`results/20260929-qemu-chunk-port/`](results/20260929-qemu-chunk-port/README.md).

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
| double free into the block allocator's free list | the same chunk is wmem_free'd twice into the block allocator; the second free is the defective access, and the recycler list written over freed chunks is corrupted (case 22, added 2026-10-11) |

## Arms

| arm | target | what it establishes |
|---|---|---|
| `spatial` | Capstone domain | the sequence completes without protection |
| `sublet` | Capstone domain | fault at the labelled probe the oracle names |
| `cheribsd` | CheriBSD purecap, libc revocation ON, plain build | whether the system allocator sees these defects |
| `poisoncap-spatial` | CheriBSD purecap, PoisonCap mode 0 | exact bounds, no invalidation: the matched control |
| `poisoncap-protected` | CheriBSD purecap, PoisonCap mode 1 | SIGPROT at the labelled read probe |
| `native-detect` | host | declared, not written |

The two Capstone arms run the same program; the loader picks the arm at run
time. The two PoisonCap arms run the same purecap program with a mode argument,
under `supervise`. Every protected oracle names an **instruction**: the run publishes the probe addresses
and the fault must land on the one the case's oracle names — the read probe,
the write probe, or the allocator's own probe when the stale pointer is handed
back to `wmem`.

## What the four systems do, measured 2026-09-21

| system | what it acts on | caught |
|---|---|:--:|
| Capstone | bounds and tags; no lifetime event | **0 / 13** |
| **Sublet** | the block's epoch at a pool reset | **12 / 13**, cause 24 at the read |
| CheriBSD default | `free()` → quarantine → revoker sweep, libc revocation on | **0 / 13** |
| **PoisonCap** | poison at reset, at release and at the recycler's individual free, then a sweep | **13 / 13**, SIGPROT at the read |

By shape, for the two mechanisms that catch anything:

| shape | cases | Sublet | PoisonCap |
|---|---|:--:|:--:|
| stale pointer held by a global across packets | 0, 1, 6 | 3 / 3 | 3 / 3 |
| packet-scope object held by a column or address past the scope's end | 2, 3, 4, 5 | 4 / 4 | 4 / 4 |
| packet-scope object kept by file-scope state across packets | 7, 8, 9, 10, 11 | 5 / 5 | 5 / 5 |
| stale pointer after an individual recycler free | 12 | **0 / 1** | **1 / 1** |
| double free into the block allocator's free list | 22 | region build **0 / 1**; chunk port **1 / 1** (cause 24 at the allocator's handback probe) | **1 / 1** (SIGPROT in wm_widen) |
| cursor advanced past its chunk by a fixed skip, read inside the same block | 13 | **1 / 1** (bounds, cause 5) | not run |
| loop reads a fixed offset past its chunk into the next chunk of the same block | 14 | **1 / 1** (bounds, cause 5) | not run |
| packet-controlled negative index reads below the chunk, inside the same block | 15 | **1 / 1** (bounds, cause 5) | not run |
| fixed-offset parity write lands past the buffer, inside the same block | 16 | **1 / 1** (bounds on a STORE, cause 7) | not run |
| a size one larger than the chunk permits a one-byte write past it | 17 | **1 / 1** (bounds on a STORE, cause 7) | not run |
| a guard disabled by its own default preference lets the counter run off the arrays | 18 | not run | not run |
| metadata written before the allocation it describes, so a consumer trusts a length the buffer lacks | 19 | not run | not run |
| a marking loop bounded by the field's extent rather than by the bitmap's length | 20 | not run | not run |
| a fixed, compile-time-known output larger than its fixed buffer | 21 | not run | not run |

**Cases 18-21 were added 2026-10-07, and two candidates from the same lists were REJECTED rather
than built.** The leads came from [`../LIVE-CANDIDATES.md`](../LIVE-CANDIDATES.md) and from the
class-B candidates `docs/ref/wireshark-spatial-defect-triage.md:226-229` named and left unbuilt.
Rejected, with the reason recorded so nobody re-derives it:

| candidate | why not |
|---|---|
| `1c090e9292` | a **stack** buffer overflow — no heap allocator boundary, the same reason memcached's `11b5f9b` was disqualified |
| `d7d1686a95` | indexes a **static** `value_string` array — likewise no allocator |
| `0939cf989d` | the same ETSI DCP defect as case 16 under a second hash — a duplicate, not a case |
| `b2bc518e4d` | crosses a **tvb** bound, not an allocator's |
| `76459b8134`, `de719cc5ac` | signed/unsigned overflow of a length, with the spatial consequence downstream — the class this project retracted `f207d25f4b` for |

**Why the corpus stopped at 22 rather than going further.** The spatial-wording population is large —
234 commits across `v4.4.0..v4.6.8` and `v4.6.8..HEAD` in `epan` and `wiretap` — but it and the
hand-verified LIVE list are both **dominated by integer overflow**, which this corpus's filter
excludes by design. The table above is what the remaining named candidates came to. Going further
would have meant admitting rows whose defect class would then have to be misdescribed, which is how
the two retractions of 2026-10-06 happened.

Across cases 0-12 every `spatial` and every plain-CheriBSD arm completed. **Case 13 is the
exception and it is not a counter-example:** it is the one spatial row, so its unprotected arm
faults too (cause 5, measured both builds — `results/20261005-qemu-spatial-case13-ON/`). For the
twelve temporal rows the unprotected
allocator returns either the old bytes (the column cases, where nothing
intervenes) or another object's bytes (the cases that assert reoccupation),
and libc's quarantine never sees an event because wmem hands storage back to
libc neither at a reset nor at a free. Every protected fault landed on the
labelled instruction, compared against the address the run resolved from the
image or, on CheriBSD, from the child's own map. The records are
`results/20260921-qemu/` and `results/20260921-cheribsd/`, the negative
control beside the first.

Case 12 is the row the two mechanisms disagree on, and both readings were
predicted before the runs. Its lifetime ends by an individual `wmem_free`
into the block allocator's recycler, inside a live 8 MiB block. Sublet's epoch
is the block's, so the free ends nothing and the read completes. PoisonCap
acts on the chunk: the port's hook poisons its granules at the free, one
sweep later the registry's stale name is dead, and the read faults —
`epochs=0 released_chunks=1` in that arm's counters, where the other twelve
read `epochs=1 released_chunks=0`. Sublet could revoke the chunk at that
free, as the PostgreSQL port does at every `pfree` — but only as a linear
piece, because `mrev` takes a linear source, and `block` coalesces
neighbouring free chunks, which linear pieces cannot re-form without a merge
Capstone lacks. The shipped port keeps region granularity for that reason:
the allocator's policy stays byte-identical to upstream, and this one case
is the price. `docs/design/capability-merge-primitive-proposal.md` states the
trade, the missing primitive, and what a per-chunk port would cost.

**Measured 2026-09-29, with the chunk port** (`results/20260929-qemu-chunk-port/`):

- **Sublet catches 13 / 13**, each at the labelled read probe. Case 12 faults on the chunk build
  ×3 and completes on the region-granular build ×3, one option apart.
- The spatial arm completes 13 / 13, and the negative control fails 26 / 26.
- The chunk port pays for this differently from what that paragraph foresaw. It does not merge
  freed neighbours, and a reset returns each block UNINIT and pays a capability-initialising fill;
  see `ports/wireshark/wmem/results/20260929-qemu-chunk-port/`.

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
the exit status 0 only if every one did. `../../tools/check-corpus.py --self-test`
enforces the contract and first proves it can reject four corruptions.

## Vetted upstream candidates NOT built here (2026-10-08)

Seven wmem defects were mined and vetted on 2026-10-08 but not reduced, because the batch that day
went to the two cells that were EMPTY — `wireshark/plain-heap-repros` and the new
`wireshark/plain-temporal-repros` — while this corpus already held 22 cases. They are recorded so
the next pass does not re-mine them. Each was read at the fix's parent; the class and the crossing
below are from that read, not from the commit message.

| fix | path | kind | what crosses |
|---|---|---|---|
| `5441003874` | `epan/conversation.c` | spatial | `wmem_alloc(..., sizeof(conversation_element_t) * (DEINTD_ENDP_NO_PORTS_IDX+1))` is 3 elements, and the NO_PORTS arm writes index 3 as well — one 24-byte element past the chunk |
| `d5f2657825e6` | `epan/to_str.c` | spatial | `wmem_alloc0(pool, 256+64)` sized for "256 bits and one space per four", while the loop is bounded by the caller's `no_of_bits` and emits 11 characters per 8 bits — unbounded, overflowing from about `no_of_bits = 234` |
| `9de18e88f501` | `epan/addr_resolv.c` | allocator mismatch | `wmem_new(wmem_epan_scope(), ...)` released with `g_free` — the platform allocator handed an interior pointer into a wmem block |
| `661743e4da6b` | `epan/addr_resolv.c` | allocator mismatch | `fgetline`'s `wmem_alloc`/`wmem_realloc` line buffer released with `g_free` |
| `14f2a654d43c` | `epan/addr_resolv.c` | allocator mismatch | the same shape in the sibling reader; collapse with the row above if one row per MECHANISM is wanted rather than per instance |
| `a2c8ff7cb6b0` | `epan/dissectors/packet-umts_rlc.c` | allocator mismatch | `tvb_memdup(wmem_file_scope(), ...)` released with `g_free` on both teardown paths |
| `b1e0cb01b33d` | `epan/dissectors/packet-coap.c` | temporal | a PACKET-scope string stored in a FILE-scope struct, so the next `wmem_free_all` of the packet pool reclaims it while the struct still points at it |

**Every one of these crossings leaves the wmem allocation and stays inside the platform one** —
wmem's block allocator carves 16-byte-aligned chunks out of an 8 MiB `g_malloc`'d block
(`WMEM_ALIGN_AMOUNT`, `WMEM_BLOCK_SIZE` in `wsutil/wmem/wmem_allocator_block.c`). That is the
nested-allocator property this corpus exists to exercise, and it is why these belong here rather
than in `../plain-heap-repros`.

Two more were mined and REJECTED, with the reason worth keeping: `9ea014637ece` (`wiretap/pcapng.c`,
a genuine heap write-overflow where `pcapng_compute_string_option_size` returns
`strlen(...) & 0xffff` and the writer then copies the full string) needs two functions to explain,
and `984e52244f08` (`epan/column-utils.c`) has its allocation in a different file from its crossing.
