# Wireshark 4.6.8: wmem allocators

Ports Wireshark's memory manager `wmem` — its core and all four allocators — into
native and Capstone-domain replays. The domain executes allocator calls with
checked synthetic payloads; it dissects no packets and runs no dissector. The
pin is the 4.6.8 release archive, verified by SHA256 in `upstream.json`; both
block allocators are blob-identical from 4.4.0 through 4.6.8, and the only
later change in the development line is a one-word comment.

Why this allocator: in a default Wireshark run, five pools are backed by a
nested allocator that retains its blocks across a reset and reissues their
storage. The per-dissection packet pool (`block_fast`) keeps its first 2 MiB
block and rewinds; the file and epan scopes (`block`) keep every block and
rebuild their free lists. Upstream finds the resulting stale-pointer defects
only by substituting a per-object allocator (`WIRESHARK_DEBUG_WMEM_OVERRIDE`)
so that Valgrind or ASan can see them. This port keeps the production
allocators and gives them authority instead.

## Ownership and lifetime boundary

`wmem_alloc(NULL, n)` is wmem's only request to the system allocator: whole
blocks, jumbo objects and its own descriptors. Every such request becomes one
**region** with its own revocation handle (`src/shared/backing.c`); released
regions of the same size are reissued before fresh payload is carved, as a
system allocator would reissue a freed block. Objects handed to callers are
narrowed to their request in both modes.

The allocator hooks (`patches/…-0001-wmem-authority-hooks.patch`, every hunk
guarded by `WMEM_PORT_HOOKS`) do three things:

* **narrow** each returned object to its request (`wm_narrow`);
* **widen** an object pointer the allocator is handed back — for `free` and
  `realloc` — to block-wide authority, after first reading through the
  pointer's own authority (`wm_widen`), so a stale pointer given back to the
  allocator faults at the allocator's probe rather than corrupting its lists;
* **start a new epoch** for every block a reset retains (`wm_epoch`):
  `block_fast` renews its first block, `block` renews each retained block and
  rebuilds its block list from the renewed aliases.

Both modes run the same allocator and backing layout:

- `spatial`: objects are request-bounded; a reset does not revoke old aliases.
- `sublet`: additionally, a reset, a jumbo release, a returned block, a pool
  destruction and every `strict`/`simple` free revoke the authority of the
  storage they end.

What is deliberately **not** revoked: an individual `wmem_free` in the `block`
allocator returns the chunk to a free list inside a live block. Sublet lends
whole regions, and a chunk has no region of its own, so this free ends no
epoch (fixture 4). The same holds for the no-op `free` of `block_fast`.

The scope layer — file and epan scopes, and the per-dissection packet pool
recycled through a one-entry cache — is modelled in `src/shared/scopes.c` on
`epan/wmem_scopes.c` and the `epan_dissect_t` pool handling in `epan/epan.c`
at the pin. It is not extracted: those files pull in the whole of `epan/`.

## Source and layout

The shared `../../common` support provides verified downloads, external build
guards, cross toolchains, run staging and serialized QEMU execution. Only the
`wsutil/wmem` subtree is extracted from the archive. Six upstream translation
units are compiled unmodified except for the guarded hook patch: `wmem_core.c`,
`wmem_user_cb.c` and the four allocators. `src/shared/shim/` shadows `glib.h`
and the `ws_*` headers those units include, mapping the handful of GLib calls
they make onto the port's services; upstream's scope assertions stay active.

`src/native/`, `src/capstone-domain/` and `src/linux-guest/` identify execution
environments. `src/allocators/sublet/` implements the authority operations;
`src/shared/` holds backing policy, the scope layer, replay and the shims.

Two source variants are prepared: `reference` applies no patch and backs the
`replay-reference` executable; `ported` applies the hook patch. Preparation
rejects checksum mismatches, reversed patches and fuzzy matching.

## Build

Source `capstone/tests/capstone-test-env.sh` from the repository root. Cross
builds use the prepared `CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`,
`CAPSTONE_QEMU_BINARY` and `PORT_MUSL_ROOT`. QEMU runners need `pexpect`
(`requirements-dev.txt`).

From this component directory:

```sh
cmake --preset native
cmake --build --preset native
ctest --preset native
cmake --preset capstone-domain
cmake --build --preset capstone-domain
cmake --preset linux-guest
cmake --build --preset linux-guest
```

Builds default to `/tmp/capstone/wireshark-wmem/build/`. Override with `-B` or
an untracked `CMakeUserPresets.json`.

## Replay and verification

```sh
/tmp/capstone/wireshark-wmem/build/native/bin/replay <trace.bin> <report.bin>
python3 host/run-qemu.py <trace.bin> /tmp/wmem-spatial --protection spatial
python3 host/run-qemu.py <trace.bin> /tmp/wmem-sublet --protection sublet
python3 security-tests/qemu/run.py /tmp/wmem-security
python3 host/summarize.py <trace.bin> /tmp/wmem-spatial /tmp/wmem-sublet /tmp/wmem-security results/<date>-qemu
```

Native tests drive all four allocators through allocation, resize, individual
free, reset, collection and destruction — including jumbo objects larger than
a block — and require the hooked build to report byte-identically to the
unhooked reference. Malformed traces (truncation, a missing end, an unknown
pool, a live object reused, an object the pool does not hold) are rejected.
The `port-support` test covers the `wireshark.wmem` trace adapter and the
launcher's validate-before-guest contract.

The paired security fixtures (`security-tests/shared/lifetimes.c`) cover live
controls with storage reuse asserted, packet-pool reset with a stale read and
a stale interior write, the recycler's reset and its individual free, request
bounds, the strict allocator's free, 2,000 reset epochs, pool destruction, a
jumbo object, the file scope's collection, a returned second block, and a
stale pointer handed back to the allocator. A fault verdict requires its stage
marker, the expected cause and the exact PC of the labelled access; a
completion requires status 0. Failed attempts are retained, never retried
silently.

## Evidence

`results/20260921-qemu/` holds the compact record: `summary.json` (both replay
reports, all 26 verdicts, run manifests with input, emulator, compiler and
image hashes, and the hash of every source file), `SHA256SUMS`, and
`raw-artifacts.json`, which points at the local archive of every attempt,
including the two that failed before the passing campaign (a runner host
without `pexpect`; a replay image the loader module refused).

Measured 2026-09-21 under QEMU, revocation node pool 65,536 (the emulator's
compiled `CAP_REV_TREE_SIZE`). The directed trace of 1,661 events — 1,000
allocations, 100 frees, 280 resizes, 80 resets, 40 collections, 80 pool
destructions across all four allocators, 44 regions created and 34 live at
peak — completed with status 0 in both modes, with reports identical to each
other and to the independent accounting in `host/summarize.py`.

| # | fixture | spatial | sublet |
|---|---|---|---|
| 0 | live controls; storage reuse after reset asserted for both allocators | completed | completed |
| 1 | packet-pool reset, stale read | completed, old byte read | fault 24 at the read |
| 2 | reset, storage reissued, stale interior write | completed, new object corrupted | fault 24 at the write |
| 3 | recycler reset retains and reinitializes its block, stale read | completed, reads the free-list node | fault 24 at the read |
| 4 | recycler individual free, stale read — documented limit | completed | completed |
| 5 | one byte past the request | fault 5 (bounds) | fault 5 (bounds) |
| 6 | strict allocator free, stale read | completed | fault 24 |
| 7 | 2,000 reset epochs on one retained block, stale read | completed | fault 24 |
| 8 | pool destroyed, stale read | completed | fault 24 |
| 9 | jumbo object released by the reset | completed | fault 24 |
| 10 | file scope left; collection returns the block | completed | fault 24 |
| 11 | packet pool's second block returned by the reset | completed | fault 24 |
| 12 | stale pointer handed back to the allocator | completed (not attempted) | fault 24 at the allocator's probe |

Every fault landed on the labelled instruction: the read, write and
allocator-probe sites resolved to three distinct PCs, so the oracle
distinguishes them rather than accepting any fault. Cause 24 is revoked
authority; cause 5 is a bounds violation and is the positive control that
faults are observed at all. Fixture 4 completing in `sublet` mode is the
recorded non-detection, not a pass by accident: the recycler's individual
free ends no epoch. Fixtures 3 and 4 completing in `spatial` mode return the
recycler's own free-list node, which it writes into the freed chunk — the
unprotected reader sees allocator metadata, silently.

## CheriBSD and PoisonCap arms

The same sources build for CheriBSD purecap through the shared toolchain
(`cmake --preset cheribsd`, or `host/cheribsd/poisoncap/build.sh BUILD
[--poisoncap] [--corpus DIR]`), with `CHERI_SDK` and `CHERI_SYSROOT` naming
the PoisonCap platform the FFmpeg port reconstructs. Two builds exist:

* **plain** — the port's native backing policy under a purecap libc. Objects
  are not narrowed and nothing is invalidated; what this arm measures is the
  guest's own temporal safety, libc's quarantine and revoker sweep, against
  an allocator that never hands the storage back to libc.
* **PoisonCap** (`WM_POISONCAP`) — `src/cheribsd/poisoncap.c` replaces the
  backing policy. Every system request becomes one mapped region that keeps
  `SW_VMEM` and `POISON` authority for the manager; every published object is
  bounded exactly and stripped of both, so a sweep revokes it and nothing
  else. Mode 0 stops there. Mode 1 invalidates — poison every granule, one
  synchronous sweep, clear, zero — a retained block at every reset, a region
  at every release, and, through a hook Sublet has no use for, a recycler
  chunk at every individual `wmem_free`, before the free list reuses it.

The corpus arms run through `host/cheribsd/poisoncap/run.py`, under
`supervise`, which reports the child's signal and trap PC from the kernel and
resolves `wm_defect_probe` from the child's own map plus the target ELF. A
protected arm that faults anywhere else carries the same exit status; only
the PC comparison in `matrix.json` tells the two apart. The plain build has
one mode, and a completion there means the layer below saw nothing.

Measured 2026-09-21 on the thirteen corpus cases, guest libc revocation on
for all three arms: the plain build completed every case (0 / 13 caught);
PoisonCap mode 0 completed every case; PoisonCap mode 1 faulted on all 13 at
the labelled read. The counters printed at each arm's ready marker name the
hook: cases 0–11 `epochs=1 released_chunks=0` (the reset's block sweep), case
12 `epochs=0 released_chunks=1` (the individual free's chunk sweep).
Records: `bug-corpora/wireshark/wmem-repros/results/20260921-cheribsd/`.
The plain arm's zero and Sublet's twelve rest on different reasons — libc
sees no event at all; Sublet sees the reset but not the chunk — and the
corpus README says which is which.

## Geometry and limits

The domain backs allocations from a 384 MiB payload with 4,096 region slots
and 8,192 replay objects; a long replay with many distinct request sizes can
exhaust either, because regions are reissued only at their exact size and
never returned to the payload. This bounded backing policy is not a libc port.

A domain image without a `.capstone_domreq` declaration is sized by the loader
module's default: an image of 1,330,288 loadable bytes loaded, one of
2,901,776 was refused (loader exit 5, no message on the console because the
module's log is suppressed). The replay image is kept well under that.

The `simple` allocator's objects are bounded to their request but each is its
own region, as upstream intends it to be a per-object allocator. The `strict`
allocator's canaries and fills are kept; its `free` revokes the object's
region in `sublet` mode, which is the per-object behaviour upstream reaches
for when it wants to see this class at all.

These are allocator-component QEMU results. They do not establish protection
of a running Wireshark, dissector correctness on Capstone, FPGA behaviour,
memory overhead of the complete application, or timing overhead. The reported
defects live in `bug-corpora/wireshark/wmem-repros/`: thirteen cases built by
this port with `-DWM_CORPUS_DIR`, twelve caught and one recorded non-detection.
