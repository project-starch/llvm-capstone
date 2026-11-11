# memcached, the `sublet` arm of the virtual gate after the conversion to CRENEW, 2026-11-11

`host/run-virtual.py --arm sublet` on `host/build-virtual.sh`'s images, built from the tree with
patch 0006 as converted: `slabs_free` ends a chunk with the one-operand `CREVOKE(chunk)`, and
`cache_free` renews the object's own lifetime with `CRENEW`. The processor model is the one with
the one-operand `CREVOKE` and `CRENEW`. The previous record of this arm is
[`../2026-10-13-fixture-24-path`](../2026-10-13-fixture-24-path/README.md) (`sublet-result.json`).

| | checks | pass | fail |
|---|---|---|---|
| 2026-10-13 | 108 | 106 | `fx24_prediction`, `fx24_trigger` |
| 2026-11-11 | 109 | 101 | those two, the four `*_stderr` checks, `fx18_prediction`, `fx18_trigger` |

The 109th check is `memory_mode_verified`, which the runner gained since; it passes. The gate stays
**FAIL**. Every protocol, identity, exit and worker check passes, and so does every fixture
prediction except 18 and 24. The predictions file is the one of 2026-10-13
(`predictions_sha256` `a183e1c5...`).

## The three groups of failures

**Fixture 24: unchanged.** The same no-op as on 2026-10-13, line for line: `header-only=0`,
`triggering condition not created`, mark `18e0018`. Fixture 25 returns `1900020` as before.

**The four `*_stderr` checks: the launcher's exit line.** `normal1`, `normal2`, `usr1` and `marker`
each compare the server's stderr with the native server's. With `CAPSTONE_EXEC_DIAGNOSTICS=1`, which
the gate script exports, the launcher now ends every run with

    capstone-exec: exit image=./memcached.dom sha256=<image hash>

(`runtime/virtual/exec.c`, the `tid == 0 && (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))`
branch). That line is the whole of each `server.err`; no run recorded a fault. The launcher source
and `run-virtual.py` are byte-identical on the base revision of the conversion, so these four checks
fail there too. The conversion does not cause them.

**Fixture 18: its precondition did not arise, on the base revision either.** Fixture 18 is the
logger watcher: `calloc(1, 64)`, `free`, `calloc(1, 64)`, then a write through the first pointer. It
needs the second `calloc` to receive the first one's address, and it stops with the no-op mark when
that does not happen:

    MCAPP-FIX 18 watcher cursor=3f9a401280 bounds=[3f9a401280,3f9a4012c0) len-from-cursor=64
    MCAPP-FIX 18 storage NOT reissued: triggering condition not created
    MCAPP-FIX 18 mark=12e0018

On 2026-10-13 the storage was reissued and the stale write faulted (`FAULT temporal`). Here the
stale write never runs, so this is a finding about the fixture and says nothing about the
protection: `safety-expect.txt` rules that such a mark is never a pass.

Two boots settle where it comes from. Each ran fixtures 17 and 18 alone, as the gate runs them, on
the nested safety image and on the plain one:

| build and platform | image | fixture 17 | fixture 18 |
|---|---|---|---|
| this tree, this platform | nested (patch 0006) | fault, cause 25 | `storage NOT reissued`, mark `12e0018` |
| this tree, this platform | plain | fault, cause 25 | `storage NOT reissued`, mark `12e0018` |
| base revision, its own build and platform | nested (patch 0006 before the conversion) | fault, cause 25 | `storage NOT reissued`, mark `12e0018` |
| base revision, its own build and platform | plain | fault, cause 25 | `storage NOT reissued`, mark `12e0018` |

So fixture 18 reads the same on the base revision, with and without patch 0006. The conversion does
not cause it. The watcher is a plain libc allocation, and whether the next allocation takes a freed
slot is mallocng's choice. Something between 2026-10-13 and the base revision changed that choice
for this size class; this record does not look for it. Fixture 17, which has the same shape with
another size, still gets its storage back and faults as predicted.

## Platform and options

`run-virtual.py --arm sublet --unpinned`, the platform's gate copy of the module (32 contexts,
`unpinned=1`), and

    CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=false,x-capstone-supervisor-slots=32,x-capstone-raw-moves=true,x-capstone-exact-bounds=true

The first three properties are the landing gate's set. Exact bounds is added because patch 0006
derives each class's generation as `size * perslab` bytes from the page base, which the model's
default compact bounds refuses for most classes (cause 29 at the `CDERIVE`, measured on the
allocator corpus's control `chunk-free-90`). `sublet-result.json` carries every input hash: QEMU
`2d5a7107...`, launcher `0f8e0f90...`, module `38d4d411...`, kernel `48eb725a...`, firmware
`5b5c047a...`, rootfs `62049800...`, server image `7d874798...`, safety image `8bc9452e...`.

## Build

`host/build-virtual.sh <fresh dir>` passed all its gates. The instruction gate of
`host/build-sublet.sh` reads, for both the oracle and the safety tree: `memcached-slabs.o` CDERIVE 6,
CREVOKE 3; `memcached-cache.o` CRENEW 2; the recompiled stock objects none. In the linked server the
instructions sit in `do_slabs_newslab` (CDERIVE 2), `slabs_alloc` (CDERIVE 2),
`slabs_finalize_page_move` (CDERIVE 2, CREVOKE 1), `slabs_free` (CREVOKE 1), `do_slabs_free`
(CREVOKE 1), and `do_cache_free` and its inlined copy in `cache_free` (CRENEW 1 each). A
`capstone_cap_derive` call site holds two CDERIVE encodings, the derive and the refusing zero-size
form. The base revision's objects read `memcached-slabs.o` CDERIVE 10, CREVOKE 3 and
`memcached-cache.o` CDERIVE 8, CREVOKE 2.

`sublet-fixture-lines.txt` holds fixtures 18, 24 and 25 as the guest printed them.
