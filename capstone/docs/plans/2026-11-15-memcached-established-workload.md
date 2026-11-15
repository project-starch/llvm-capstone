# memcached 1.6.45 under mc-crusher: an established workload for the reuse study, pre-registration 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards.

## The question

The question of `2026-10-11-nested-reuse-distance.md`: does the lifetime patch change when memcached's
nested allocators reuse an address? The October row and run-3 ran the port's protocol oracle. That
workload is small. On the trap route it was also broken (finding 1 below). This row replaces it with
mc-crusher, the load generator of memcached's maintainer, sized so that both boundaries reuse many
addresses. It adds a noise floor: the same arm run more than once.

## Definition, observer and boundaries (unchanged)

Reuse distance counts allocations between the release of an address and its next hand-out, at one
boundary, in power-of-two buckets by bit length (bucket k holds 2^(k-1) <= d < 2^k). The observer is
`experiments/study/reuse-gap-observer.h`. The hooks are `experiments/study/reuse-hooks/memcached/`
unchanged (`memcached-1.6.45-reuse-gap-hooks.patch`, `mc-reuse-gap.h`, 2^15 slots per boundary):

- **slabs**: slab chunk hand-out (`do_slabs_alloc`) and free (`do_slabs_free`, `do_slabs_free_chunked`);
- **cache**: `cache.c` object hand-out and free, one observer over every `cache_t` (the per-worker
  read-buffer and response-bundle caches, and the I/O cache).

## Findings that shape this row

1. **run-3's memcached rows ran without responses.** On the trap route the kernel answers `sendmsg`
   with ENOSYS: its msghdr loader does not exist yet (kernel patch 0035; `writev` has an iovec loader).
   memcached sends every TCP response with `sendmsg`. In run-3 each connection therefore closed after
   its first command ("Failed to write, and not due to blocking: Function not implemented" in a
   `-vv` rerun on 2026-11-15), and the oracle transcript of every run-3 repetition is 780 `<EOF>`
   markers out of 791 lines. Run-3's 80 slab and 32 cache hand-outs are what the server did before
   each connection closed.
2. **The slab mover is outside the hooks.** Chunks that the mover rescues, or reclaims during a page
   move, leave the page without `do_slabs_free` (October README, "Deviations and limits"). With
   evictions under a small memory limit and several value sizes, the default automover (`slab_automove=1`)
   moves pages. A later hand-out at a rescued chunk's address would then be an overlapping issue
   (observer error 2). This row turns the automover off.

## Workload

Each server run, one per boot:

1. **Server:** `capstone-job JOB.json --user 65534:65534 -- capstone-vexec ./memcached.dom
   -l 127.0.0.1 -p 21299 -U 0 -m 32 -t 4 -A -o slab_automove=0`, with `EVENT_NOEPOLL=1` (libevent's
   poll backend, no epoll on this platform). Against run-3: `-m 32` instead of `-m 64` (the memory
   limit that makes the slab allocator evict) and `-o slab_automove=0` (finding 2).
2. **Oracle:** `host/mc-harness/mc-harness.c` phases 1 to 4 unchanged. Its transcript is the output
   compared between runs and arms.
3. **Load:** mc-crusher (github.com/memcached/mc-crusher, commit
   7ca3f14c128495a509cd0aaff464545377fa5cee, unmodified, compiled with the line of its Makefile plus
   `-static` against libevent 2.1.12-stable, sha256 92e6de1be9ec...; `build-load-client.sh`), natively
   and unprotected in the guest, over loopback TCP, with the configuration below. mc-harness's new
   `--load` phase starts it after phase 4, reads `stats` on one monitor connection every 10 s (as
   mc-crusher's own sampler does) and ends the client with SIGTERM once the server has processed the
   expected number of commands (cmd_get + cmd_set since the load began). The load has stalled if the
   count does not move for 120 s.
4. **Stop:** the `shutdown` command (`memcached-1.6.45-shutdown-without-signal.patch`); the observer
   reports at exit.

### mc-crusher configuration

`experiments/study/reuse-hooks/memcached/mc-crusher-established.conf` (sha256 ecd2d0751f0b...), one
connection per line:

```
send=ascii_set,recv=blind_read,conns=1,key_prefix=a:,value_size=100,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=a:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=a:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=a:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_set,recv=blind_read,conns=1,key_prefix=b:,value_size=500,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=b:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=b:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=b:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_set,recv=blind_read,conns=1,key_prefix=c:,value_size=2000,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=c:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=c:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=c:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_set,recv=blind_read,conns=1,key_prefix=d:,value_size=8000,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=d:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=d:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
send=ascii_get,recv=blind_read,conns=1,key_prefix=d:,key_count=10000,live_rand_zipf=1,pipelines=4,stop_after=15000
```

Four value sizes (100, 500, 2,000 and 8,000 bytes) under four key prefixes, 10,000 keys each, keys
drawn by mc-crusher's own zipf generator at its default skew (0.25). Per prefix one connection sets
and three get (a read-heavy mix, 3:1), each 15,000 commands in pipelines of 4, so all sixteen
connections run for about the same time. Expected commands: 60,000 sets + 180,000 gets = **240,000**.
The four sizes fall into slab classes 3, 8, 13 and 19 in both arms (stock chunks 208, 704, 2,192 and
8,448 bytes; protected 240, 768, 2,368 and 9,088, because the protected item carries a 16-byte
`sublet_parent` field). Every slab hand-out of the load is a set: 60,000 hand-outs, above the 10^4
the row asks for. Without the automover the pages stay with the class that took them first, and
under `-m 32` every one of the four classes evicts once it is full.

### Calibration (stock only, before this plan)

Images built by `build-reuse.sh` from the bench/memcached snapshot of ed8cc375c2 with the writev
patch (stock memcached.dom sha256 f272d96abda0..., protected 734fea638b00...). Three stock calibration
boots on the platform above:

| Run | Load | Commands | Load seconds | Slab issues / reuses / distinct | Cache issues | Evictions |
|---|---|---|---|---|---|---|
| cal1 | 4 sizes x (1 set + 1 get), 20,000 keys, 5,000 each | 40,000 | 259 | 20,352 / 7,601 / 12,751 | 10,681 | 5,371 |
| cal2 | the same, paced (usleep 20 ms), 2,000 each | 16,000 | 104 | 8,352 / 453 / 7,899 | 3,283 | 0 |
| cal3 | this configuration at a quarter (3,000 each) | 48,000 | 208 | 12,352 / 2,539 / 9,813 | 12,490 | 764 |

All three completed with processed == expected and observer error 0, and every oracle transcript had
sha256 ee1522c8e0ec..., the transcript of the October run on the virtual route: the writev patch
restores the oracle's output. Scaled from cal3, the full load takes about 17 to 20 minutes per
stock boot. Paced mode was rejected: the server, not the pace, set the rate.

## Study patches (both arms, outside the allocators)

- `memcached-1.6.45-shutdown-without-signal.patch` (as in run-3).
- `memcached-1.6.45-tcp-writev.patch` (new, finding 1): `tcp_sendmsg` sends a message without name,
  control data or flags through `writev` with the same iovec. For a connected TCP socket that is the
  same byte stream. The source gate still requires `slabs.c`, `cache.c`, `slabs_mover.c` and
  `memcached.h` to be byte-identical between the arms.

## Arms and repetitions

**stock** (`-DEXP_REUSE_GAP`) and **protected** (`-DEXP_REUSE_GAP -DMC_CAPSTONE_SUBLET`), built by
`build-reuse.sh` from one tree. Three servers per arm, each in its own boot (stock-1..3,
protected-1..3); two per arm at the least if a run must be rejected. The trap-route platform of run-3
and run-4 (QEMU 9289114e06, kernel Image-0133b9e21cd, firmware f-keep, the run-4 kit).

A run is rejected if it did not complete, if the observer's `error` is nonzero at either boundary, if
`capstone-job` did not record exit 0, if the load did not complete with processed == expected, or if
its oracle transcript differs from the others or holds `<EOF>` or `<TIMEOUT>`.

## Sources of variation

Four worker threads on one hart interleave their hand-outs in an order that is not fixed. mc-crusher
seeds each connection's key generator from the clock and the connection's address, so the key
sequences differ between runs. Both act on both arms alike.

## Predictions

1. **Output.** Every run's oracle transcript is byte-identical to the calibration transcript
   (sha256 ee1522c8e0ec...), and every load completes with processed = 240,000.
2. **slabs counts.** Issues = 60,352 in every run of both arms: the 60,000 sets plus the oracle's
   352 hand-outs, since every set takes a new chunk. Releases and reuses vary from run to run (they
   follow the key sequences). Distinct starts are lower in protected (fewer chunks per page).
3. **slabs distribution.** The median bucket is 1 in every run. The noise floor (within one arm) is
   at most 0.02. For every stock x protected pair, reusediff is at most the larger of the two
   within-arm noise floors plus 0.03. The allowance is for the item header: each protected page
   holds 6 to 9% fewer chunks (5,041 against 4,369; 1,489 against 1,365; 478 against 442; 124
   against 115), so evictions can take a slightly larger share of the reuses.
4. **cache distribution.** The median bucket is 1 in every run. The noise floor here is larger than at
   the slab boundary (worker interleaving decides which buffer a worker gets back). For every stock x
   protected pair, reusediff is at most the larger within-arm noise floor plus 0.02.

Predictions 3 and 4 together are the result: the lifetime patch leaves reuse at both boundaries
within the run-to-run spread, up to the stated allowance.

## Report

Per arm and repetition and per boundary: issues, releases, reuses, distinct starts, error, the load
result and the output check. Per boundary, the share of reuses whose bucket differs (`reusediff`,
`sum_k |a_k/N_a - b_k/N_b| / 2`, as in `cpython/collect-reuse.py` and `memcached/assemble-reuse.py`):
within stock (noise floor), within protected, and stock against protected (every pair), and the
median bucket per run. Results go to `~/asr-work/eval/run-5/reuse/memcached-<arm>-<rep>/` on the build
host with a `provenance.txt`.
