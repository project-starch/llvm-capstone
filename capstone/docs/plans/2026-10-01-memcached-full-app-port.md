# memcached 1.6.45 as a full application on the delegated runtime

**Status:** M0–M5, the Safety fixtures and S1 (Sublet inside the slabs) done (2026-10-01). Open and
awaiting the project lead: stretch S2 (upstream `t/*.t`) and corpus case 02 over the protocol.
Lane branches `memcached-app`, `slab-sublet`.

## Why memcached, and why now

memcached was declined as a whole application because the program *is* a thread pool on libevent
and sockets (`docs/plans/2026-09-23-ffmpeg-full-app-port.md`: "memcached and httpd are pools by
architecture"); only its allocators were extracted (`ports/memcached/allocators`, corpus
`bug-corpora/memcached/allocator-repros`). The delegated runtime now serves what it needs:

- **threads**, up to 15 per application besides main (`docs/design/delegated-threads-model.md` §11);
  `-t 4` runs about nine;
- **sockets** (`socket` .. `accept4`, `sendmsg`/`recvmsg`), **epoll**, `eventfd2`, `pipe2`, futex
  wait/wake, `prlimit64` on the task itself (`runtime/applications.md`).

It is chosen over Apache httpd (`-X`, single process) because it has one dependency (libevent)
against APR, APR-util, expat and PCRE2, and because its protocol carries no dates or pids, so a
byte-exact transcript against a native build is a natural oracle.

**What it cannot have here:** parallelism (one hart), `fork` (so no `-d`), `setuid` (so it never runs
as root: started as `nobody`), `mlockall` (`-k`), `madvise` (`-L`), file mappings (`-e`), QEMU only.

## Pins

| input | version | pin |
|---|---|---|
| memcached | 1.6.45 | sha256 d362c64e… (`ports/memcached/sources.sha256`) |
| libevent | 2.1.12-stable | sha256 92e6de1be9ec176428fd2367677e61ceffc2ee1cb119035037a27d346b0403bb (buildroot `package/libevent/libevent.hash`) |

## M0 findings so far (read at the pins)

1. **libevent's epoll backend keys events by descriptor**, not by pointer: `epoll.c:198,288`
   store `epev.data.fd`, `:485,505` read it back. The runtime carries `epoll_event.data` as 64 bits
   (a pointer there would lose its tag; an fd survives), so the epoll backend is usable as is.
2. **Item alignment needs a patch.** `memcached.h:121` `CHUNK_ALIGN_BYTES 8`; `slabs.c:252` rounds
   every slab class size to 8, and items are carved at `page + i * size`. An item holds three
   pointers (`next`, `prev`, `h_next`), 16 bytes each here, so in a class whose size is 8 mod 16
   every other item stores capabilities at 8-aligned addresses and faults. Chunked items have the
   same shape: `ITEM_schunk` (`memcached.h:676`) and `htotal` (`items.c:282`) round to 8. Patch 0001
   will align to the pointer size; it is decided, not yet written.
3. **The 1.6.29-era facts hold at 1.6.45:** `-P` writes the pid file without `-d`
   (`memcached.c:6198`); `no_lru_crawler`, `no_lru_maintainer`, `no_slab_reassign`, `no_hashexpand`
   and `tail_repair_time` are `-o` names; the root branch (`:5858`) needs `setgroups`/`setuid`, which
   the runtime refuses, hence `nobody`; SIGINT/SIGTERM end the loop normally and SIGUSR1 stops
   gracefully with `stop_threads()` (`:4886-4889`, `:6218-6240`); `stats pointer_size`
   (`:1767`) exists for the identity witness.

## M0 threads probe — PRE-REGISTERED before its first boot

What memcached does with threads, reduced to a probe (`ports/memcached/app/probe/`): a domain that
listens on 127.0.0.1, accepts on the main thread and hands each connection to one of four worker
threads through a per-worker queue and an `eventfd`, each worker blocked in `epoll_wait` on its
own epoll set; a fifth thread loops on `pthread_cond_timedwait` (the LRU maintainer's shape); main
waits in `epoll_wait` with a 1 s timeout (memcached's clock tick) and leaves on a SIGTERM flag,
then stops and joins every thread. A native guest client (`linux-guest.cmake`) opens eight
connections, sends one line on each and checks the echo, which names the worker that served it.

Predictions, written before the probe has run:

| | prediction | if it fails |
|---|---|---|
| P1 | the client connects to the domain's listener and all 8 echoes are correct | a native guest process cannot reach a domain listener over loopback: the oracle design changes |
| P2 | the round-robin hand-off reaches every worker: each of the 4 serves exactly 2 connections | eventfd wake-up across contexts does not work as memcached needs |
| P3 | SIGTERM ends the probe within 2 s: `STOPPED joined=5`, exit 0 | signal delivery to a domain whose contexts all block does not end the loop |
| P4 | the exit report lists no unserved syscall | memcached will meet the same gap |

### Result, 2026-10-01 (one boot; predictions committed first, d086d1068f2d)

Images: probe `mc-threads-probe.dom` 4c09f260…, client `mc-probe-client` cc8bf6ce…. Platform as dev pins
it: capstone-qemu 674cdab0 (32e7c975…), fw_jump.elf 6f2b082c… (OpenSBI cf344cf3 + capstone-sbi
4674ab6a), capstone.ko fdb947f2… (buildroot 8fd1ea12), the lane's launcher e7e27f49…, compiler 612b3ec5.

| | outcome |
|---|---|
| P1 | **holds**: the native client reached the domain's listener; 8/8 echoes correct |
| P2 | **holds**: `served=2,2,2,2`, every reply naming the worker it predicted |
| P3 | **holds functionally, misses its bound**: `STOPPED joined=5`, exit 0, but 2.21 s from `kill` to `wait`, not ≤ 2 s. The bound was too tight for a 1 s tick plus five joins and the launcher's exit; it is recorded as missed, not moved |
| P4 | **holds**: no unserved line. The silence is evidence because the report was shown to fire: a domain calling `mlockall` (fd690a55…) under the same switch prints `capstone-domain: UNSERVED syscalls: 230`. The launcher's own counter said `refused=0` there too, so it cannot stand in for this report |

## M0 census, M-deps, M0-link (2026-10-01)

**libevent** (`deps/build-libevent.sh`):
- **Native gate.** `make verify` must pass. Rule for a failed test: it is rerun alone, 3 times plain
  and 3 times under `EVENT_DEBUG_MODE`, and must pass all six.
  - `util/monotonic_prc_fallback` asserts two consecutive monotonic reads are under 1 s apart. It
    failed in `make verify` on a loaded host and passed 6/6 alone, twice.
- **Out of scope.** `dns/*`, `http/*` and `rpc/*` failures are reported, not gated: memcached calls
  none of them. `host/build-domain.sh` proves that on the image: 0 evdns/evhttp/evrpc functions
  linked, while the same pattern finds 244 in `libevent.a`.
  - `dns/getaddrinfo_cancel_stress` asserts that some of 1000 lookups were cancelled before
    completing. It failed 0/6 alone on a loaded host, and passed in another run.
- **Cross build.** 13 cast sites; `regress` links as a domain.
- **`config.h`, native vs cross**, 7 lines, all expected:
  - arc4random: glibc only;
  - zlib: native only, used by tests only;
  - `sys/queue.h` and TAILQFOREACH: glibc only, compat used;
  - `issetugid`: musl only.

**memcached** (`host/build-domain.sh`): `memcached.dom` 1,972,744 bytes, sha256 9e763e696db3ee62….
- **Configure.** `--disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs`.
- **Patches.**
  - 0001: alignment.
  - 0002: xxhash's `alignas(8)` on two pointer *variables*, an error where a pointer is 16 bytes.
- **Link gates.** Nothing undefined except `_DYNAMIC`, which every SDK image carries (the threads
  probe and every tshark arm do, and run); no undefined weak symbol.
- **`config.h`, native vs cross**, 4 lines:
  - `HAVE_LIBEVENT_NEW` is unset under cross; no source reads it.
  - `NEED_ALIGN` is set: configure cannot run its probe, so it assumes alignment is needed.
  - `SIZEOF_VOID_P` is 16. It is read only by `restart.h`, for `-e`.

**Census: every lossy cast classified; none reconstructs a pointer on a reachable path.**

| where | sites | class |
|---|---|---|
| memcached `restart.c`, `memcached.c:6024` | 31 | warm restart with `-e`: needs a file mapping, which the runtime refuses (ENODEV) — unreachable |
| memcached `xxhash.h:1969,2402,4475` | 3 | address only: alignment tests and an aligned-offset computation |
| libevent `test/*` | 12 | not in the library |
| libevent `bufferevent_ratelim.c:666` | 1 | address only (a weak-RNG seed), and not linked: 0 bufferevent functions in the image, 102 in `libevent.a` |

**Oracle (`host/mc-harness`, `host/run-oracle.sh`), native half.**
- **Native controls.** Two native runs are byte-identical (1,931,207 bytes). A perturbed value byte
  and a perturbed cas token each change the transcript. `stats pointer_size` reads 64; stderr is
  empty; SIGTERM gives exit 0 in 0.9 s.
- **Two settings changed from the plan, before any domain run:**
  - **Port 21299, not 11211.** On a shared host 11211 may be someone else's server.
  - **`-m 64` (the default), not 16.** Items carry an 80-byte header here and 48 natively, so values
    land in different slab classes. Under memory pressure, eviction could then differ for a layout
    reason rather than a correctness one. Pages are allocated on demand, so `-m 64` costs nothing
    unused.
- **Two harness defects, found before any domain run and fixed:**
  - a 250-byte-key test truncated to a valid key;
  - the long-key error is two lines (`CLIENT_ERROR ...\r\n\r\n`), which shifted every later
    response by one.

## M1–M4 (SIGTERM), default heap: the first domain runs (2026-10-01)

**Run 1** (image 9e763e69…): the server died of SIGSEGV after `get k1 k2 k3`, on the first `gets`.
- The fault record (`CAPSTONE_FAULT_RECORD`, now always set by `run-oracle.sh`): cause 6, a
  misaligned store, at an address 8 mod 16.
- The runtime's symbolizer places it in `do_item_bump` → `lru_bump_async` (`items.c:1295`).
- Cause: the LRU bump queue keeps `lru_bump_entry` records (an item pointer and a hash, 32 bytes) in
  a bipbuffer whose `data[]` starts at offset 24, so every record's pointer sits at 8 mod 16.
- **Patch 0003** aligns `data[]` to 16 under `__CAPSTONE__`.
- The other byte buffers were checked. The logger's bipbuffer records hold no pointer and are
  written only while a watcher is attached; the storage and proxy casts are compiled out.

**Run 3** (image f3aba04f…, patches 0001–0003, default level0 heap with per-object bounds):

| check | result |
|---|---|
| null (two native runs) | identical, 1,931,207 bytes |
| positive (value byte, cas value) | each changes the transcript |
| oracle | **domain transcript identical to native**, 1,931,207 bytes: every phase, including 8 concurrent pipelined connections over 4 workers |
| identity | `pointer_size` 64 natively, 128 in the domain |
| status (SIGTERM) | exit 0 both (capstone-job's record) |
| stderr | empty both |
| stop time | native 0.89 s, domain 2.27 s (the 1 s clock tick, then nine contexts joined) |

**Still open then:** M3's worker-coverage marker and M4's SIGUSR1 stop, both settled below.

## M5 — PRE-REGISTERED before its first boot

M5 runs three arms, three runs each. The arms differ only in the heap the image links; the
memcached objects are the same:

- level0: `-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`, no heap safety;
- shrink: the default level0, per-object bounds;
- sublet: `HEAP=sublet`, `HEAP_LOG 26`.

| | prediction |
|---|---|
| Q1 | all 9 runs give a transcript identical to the native one; identity reads 128; exit 0 on SIGTERM; stderr empty |
| Q2 | the three arms' images differ from one another, and their malloc comes from `level0.c` (level0, shrink) or `sublet_heap.c` (sublet), read from each image's symbols |
| Q3 | one SIGUSR1 run on the shrink arm: `Gracefully stopping` on stderr, exit 0, every thread joined, as native |

If an arm's transcript differs, that is a finding about the arm, not an edit to the script.

**Harness defect, found before any M5 result:** the first arm build linked all three arms against
one runtime. `deps/env.sh` exports `CAPSTONE_SDK`, and the SDK's `capstone-cc` reads that before
anything else, so setting `MC_RUNTIME_DIR` per arm changed nothing. The three images came out with
one hash. The heap-symbol check of that version could not have caught it: it read zero in every arm,
including one that must carry the Sublet heap. The run was stopped and no result was recorded.

`build-domain.sh` now:
- sets `CAPSTONE_SDK` per arm;
- reports each arm's hash, its count of `sh_free` and `sh_carve_block`, and the heap and flags in
  its SDK's CMake cache;
- gates on three pairwise-distinct hashes, with the two Sublet symbols present in the sublet image
  and absent from the shrink image.

### M5 first run (2026-10-01): Q1 fails on one run, Q2 holds, Q3 void

The images were level0 daa68667…, shrink 1c67f7bd…, sublet 4266d329…. The launcher was dev's,
e7e27f49…. The shrink image is no longer f3aba04f… because patch 0004 is now in its source, compiled
out without its define.

| | result |
|---|---|
| Q2 | holds: three distinct hashes (ARM GATE); `sh_free` and `sh_carve_block` in sublet only; each SDK's CMake cache names its heap and flags |
| Q1 | **8 of 9.** level0 3/3, sublet 3/3, shrink 2/3: every transcript identical to native, identity 128, exit 0 on SIGTERM, stderr empty. Shrink run 2 failed: see below |
| Q3 | **void**: the instrument signalled the wrong process |

Every null, positive and identity control fired in every boot.

**Shrink run 2.**
- In phase 3, connection 4 received connection 7's replies for items 36–39, with the same lengths.
- Connection 5 received 180 bytes of binary in place of 180 bytes of its own, then timed out.
- The server's phase-4 counters (`curr_items` 328, `cmd_set` 345, `cmd_get` 342) equal native's.
  So every request was read and served, and the bytes went wrong on the way out.
- The binary is not text. It opens with one iovec wire pair, `{0x420, 8}`, and continues with
  fragments of the same kind cut at irregular boundaries.

**Candidate cause** (signature match; the run itself was not traced): the launcher's `msg_call`
rebuilt every sendmsg and recvmsg msghdr in two function statics, shared by the launcher's
per-context threads. Connections 4 and 7 are on different workers under round-robin dispatch. Lane
branch `launcher-msg-call-reentrant` has:
- the history note `docs/history/01-10-2026_16-30-00_launcher-msg-call-shared-msghdr.md`;
- the fix (the two become locals);
- a pre-registered matched pair in one boot. The stock launcher mixed 3/1200 recvmsg and 4/4000
  sendmsg messages, each carrying another thread's bytes. The fixed one mixed 0/1200 recvmsg and
  0/1200 sendmsg, with its 1000-round arms pending.

**Q3 void.** The harness sent SIGUSR1 to capstone-job, which forwards SIGINT, SIGTERM and SIGHUP
and nothing else (`runtime/linux/job.c`). USR1 killed the helper (`signal=10`, 0.00 s, no status
record) and never reached the domain.

The same shape reproduces natively: memcached under a forking `bash -c` wrapper, USR1 sent to the
wrapper. The wrapper dies and memcached is orphaned.

The harness gained `--signal-child`, which signals the server command's child (its process group
when it leads one). `run-oracle.sh` uses it for USR1 only, so the TERM path stays the one M5 ran.
Natively, through the same wrapper, it gives `Gracefully stopping`, exit 0 from both processes,
and a transcript identical to native's.

### M5, M4 and M3 on the fixed launcher (2026-10-01)

The images and Q1–Q3 were the same as the first run. The launcher was 63e8a39a… (lane branch
`launcher-msg-call-reentrant`), with capstone-job 5a160efa…. Each boot prints the hashes the guest
actually runs.

| | result |
|---|---|
| Q1 | **holds, 9/9**: level0 3/3, shrink 3/3, sublet 3/3. Every transcript is identical to native (1,931,207 bytes); identity 128; exit 0 on SIGTERM; stderr empty. Domain stop 2.2–2.3 s (level0, shrink), 3.0 s (sublet); native 0.9 s |
| Q2 | holds (unchanged images) |
| Q3 | **holds**: shrink, SIGUSR1 to capstone-exec's process group. `Gracefully stopping` on stderr, exactly native's 20 bytes; exit 0; transcript identical; stop 2.29 s against native's 0.96 s. Exit 0 after that line means `stop_threads()` returned (`memcached.c:6236-6237`), and it stops and joins every background thread and reaps each worker (`thread.c:206` onward) before main returns |

Every null, positive and identity control fired in each of the five boots.

**M3 worker marker (W1–W3, predictions committed first in 5013e7e02fc3).** Images: native
4a939832…, domain d35bee6b… (shrink runtime); `build-marker.sh` showed both carry the marker and no
oracle image does.

| | result |
|---|---|
| W1 | **holds**: native, two runs, 0:3 1:2 2:2 3:2 each |
| W2 | **holds**: domain 0:3 1:2 2:2 3:2, every worker 0–3 present |
| W3 | **holds**: the transcript is identical to the oracle's native one, and stderr is empty once the marker lines are removed |
| control | the same gate fails on a stderr with no marker line |

**M3 and M4 are complete.** M5 passed 9/9 on the fixed launcher, but that is not evidence the fix
resolves the shrink-run-2 failure. At one run in nine, nine clean runs happen about a third of the
time, and the harness binary changed between the two series.

The experiment that ties the failure to the mechanism, or does not, is memcached under the two
delay launchers. It is pre-registered below.

### memcached under the delay pair — PRE-REGISTERED before its boot

The predictions MD1 and MD2 are in lane branch `launcher-msg-call-reentrant`'s history note
(`docs/history/01-10-2026_16-30-00_launcher-msg-call-shared-msghdr.md`), committed before the boot.
The setup:
- `run-oracle.sh --arm shrink --runs 10 --alternate <stock+delay> <fixed+delay>`;
- odd runs stock+delay e92dedfd…, even runs fixed+delay 584078a4…, one boot.

The harness now stops waiting on a connection after its first 60 s timeout (`fill`, the `dead`
flag), so a desynchronised connection costs one timeout, not one per response it is still owed.
Passing runs are unaffected.

**Result.** stock+delay: 5 of 5 runs differ from native, every one with phase-3 connections carrying
other connections' replies (7 or 8 of 8 per run, and non-text bytes in one run). fixed+delay: 5 of 5
identical to native.

MD1's clause "phase-4 counters equal native's" missed in one stock run. There, two connections lost
their framing and the harness closed them, likely before the server had read their last pipelined
commands. MD2 holds.

So the shrink-run-2 signature is the shared msghdr's. It reproduces in every stock+delay run and
never with the fix. Details: the launcher branch's history note.

## M3 worker marker — PRE-REGISTERED before its first boot

The instrument is patch 0004. A build with `-DMC_CAPSTONE_WORKER_MARKER` prints
`MC-WORKER <index> conn` to stderr when a worker thread sets up a connection. Every other build
compiles it out.

`host/build-marker.sh` builds a native and a domain marker image (domain on the shrink runtime) in
`$MC_WORK/marker`. It checks both directions:
- both marker images carry the format string;
- none of the oracle images do.

`run-oracle.sh --marker` runs those two images.

The harness makes nine connections, in a fixed order:
- `c0` for phases 1, 2 and 4;
- then 8 for phase 3, each dialled before the next.

memcached's dispatcher hands connections out round robin starting at worker 0
(`thread.c`, `last_thread`). With `-t 4` that gives:

| | prediction |
|---|---|
| W1 | native marker run: 9 marker lines, per-worker counts 0:3 1:2 2:2 3:2 |
| W2 | domain marker run: the same counts; every worker 0–3 present |
| W3 | both marker transcripts identical to the oracle's native transcript (the marker writes only stderr), and stderr empty once the marker lines are removed |

The order of the lines on stderr is not predicted: each worker prints from its own thread.

**Control:** the same gate, applied to the stderr of an oracle run (no marker, zero lines), must fail.

## Safety (2026-10-01): 90 of 90 as pre-registered

Ten fixtures (`src/mcapp-safety.c`) ran on all three heap arms, three boots each, in a worker of a
running server, through the hidden command of patch 0005. Predictions were pushed first, in
92a837b435f4. Results, images and the premise that was wrong are in
`ports/memcached/app/results/2026-10-01-qemu-safety/`.

- **shrink** faults on heap overflow and one-past-the-end, and has no temporal safety.
- **sublet** faults on those, on use after free and reuse, and on a stale free (inside `free`).
- **level0** faults only on the global and stack controls.
- **memcached's slab items** (fixtures 9, 10) are unprotected on every arm. An item is bounded to
  its 1 MiB slab page on shrink and sublet, and to the arena on level0, and the slab never frees.
  That is the S1 gap.

Fixtures 1–8 and their predictions are tshark's, carried over; 9 and 10 are new. An independent
audit resolved every fault pc into the fixture's access functions (sublet fixture 6 into `sh_free`'s
probe), and doctored the classifier's inputs to show its gates fire.

Corpus case 02 over the protocol is open: paused, awaiting the lead.

## S1 — Sublet inside memcached's own allocators — PRE-REGISTERED before its first boot

The gap the safety fixtures measured: slab items are bounded to their 1 MiB page and never revoked.
The allocators component port (`ports/memcached/allocators`, patch 0002) hooks exactly the
transitions the slab reuses chunks on. S1 carries those hooks into the server:

- **patch 0006** (`-DMC_CAPSTONE_SLAB_SUBLET` only): the component's hooks, hunk for hunk, on the
  app's own `slabs.c` and `cache.c`, each behind the define with upstream's line in the `#else`;
  plus the adapter's init before `slabs_init` in `main`. `host/build-slab-sublet.sh` has a DRIFT
  gate: patch 0006's hook calls equal patch 0002's, name for name and count for count, and a
  self-test shows the gate refuses a patch missing one hook;
- **the glue** `src/slab-sublet/mcapp-slab-sublet.c`: the component's ledger, metadata heap and
  Sublet authority compiled unchanged (the ledger takes the item layout from the app's
  `memcached.h` through a one-line `mc_slabs_shim.h`); the 64 MiB payload lent LINEAR by the
  Sublet runtime heap at `HEAP_LOG 27`; 16 MiB of metadata from `malloc`; one lock around every
  hook (the ledger is unlocked statics, and `cache.c` runs without `slabs_lock`); the mode from
  `MC_SLAB_SUBLET_MODE`, required (0 spatial: per-chunk bounds; 1 sublet: revoke on every release
  and issue); an adapter refusal is one line and exit 97;
- **two images** on one build: `memcached-slabsublet.dom` for the oracle and
  `memcached-safety-slabsublet.dom` with the fixture hook. HEADERS, ARM and LINK gates as in the
  script's header.

**Run settings, native and domain alike:** `-m 48` (the adapter's page half is 48 MiB) and
`-o no_slab_reassign` (the page mover is not hooked and walks a page by pointer arithmetic).
`CAPSTONE_REV_NODES=16777216`: carving is eager, about two nodes per chunk, so this is a QEMU-only
arm; silicon's pool of 65,536 cannot hold it. In 1.6.45 the server itself never calls `cache.c`
(`cache_create` appears only in `testapp.c`), so the slab hooks are the ones that matter at run time.

The launcher is the fixed one (63e8a39a…, PR #176), as for the M5 second series. The runtime is
the one `deps/env.sh` built for this tree.

| | prediction |
|---|---|
| W1 | oracle, mode 0, 3 runs in one boot: transcript identical to native (same flags), identity 128, exit 0 on SIGTERM, stderr empty |
| W2 | oracle, mode 1, 3 runs: the same. Phase 2's 900 KB value is a chunked item, so the chunked release hook runs on every arm |
| S1 | safety fixtures 1–8 on both arms: the sublet arm's verdicts |
| S2 | fixture 9: **FAULT oob** on both arms (per-chunk bounds) |
| S3 | fixture 10: RETURN `a0015b` in mode 0; **FAULT temporal** in mode 1 |
| R | one oracle run with `MC_SLAB_SUBLET_REPORT=1`: the exit line shows pages > 0 and chunk_releases > 0 |

S2 and S3 are the positive controls that the hooks are live; the plain sublet arm's RETURN on 9
and 10 stays the negative control. An oracle difference, or an adapter refusal, is a finding about
memcached or the hooks, not an edit to the predictions.

### S1 result (2026-10-01; predictions committed first, 1cb89fb5ddf7)

Images: oracle 90d0da59…, safety 2193cf9e…; launcher 63e8a39a…. Everything as pre-registered:

| | result |
|---|---|
| W1 | holds: mode 0, 3/3 identical to native, identity 128, exit 0, stderr empty |
| W2 | holds: mode 1, 3/3, the chunked value included |
| S1 | holds: fixtures 1–8 as the sublet arm, 3 boots per mode |
| S2 | holds: fixture 9 faults oob on both modes, 3/3 each (it returns on every other arm) |
| S3 | holds: fixture 10 returns `a0015b` in mode 0 and faults temporal in mode 1, 3/3 each |
| R | holds: `pages=9 chunk_releases=20 chunk_reuses=20 object_releases=127 object_reuses=111` |
| 11 | holds (registered after the first boots' audit, 6c73bc73fb61): the chunked item's release returns `b0015b` in mode 0 and faults temporal in mode 1, 3/3 each, on the rebuilt safety image 286e2211… |
| guard | a run with no `MC_SLAB_SUBLET_MODE` printed `fail 903` and exited 97 before listening |

Item bounds went from the page (`[c8800000,c8900000)` on the sublet arm) to the chunk
(`[cc0fff00,cc0fffd0)`, 208 bytes).

**Two sentences above are wrong and stay as written.** The server does call `cache.c`, through
`do_cache_alloc`/`do_cache_free` for every worker's read buffers and IO objects (upstream
`memcached.c:398-432, 1047-1199`, `thread.c:314,327`; caches made at `thread.c:454,471`); the grep
that produced the sentence was cut by `| head`. And W2's "the chunked release hook runs" is false:
the oracle never frees its large values, so that hook never ran. Fixture 11 (below) was added for
it. The predictions did not rest on either sentence. Details and the limits (QEMU only,
`-m 48`, no page mover) are in `ports/memcached/app/results/2026-10-01-qemu-slab-sublet/`.

**S1 is done.** Open: stretch S2 (upstream `t/*.t` against the domain) and corpus case 02 over the
protocol. Both are paused awaiting the project lead; neither is blocked by the platform.

For S2 one risk is retired: the plan's risk 7 asked whether the guest's SSH server would refuse
port forwarding. It does not. The guest dropbear is built with `DROPBEAR_SVR_LOCALTCPFWD` and
`DROPBEAR_SVR_REMOTETCPFWD` (`default_options.h:71-72`), and the `dropbearmulti` the VM runs carries
both `direct-tcpip` and `tcpip-forward`. So a host-side harness can reach a domain server's port.

## Milestones

| | content | gate |
|---|---|---|
| M0 | census of memcached + libevent (`TS_CENSUS=1`), the findings above, the threads probe | every lossy cast site classified; P1-P4 |
| M-deps | libevent in `ports/memcached/app/deps/` after the wireshark template | native `make check`; cross build; `regress` links as a domain |
| M0-link | memcached configured `--disable-extstore --disable-proxy --disable-tls --disable-sasl` | `.dom` links, no undefined or weak symbol |
| M1 | listener up, first `-t 1` with the background threads off, then the defaults | `VERSION 1.6.45`; no unclassified unserved syscall |
| M2 | one connection: storage, arithmetic, meta commands, values across slab classes | transcript identical to native (CAS uniques as per-connection ordinals) |
| M3 | 2 x `-t` concurrent connections | every worker serves; transcripts identical |
| M4 | SIGTERM and SIGUSR1 | exit statuses and stderr match native |
| M5 | level0, shrink, sublet; N = 3 | identical; null, positive and identity controls fire |
| Safety | fixtures pre-registered in `host/safety-expect.txt`; corpus case 02 over the protocol | outcomes as predicted — **fixtures done, 90/90**; case 02 open |
| S1 | Sublet inside slabs.c and cache.c (patch 0006, `host/build-slab-sublet.sh`) | oracle identical on both modes; fixtures 9, 10 and 11 flip to FAULT — **done: oracle 6/6, fixtures 60/60 then 18/18** |
| S2 | upstream `t/*.t` against the domain server | open; the guest's SSH forwarding is confirmed available |
