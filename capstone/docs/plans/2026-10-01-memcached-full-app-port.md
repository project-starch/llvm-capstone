# memcached 1.6.45 as a full application on the delegated runtime

**Status:** M0 open (2026-10-01). Lane branch `memcached-app`.

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
| Safety | fixtures pre-registered in `host/safety-expect.txt`; corpus case 02 over the protocol | outcomes as predicted |
