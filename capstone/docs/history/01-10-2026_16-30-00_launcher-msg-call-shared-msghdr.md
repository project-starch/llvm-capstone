# The launcher's sendmsg and recvmsg shared one msghdr across contexts (2026-10-01)

## Symptom

memcached 1.6.45 as a full application (lane `memcached-app`, M5, shrink arm, run 2 of 3) produced a
transcript that differed from native in phase 3, where eight connections run at once over four workers.

- Connection 4 received connection 7's replies for items 36–39 (`VALUE c7:36 36 12` / `conn7-item36`)
  in place of its own, with the same lengths.
- Connection 5 received 180 bytes of binary in place of 180 bytes of its own replies, then lost its
  place in the protocol and timed out.

The server's counters at the end (`curr_items` 328, `cmd_set` 345, `cmd_get` 342) equal native's, so
every request was read and executed correctly. The bytes went wrong on the way out.

The binary is not text. Its first 16 bytes are one little-endian `{offset, length}` pair, `{0x420, 8}`:
the wire form of an entry in a msghdr's iovec table. The rest are fragments of the same kind, cut at
irregular boundaries: offsets stepping by 0x10 (0x430 … 0x5d0) and lengths of 5 to 62.

That is what a reply looks like when its iovecs are each read from the wrong place in a region that
holds such a table. It is consistent with the mechanism below, not proof of it.

The other eight M5 runs (level0 3/3, shrink 2/3, sublet 3/3) were identical to native.

## Mechanism (hypothesis, under test below)

`runtime/linux/delegate-service.c`, `msg_call`, serves sendmsg and recvmsg. It rebuilds a host
msghdr over the context's bounce buffer in two **function-static** objects:

```c
static struct capstone_msghdr_view view;
static struct iovec iov[CAPSTONE_MSGHDR_IOVS];
```

Since the threads chain, each context is served by its own launcher thread
(`delegated-threads-model.md`; `delegate-service.c`, "the Linux thread serving this context"). So two
contexts in sendmsg at once write the same `view` and `iov`.

If thread A is preempted between filling `iov` and the kernel copying it at `sendmsg` entry, and
thread B fills it meanwhile, then A's socket carries B's bytes. A view mixed between the two calls
sends whatever lies at the other call's offsets, which can be an iovec pair table. recvmsg has the same
window after the call, where it copies `iov` back into the exchange region.

Everything else per call is per context:
- the domain marshals into its `__thread` exchange (`musl-capstone/runtime/delegate.c`);
- the bounce buffer is per `host` (`delegate-service.c`, `host->bounce = malloc(...)`);
- `vector_call` (readv/writev) keeps its iovec array on the stack.

The threads plan's lock audit (Q6/T3, `delegation-threads.md`) listed the heap, mmap tables, spawn
blocks and stdio. It did not list the launcher's function statics. Nor did it list a domain-side
one found by this note's audit (`dl_epoll_pwait`, below), so "the domain's shared state is audited"
is not a claim this note can make.

The statics date from 1115299cd43e (delegated sockets), when one thread served every call.

## The fix

`view` and `iov` become locals of `msg_call`, as `vector_call`'s are. The launcher is glibc-linked,
and its context threads are made with default attributes, so 8 MiB stacks; the two locals take
about 32 KiB.

## Matched pair — PRE-REGISTERED before its boot

There are two launchers, built from this branch's tree in the same way:
- **stock:** the tree without the fix, `capstone-exec` e7e27f49…, byte-identical to the launcher M5 ran;
- **fixed:** the tree with it, 63e8a39a….

There is one probe image: `pthread-probe.dom`, with two new modes.
- `sendmsg-concurrent`: four threads, each with its own `AF_UNIX` stream pair. Each sends 32 iovecs
  of 1 KiB with sendmsg and reads them back with read.
- `recvmsg-concurrent`: write out, recvmsg with 32 iovecs back.

A byte's top two bits name the thread that wrote it. A message counts as mixed when any of its bytes
differs from what the thread sent.

`run-msg-race.sh` uses one boot. Eight arms alternate stock and fixed: sendmsg and recvmsg, at 300
and then 1000 rounds. All are expected to return.

| | prediction |
|---|---|
| R1 | stock, sendmsg: mixed > 0 in at least one of its two arms |
| R2 | fixed: mixed 0 and failed 0 in all four arms |
| R3 | stock, recvmsg: mixed > 0 in at least one of its two arms (less sure than R1: its window is the copy-back after the call) |

How the readings are read:
- **If R1 fails**, the probe does not create the triggering condition, and R2's zeros are void:
  "fixed" would then be unproven, not shown. The next step would be a longer window (more and
  larger iovecs, more threads), not a different conclusion.
- **R1 and R2 together** show that the static msghdr mixes concurrent sendmsg calls and that the
  locals stop it. They do not prove it was the only cause of the memcached transcript. That needs
  memcached runs on the fixed launcher, in lane `memcached-app`.

### Result, first pair (2026-10-01; predictions committed first, 1ac724c19edc)

The inputs were probe 7683fb05…, stock e7e27f49…, fixed 63e8a39a…. The first attempt at this boot
ran nothing: all eight arms exited 127, because the guest's busybox has no `timeout` applet. The
script now uses a shell watchdog, and the boot below is the second.

| arm | rounds | stock: mixed / messages | fixed: mixed / messages |
|---|---|---|---|
| sendmsg | 300 | 0 / 1200 | 0 / 1200 |
| recvmsg | 300 | 3 / 1200 | 0 / 1200 |
| sendmsg | 1000 | 4 / 4000 | 0 / 4000 |
| recvmsg | 1000 | 30 / 4000 | 0 / 4000 |
| **total** | | **37 / 10400** | **0 / 10400** |

No arm in either launcher had a failed call.

Every mixed message carried another thread's bytes. For example:
- sendmsg, thread 1, round 483: byte 0 is 0x35, which is thread 0's;
- recvmsg, thread 2, round 235: byte 31744, iovec 31, is 0x0c, which is thread 0's.

So a message was mixed in part, from whichever iovecs the other call had rewritten.

All three predictions hold:
- **R1** (stock sendmsg mixes): 4 in the 1000-round arm, none in the 300-round arm;
- **R2** (fixed: zero);
- **R3** (stock recvmsg mixes).

Weight:
- **recvmsg** (33 against 0) is strong on its own.
- **sendmsg** (4 against 0) is thin on its own. With equal rates, all four landing in stock has
  probability 1/16, and the four need not be independent events. Treating every mixed message as
  independent overstates both figures.
- The second pair supplies the sendmsg evidence.

## Second pair, window widened — PRE-REGISTERED before its boot

The first pair's sendmsg arms can only show mixing if a launcher thread is preempted inside the
window. On one vCPU that window is the few microseconds between filling `iov` and the syscall
copying it. A clean fixed sendmsg arm is void unless the stock launcher mixes sendmsg under the same
conditions.

So the second pair creates the condition. Both launchers add the same test-only two lines to
`msg_call`, and are not for commit:
- `usleep(200)` before the syscall, between the fill and the kernel's copy of `iov`;
- `usleep(200)` after it, before recvmsg's copy back.

The variants:
- **stock+delay** e92dedfd…: dev plus the two lines;
- **fixed+delay** 584078a4…: this branch plus the two lines.

They differ only in the fix. The sleep blocks the serving thread, so whenever another context has a
call ready, it is served inside the window.

`run-msg-race.sh` uses the same probe image, one boot, alternating, 300 rounds each: stock sendmsg,
fixed sendmsg, stock recvmsg, fixed recvmsg.

| | prediction |
|---|---|
| D1 | stock+delay, sendmsg: mixed > 0, in at least 10% of its 1200 messages |
| D2 | fixed+delay: mixed 0 and failed 0, sendmsg and recvmsg |
| D3 | stock+delay, recvmsg: mixed > 0 |

### Result, second pair (2026-10-01; predictions committed first, 6806764cfaa4)

The inputs were probe 7683fb05… (the same image), stock+delay e92dedfd…, fixed+delay 584078a4….
One boot.

| arm | rounds | stock+delay: mixed / messages | fixed+delay: mixed / messages |
|---|---|---|---|
| sendmsg | 300 | 54 / 1200 | 0 / 1200 |
| recvmsg | 300 | 13 / 1200 | 0 / 1200 |

No call failed in either launcher.

- **D1 holds in kind and misses its bound:** stock sendmsg mixes, but in 4.5% of messages, not the
  ≥ 10% predicted. The bound assumed every sleep would meet another context's call. It is recorded
  as missed, not moved.
- **D2 holds:** the fixed launcher mixes nothing with the window held open 400 µs per call.
- **D3 holds.**

## Conclusion

The two function statics in `msg_call` mix concurrent sendmsg and recvmsg calls from different
contexts. Making them per-call locals stops it in both matched pairs.

| | sendmsg | recvmsg |
|---|---|---|
| production window | stock 4/5200, fixed 0/5200 | stock 33/5200, fixed 0/5200 |
| window held open | stock 54/1200, fixed 0/1200 | stock 13/1200, fixed 0/1200 |

An independent audit (claim-auditor, 2026-10-01) normalised the immediates in both binaries'
disassembly. Of 266 functions, only `msg_call` differs, plus a linker relaxation in
`capstone_spawner_spawn`. The only data difference is that the two statics are gone.

**What this does not show.** The memcached transcript failure has this race's signature:
- connection 4 carried connection 7's replies at equal length, while 7's own were intact;
- connection 5 carried exchange-wire offset/length pairs;
- the two connections are on different workers under round-robin dispatch;
- the server's counters equal native's.

Nothing traced that run, though, so the race is the *candidate* cause, not a shown one.

Nine clean memcached runs on the fixed launcher do not show the fix resolves it. At a rate of one
run in nine, nine clean runs happen about a third of the time anyway, and the harness binary also
changed between those runs.

The discriminating experiment is memcached itself under the two delay variants, alternating in one
boot. It is pre-registered below.

## A separate defect found by the same audit: `dl_epoll_pwait`'s static result buffer

`ports/musl-capstone/runtime/delegate.c` (`dl_epoll_pwait`):
`static struct dl_epoll_wire wire[1024];` receives every context's epoll results, and they are then
copied into the caller's array. The compiled runtime holds it in `.bss`, process-wide, with no lock;
the transport (`dl_entry`, `dl_exchange`) is in `.tbss`.

Two contexts in `epoll_pwait` at once can therefore hand one context the other's events, if one is
preempted between the copy into `wire` and the copy out.

For libevent's level-triggered registrations, memcached's case, this is probably benign:
- a foreign fd is dropped (`evmap.c`: no context for it in this base);
- a lost readiness is reported again by the next wait.

An edge-triggered or one-shot user could lose an event for good. This is **not fixed here**: it is
a separate change to the domain runtime, for the threads lane to take or assign.

## memcached under the delay pair — PRE-REGISTERED before its boot

This is the experiment the audit named as the one that ties the memcached failure to the mechanism,
or fails to. It uses lane `memcached-app`'s `run-oracle.sh --alternate`:
- the shrink image 1c67f7bd…, M5's;
- one boot, ten runs alternating stock+delay (e92dedfd…, odd runs) and fixed+delay (584078a4…,
  even runs);
- each run is the full oracle script, compared with the native transcript.

| | prediction |
|---|---|
| MD1 | stock+delay: at least 1 of its 5 runs differs from native in phase 3 in this race's shape: a connection carrying another connection's replies, or non-text bytes in place of its own. Phase-4 counters stay equal to native's |
| MD2 | fixed+delay: 5 of 5 identical to native |

How the readings are read:
- **If MD1 fails**, the delay does not create memcached's triggering condition, and MD2's clean runs
  are void.
- **MD1 and MD2 together** tie the memcached failure's shape to the shared msghdr. They still do not
  prove the one failure on dev's launcher had no other contributor.

### Result, memcached under the delay pair (2026-10-01; predictions committed first, 5dda6f629ac3)

The run used shrink image 1c67f7bd… and one boot: stock+delay e92dedfd… on odd runs, fixed+delay
584078a4… on even runs. The guest's `/usr/bin/capstone-exec` (e7e27f49…) was not used: each run
named its launcher.

| run | launcher | transcript | phase-3 connections carrying another connection's replies | phase-4 counters |
|---|---|---|---|---|
| 1 | stock+delay | differs | 8 of 8 | equal to native |
| 3 | stock+delay | differs | 7 of 8 | **lower**: `cmd_set` 342 vs 345, `cmd_get` 338 vs 342 |
| 5 | stock+delay | differs | 8 of 8, and non-text bytes on connections 2 and 7 | equal |
| 7 | stock+delay | differs | 8 of 8 | equal |
| 9 | stock+delay | differs | 8 of 8 | equal |
| 2, 4, 6, 8, 10 | fixed+delay | **identical** (1,931,207 bytes) | none | equal |

Every run exited 0 on SIGTERM with empty stderr. Native's null, positive and identity controls
fired.

- **MD1 holds in its shape and misses its counter clause once.** Every stock+delay run carried
  other connections' replies. Run 3's counters came out lower. Its connections 1 and 6 lost their
  framing: 28 and 41 `STORED` lines against native's 40, and neither ends in `MN`. The harness stops
  reading a connection at the number of responses it expects and then closes it. The likely
  consequence is that the server never read the rest of their pipelined commands. That is an
  inference: the server side was not traced. It is recorded as a miss of the clause as written.
- **MD2 holds:** 5 of 5 identical.

This ties memcached's failure shape to the shared msghdr. With the window held open, the stock
launcher reproduces the shrink-run-2 signature in every run, and the fixed launcher never does.
