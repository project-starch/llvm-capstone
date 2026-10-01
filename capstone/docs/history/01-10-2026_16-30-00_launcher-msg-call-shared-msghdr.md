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

The binary decodes as little-endian `{offset, length}` pairs (0x420/8, 0x430/14, …): the wire form of
a msghdr's iovec table, sent as if it were data.

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
sends whatever lies at the other call's offsets, such as an iovec pair table. recvmsg has the same
window after the call, where it copies `iov` back into the exchange region.

Everything else per call is per context:
- the domain marshals into its `__thread` exchange (`musl-capstone/runtime/delegate.c`);
- the bounce buffer is per `host` (`delegate-service.c`, `host->bounce = malloc(...)`);
- `vector_call` (readv/writev) keeps its iovec array on the stack.

The threads plan's lock audit (Q6/T3, `delegation-threads.md`) covered the domain's shared state:
the heap, mmap tables, spawn blocks and stdio. It did not cover the launcher's function statics.

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
