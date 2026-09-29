# Delegated sockets: rows like files, one length rule, one packed block

Status: IMPLEMENTED, 2026-09-30, on `delegation-sockets` (over
`delegation-cheap-rows`). The contract below passes 11 of 11 in the guest and
natively; the application gate, the binfmt contract, the signal contract (30
modes, three of them new) and the Perl and mruby round counts are unchanged on
the same images; CPython's `test_epoll` is 10 of 10, `test_selectors` 77 of
121, `test_socket` 96 of 740 with 350 of the errors threads. The record is
`runtime/tests/application/results/20260930-sockets.json`. Three commits
rather than the five planned: the word rule, the block and the rows share
files with the launcher's and the libc's call sites, and a commit that builds
is worth more than one that splits them.

## Responsibility

Linux does everything that is networking: address families and protocols,
ports and their privileges, connection state, buffering, readiness, socket
options, credentials, name-resolution transport, namespaces and the firewall.
A socket is a descriptor in the task's descriptor table, and the task is the
launcher's process, exactly as an open file is today. The monitor and the
kernel module know nothing about sockets; they do not see descriptors now and
will not see them then.

The launcher checks its own responsibility only, as it does for every row:
the wire (offsets and lengths inside the exchange region, including the two
new ways a length can be given), and its private descriptors, in every
descriptor position and inside an `SCM_RIGHTS` control message on the way
out. It does not look at family, type, protocol, address, port or option.
Whether a domain may reach a network is the operator's decision in Linux, a
network namespace, a firewall rule or an outer seccomp filter, not a profile
in the launcher. This retires the sentence in `delegation-abi.md` that made
sockets a profile a port asks for in its descriptor: one version, the rows
exist, the seccomp allowlist is the table's delegated group as it is now.

The libc is responsible for the two things the wire cannot carry as they are,
because our pointers are 128 bits and the kernel's are 64: a length that lives
behind a pointer (`socklen_t *`), and a structure that contains pointers
(`msghdr`). And for one rule Linux would otherwise never need: a datagram is
never silently cut to the exchange region.

## What crosses: 19 rows

RV64 numbers; shapes in the table's notation. Twelve rows need nothing new:

| row | shape |
|---|---|
| `socket` 198 | `{I, I, I}` |
| `socketpair` 199 | `{I, I, I, OUT_FIX(8)}` |
| `bind` 200, `connect` 203 | `{I, IN_ARG(2), I}` |
| `listen` 201, `shutdown` 210 | `{I, I}` |
| `setsockopt` 208 | `{I, I, I, IN_ARG(4), I}` |
| `sendto` 206 | `{I, IN_ARG(2), I, I, OPT_IN_ARG(5), I}` (`send` is `sendto` with no address) |
| `epoll_create1` 20 | `{I}` |
| `epoll_ctl` 21 | `{I, I, I, OPT_IN_FIX(16)}` (`EPOLL_CTL_DEL` takes a null event) |
| `epoll_pwait` 22 | `{I, OUT_SCALED(2, 16), I, I, OPT_IN_FIX(8), I}`, copied back for the events the kernel counted; with a mask it is a temporary-mask wait like `ppoll` and `pselect6`, and the set size must be 8 |

`OPT_IN_ARG` is a macro for two things the argument struct already expresses
(an optional buffer whose length is an argument). musl issues `epoll_create1`
and `epoll_pwait` for `epoll_create` and `epoll_wait` on riscv64, so three
epoll rows cover the interface. `epoll_event` is 16 bytes on riscv64 and 12,
packed, on x86_64: the launcher converts on a non-riscv host as `stat_call`
does for `stat`, so the native tests stay meaningful.

Five rows need the new length rule of the next section: the address or option
buffer's length is the value of a 32-bit word the caller also passes, and the
kernel updates that word.

| row | shape |
|---|---|
| `accept` 202 | `{I, OPT_INOUT_WORD(2), OPT_INOUT_FIX(4)}` |
| `accept4` 242 | `{I, OPT_INOUT_WORD(2), OPT_INOUT_FIX(4), I}` |
| `getsockname` 204, `getpeername` 205 | `{I, INOUT_WORD(2), INOUT_FIX(4)}` |
| `recvfrom` 207 | `{I, OUT_ARG(2), I, I, OPT_INOUT_WORD(5), OPT_INOUT_FIX(4)}` |
| `getsockopt` 209 | `{I, I, I, OPT_INOUT_WORD(4), OPT_INOUT_FIX(4)}` |

The buffers are INOUT, not OUT, on purpose: the kernel writes only as many
bytes as the object has, and the domain's bytes beyond that must come back
unchanged rather than as whatever the exchange region held. The copy in costs
at most the requested length, 128 bytes for any address.

Two rows carry a packed block, the way `readv` and `writev` carry a flattened
`iovec` array today:

| row | shape |
|---|---|
| `sendmsg` 211 | `{I, IN_ARG(3), I}` |
| `recvmsg` 212 | `{I, INOUT_ARG(3), I}` |

The kernel's form is `(fd, msghdr *, flags)`; the wire form puts the block's
offset in argument 1 and its length in argument 3, a slot the three-argument
syscall does not use. The libc's dispatcher converts one into the other in its
`switch`, as it does for the vector forms and for `pselect6`'s mask pair, so a
program that issues `syscall(SYS_sendmsg, ...)` directly is served too.

Not rows: `sendmmsg` and `recvmmsg` (musl has the wrappers, no port issues
them; they stay unknown and answer ENOSYS), and every socket `ioctl` beyond
`FIONBIO` and `FIONREAD`, which the buffer form admits already. `getifaddrs`
and `if_nameindex` in musl speak netlink over `socket`, `send` and `recv`,
the `sendto` and `recvfrom` rows, so they need no ioctl either.

## The length rule: `CAPSTONE_LEN_WORD`

The table knows three ways to size a buffer: a constant, an argument's value,
and an argument's value times a scale. The fourth is the value of a 32-bit
word in the exchange region, and the row names which argument holds that
word's offset. The word's own row entry is a fixed four-byte INOUT buffer, so
its offset is bounded by the existing rules before the word is read.

Evaluation needs the exchange region, which the length functions do not see
today: `capstone_delegate_arg_bytes` gains a pointer to it, the libc passes
its exchange, the launcher passes the bounce copy it validates from already,
so the word the validator reads is the word the call uses. `validate` bounds
constant and argument lengths first, then word lengths. A null buffer with a
null word is what `accept(fd, NULL, NULL)` sends and is allowed; a null buffer
with a word is passed as the kernel takes it, the word untouched.

The libc clamps the word's value to the room left in the region, as it clamps
an argument length today. The kernel reports the object's true length in the
word regardless, which is the truncation semantics POSIX specifies for these
calls, so a caller sees exactly what Linux would have told it.

The wire test's consistency check learns the rule: a word length must refer
to an argument whose entry is a fixed four-byte buffer.

## The block: `msghdr` flattened

`common/msghdr.c`, packed and validated in common code like the spawn block,
tested in the wire tests like `spawn-test.c`. Little-endian 64-bit words:

    name_offset  name_length  iov_count  control_offset  control_length  flags
    then iov_count pairs of  offset  length

The libc packs: the name (if any) and the control bytes copied into the
region, every `iov_base` copied for `sendmsg` and reserved for `recvmsg`,
offsets in place of pointers, at most 1024 entries as for the vector forms.
The launcher rebuilds a `struct msghdr` and an `iovec` array over its bounce
buffer, calls, and for `recvmsg` copies the bytes the kernel wrote back into
the region and writes the updated `msg_namelen`, `msg_controllen` and
`msg_flags` into the block, which the INOUT rule copies to the domain. The
libc then unpacks the block into the caller's `msghdr`.

Control messages cross as bytes. On `sendmsg` the launcher walks the `cmsghdr`
chain for `SCM_RIGHTS` and refuses any descriptor of its own with EBADF, the
check the spawn block's descriptor list already has; `SCM_CREDENTIALS` needs
no rule, the kernel accepts only the sender's own pid and ids from an
unprivileged process, and the sender is the task. On `recvmsg` a received
descriptor lands in the task's table and is the task's; `MSG_CMSG_CLOEXEC`
passes as a flag.

## Blocking, readiness, signals

Nothing new. `connect`, `accept4`, `recvfrom`, `recvmsg` and a full-buffer
`sendmsg` are blocking rounds like `read`; the launcher's stub rule and the
libc's retry round give them the same `EINTR` and restart behaviour the signal
contract proves for `read`. `SO_RCVTIMEO` and `SO_SNDTIMEO` are `setsockopt`
with a 16-byte `timeval` and time out in the kernel's call. Readiness is
`ppoll` and `pselect6`, which exist, and the three epoll rows. Non-blocking
is `SOCK_NONBLOCK`, `fcntl(F_SETFL)` in its integer form, or `FIONBIO`, all
of which cross today. The kernel's `SIGPIPE` for a send to a closed peer
reaches the domain like any other signal; `MSG_NOSIGNAL` is a flag.

`epoll_event.data` crosses as its 64 bits: a descriptor or an index survives,
a capability stored there does not keep its tag. CPython's `selectors` and
`asyncio` store descriptors.

## The datagram rule

A stream send may be short, and the libc chunks a long `write` today. A
datagram is one unit: `sendto` or `sendmsg` with a message longer than the
room left in the exchange region is answered EMSGSIZE by the libc, the
kernel's own answer for a datagram too long for the protocol, never clamped.
`recvfrom` or `recvmsg` with a buffer longer than the room is capped to the
room, and a longer datagram is then cut by the kernel with `MSG_TRUNC` set,
which is how Linux reports truncation. The region's size is the image
descriptor's, between 4 KiB and 1 GiB; the contract's datagram mode reports
the room it found next to its result, so the record carries the number rather
than this plan assuming one.

## Scope: none, and what that means

What a domain's socket can reach is what the launcher's process can reach.
`AF_UNIX` paths resolve against the launcher's working directory, as `openat`
does. Ports below 1024 need the launcher's capability, as for any process.
`SO_PEERCRED` and `SCM_CREDENTIALS` name the launcher's pid and ids, which
is correct: `getpid` answers the same. The launcher's own descriptors, the
spawner socket and the image memfd among them, stay out of reach through the
private-descriptor check in every position, `SCM_RIGHTS` included.

The allowlist changes by nothing but the rows: the seccomp filter is derived
from the table; the five numbers the launcher already admits for its spawner
(`socketpair`, `sendto`, `recvfrom`, `sendmsg`, `recvmsg`) become ordinary
delegated rows and the duplicate entries are harmless. The wire test's
assertion that `socket` is unknown flips, and the table's comment that sockets
wait for a profile goes.

## Deviations, named

1. `epoll_event.data` carries 64 bits; a program that stores pointers there
   indexes through a table instead. musl's `struct epoll_event` on capstone64
   is 32 bytes, because the data union holds a pointer; the libc converts to
   and from the kernel's 16 (found in the guest: the first run returned the
   wrong data word).
2. `sendmmsg` and `recvmmsg` are unknown.
3. A datagram longer than the exchange region's room: EMSGSIZE on send,
   `MSG_TRUNC` on receive, Linux's own signals at a size Linux would not
   have imposed.
4. Address and option lengths are clamped to the room; the kernel's reported
   length is not.
5. In the guest the VM's user network is `restrict=on`: the contract proves
   loopback TCP and UDP, Unix sockets and `/etc/hosts`, not a name resolved
   across a network. musl's resolver transport, a bound UDP socket with
   `sendto` and receives under `poll`, and a TCP fallback, is the datagram
   and stream rows above.
6. Threads are not this branch: CPython's `socketserver` mixins and
   `asyncio`'s executor-backed `getaddrinfo` keep failing on `clone`, and the
   qualification names those errors as that class.

Two findings on the way that are not socket-specific. A file action names a
descriptor in the child's table, where the launcher's own never arrive, so
the spawn block's check of a `DUP2` target or a `CLOSE` against the
launcher's private descriptors was wrong; only a `DUP2` source and an
`FCHDIR` directory are checked now (found by the `inherit` mode, whose
listener landed on descriptor 3). And every copy between the caller's memory
and the exchange region is a data copy, `dl_bytes`, never the libc's
tag-preserving `memcpy`: a buffer that held a pointer before the call
carried its tag into the region, the launcher's stores changed the bytes,
capstone-qemu kept the granule's tag, and the copy back loaded the old
pointer instead of the kernel's bytes (found through CPython's `recv_fds`,
whose control buffer is pymalloc's, and through `inet-dgram`'s address on an
uninitialized stack slot). The contract's `tagged-buffer` mode guards it.

## Contract tests: the definition of done

`tests/application/socket-contract.c` with a runner `run-sockets.py` in the
shape of `run-signals.py`, one mode per line, every mode compiled and run on
Linux first, where Linux is the oracle the test claims to encode.

- `unix-stream`: socket, bind in `/tmp`, listen, connect, `accept4` with an
  address and a length word that reports the full length, bytes both ways,
  `getsockname` and `getpeername`, `shutdown(SHUT_WR)` seen as EOF, close.
- `unix-dgram`: two bound datagram sockets, `sendto`/`recvfrom` with the
  address, a receive into a too-small address buffer: the word says the full
  length, the name is cut, the data is intact.
- `inet-stream`: TCP on 127.0.0.1 port 0, `getsockname` shows the port,
  `SO_REUSEADDR` set and read back through a four-byte word, `SO_TYPE`, a
  non-blocking connect to a closed port: `EINPROGRESS`, `ppoll`, `SO_ERROR`
  is `ECONNREFUSED`.
- `inet-dgram`: UDP on loopback, a datagram of the largest size the room
  allows, printed with the room; one longer than the room is EMSGSIZE; a
  receive into a smaller buffer sets `MSG_TRUNC` (`recvmsg`).
- `scm-rights`: a `SOCK_SEQPACKET` pair, `sendmsg` with two iovecs and a pipe
  end in `SCM_RIGHTS`, `recvmsg` receives a descriptor that writes into the
  pipe, `MSG_CMSG_CLOEXEC` shows in `F_GETFD`, a control buffer too small sets
  `MSG_CTRUNC`.
- `epoll`: `epoll_create1`, `ADD` a socket, `epoll_pwait` with timeout 0 is
  0, after a write 1 with the data word intact, `MOD`, `DEL` with a null
  event.
- `select-poll`: `pselect6` and `ppoll` report a readable socket.
- `nonblock`: `SOCK_NONBLOCK` accept is EAGAIN, `FIONBIO`, `FIONREAD` after a
  write, `SO_RCVTIMEO` of 100 ms makes `recv` return EAGAIN after at least
  100 ms.
- `inherit`: the task listens and spawns its own image in `inherit-child`
  mode with the listening descriptor on 3; the child accepts, the parent
  connects, one byte crosses: descriptors reach children as pipes do.
- `hosts`: `getaddrinfo("localhost")` through `/etc/hosts`, a numeric host,
  `getnameinfo` numeric.
- In `signal-contract.c`, two modes mirroring `during-read-restart` and
  `during-read-eintr` for a blocking `recv`, and `epoll-pwait` mirroring
  `ppoll` for the masked wait.

Native launcher tests, one case per row as in `delegate-rows-test.c`, plus:
the private descriptor refused in every position and inside `SCM_RIGHTS`; the
wire tests for the word rule (word offset outside the region, word value
beyond the region, null buffer with null word, the consistency check) and for
the block (bad offsets, more than 1024 entries, control longer than the
region); the epoll layout conversion on the build host.

## Qualification

CPython, measured before and after on the same images as
`results/20260930-cheap-rows.json` was: `test_socket` through
`host/run-filtered.py`, `test_selectors`, `test_epoll`, `test_os`'s sendfile
tests (their setup is `socketpair`), and the `asyncio` modules that run
without threads; the errors that remain are named by class (threads, SSL,
which has no OpenSSL in the image, network beyond loopback). Perl's socket
tests are a second port's view once the `Socket` module is in its image.

## Size and order

Five commits, each standing on its own: the word rule with its wire tests;
the block with its wire tests; the rows with the libc's two dispatcher cases,
the launcher's rebuild and the native cases; the socket contract and runner,
native first, then the guest; the qualification record with the README
numbers. No change to the monitor, the module, or musl's patches.
