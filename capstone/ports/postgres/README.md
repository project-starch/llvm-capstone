# PostgreSQL's memory manager, outside PostgreSQL

PostgreSQL the program does not run under capabilities and is not meant to: it
wants an operating system, a file system, sockets and processes. Its memory
manager is another matter. The seven files of `src/backend/utils/mmgr` are a
hierarchical allocator with reset and delete semantics, the only one in this
project's set that exercises a tree, and they come out of the tree whole.

This port is those seven files, unchanged, plus what the backend would have
provided, plus a driver that replays a recording of a real workload against
them.

| Path | What it is |
|---|---|
| `port/pg_stubs.c` | the sixteen symbols the manager references and the backend would have defined. The list is not a guess: it is what the linker asks for when the seven objects are linked with an empty main, and nothing more is here than it asked for |
| `tools/replay.c` | the driver. Reads a trace, makes the same calls in the same order, counts what the manager asks of the level below, and compares that with what the backend's manager asked |
| `a11trace.h` | the trace format. It lives here, beside the replay, because a domain build has to be self-contained; the recorder fetches it from here at a pinned commit |
| `build-mmgr-host.sh` | the host arm: fetch, configure for its generated headers, build, replay |

There is no `adapted/` and no `patches/`: the manager is compiled as it ships.

## What the host arm proves

```
bash build-mmgr-host.sh <trace>
```

The manager is 6 872 lines and links against sixteen definitions and libc. The
replay then makes every call the backend made, in order, and the gate is exact:

| what the manager asked of the level below | in the backend | in the replay |
|---|---:|---:|
| blocks taken | 13 301 | 13 301 |
| blocks given back | 13 113 | 13 113 |
| blocks grown or moved | 1 | 1 |
| blocks held at once, most | 194 | 194 |

That is pgbench's default script, scale 10, one client, 1 000 transactions,
1.48 million calls. The read-only script agrees the same way, at 2 255 blocks.
An allocator driven by the same sequence asks the level below for the same
thing, so the replay is the manager's workload and not an imitation of it.

The recording and the gate that produced those traces are in the paper's
`experiments/a11/postgres`, with what they are scoped to.

## Two things the recording had to get right, and one it must not

**A reset inside a delete is the manager's own.** `AllocSetDelete` resets a
context on its way out when the type keeps a free list of contexts. The method
table sees that reset, but the program never asked for it, and a replay that
calls `MemoryContextDelete` gets it again for free. The recorder does not
record it. Before that was found, the replay reset a context that no longer
existed, 13 009 times in one run.

**A context's creation is not in the method table.** The four `*ContextCreate`
functions are recorded separately, with their size parameters, because a
context created with different parameters produces a different sequence of
blocks, and the sequence of blocks is what is being measured.

**The manager must stay as it ships.** Nothing in this directory is compiled
into the seven files. The Sublet port of them, when it comes, goes in
`sublet/` beside this, applied on top only when asked for, and the unprotected
arm of every measurement stays a build that never read a file from there.

## The census: what a capability build costs

```
bash census-capstone.sh
```

All seven objects compile for `capstone64`, and the cost is two lines and a
set of headers.

**Two lines, and the allocator names them itself.** `aset.c` asserts that its
free-list link fits in the smallest chunk, because that link lives inside the
freed chunk. A capability is sixteen bytes and the smallest chunk was eight, so
the assertion fails. Raising the minimum to sixteen doubles the ladder, so one
size class comes off to keep its top at the 8 KiB chunk limit, which a second
assertion checks. `port/aset-capstone.patch` is those two lines.

This is not Sublet's change and not a workaround: any machine with sixteen-byte
pointers needs it. The consequence is real and belongs in the measurement. The
size classes on a capability machine are 16 to 8192 and not 8 to 8192, so the
sequence of blocks differs from the x86 run, and the host arm's exact block
count does not carry over to the domain.

**Thirteen stub headers, 125 lines.** `c.h` includes fourteen system headers
before it declares anything of PostgreSQL's, and a freestanding build has none
of them. `port/stubinc/` holds the declarations the seven files actually reach.

**The capability-roundtrip warning is not ours.** The compiler flags
`DatumGetPointer`, an inline function in `postgres.h` that casts an integer
back to a pointer. Of the files in the memory manager, only `dsa.c` calls it,
and `dsa.c` is not one of the seven.

**Eleven symbols from libc**, and they are the eleven the plan predicted:
`malloc`, `free`, `realloc`, `memcpy`, `memset`, `strlen`, `strcmp`, `strcpy`,
`strcat`, `strnlen`, `stderr`.

## The freestanding arm

```
bash build-mmgr-domain.sh
```

**It links.** 160 KB of image, 107 KB loadable, no undefined symbol. That was
the question worth answering first, because it is the one another port in this
repository has been stuck on for three weeks, and the answer here is that the
memory manager does not pose it: it needs no `setjmp`, it has no heap of its
own in `.bss`, and its two lines of change are the two the allocator asks for
itself.

| | |
|---|---|
| `port/freestanding/pg_string.c` | the seven string functions the census named, byte at a time, because a word-at-a-time copy reads past the end of the last word and on a capability machine that is a fault |
| `port/freestanding/pg_level0.c` | the level below: first fit over one region, with a free list and coalescing. It has to reuse, or every block would be fresh and the comparison meaningless. `memsys5` would replace it if the measurement ever turns on what level 0 does |
| `port/freestanding/pg_printf_domain.c` | the printf helpers over the payload, with a formatter that understands what is actually emitted and copies out verbatim any conversion it does not know |
| `tools/replay_domain.c` | the driver: four regions in, the payload out, and one way back to the monitor. A fault returns the core with the message already written, because a domain that spins is a host that never reads the payload |

The loop is `tools/replay_core.inc`, the same file the host driver runs. What
differs between the arms is the compiler, the level below, and where formatted
output goes. Nothing else, which is what makes a difference between their
results a difference about capabilities.

**The allocator had the bug the compiler exists to find.** It aligned its
region by masking the address and casting it back, which discards the
capability and faults on the first dereference
(`-Wcapstone-pointer-roundtrip`). It aligns by moving the pointer now. That is
the class that made eighteen of MicroPython's twenty-one patches, met on the
first file written for this port.

**What is still missing to run it.** A host program that makes the four
regions, shares them, enters the domain and prints the payload. The SQLite
port's `sqlite_host.c` is 728 lines because it also services that port's
hostcalls; this one needs the region half of it and nothing else. Then the
trace has to be shortened for a domain: the identity table for a million
objects is twenty megabytes of capabilities, and the trace itself is
fifty-seven, which fits the hundred and thirty a domain can be given but
leaves little room. A hundred transactions still give over two thousand resets
for the scatter plot.

Keeping the host arm working is what makes a fault in the freestanding arm a
fault about capabilities rather than about the port.
