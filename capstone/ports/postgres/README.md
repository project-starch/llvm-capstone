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

Not here yet. What it still needs, in the order it will be needed:

| | |
|---|---|
| a level below | the manager calls `malloc`, `free` and `realloc`. `memsys5` already runs freestanding in a domain in this repository, as level 0 of the SQLite pass, so the two ports would then stand on the same allocator |
| the eleven libc symbols | eight are in `benchmarks/beebs/adapted/beebs_freestanding_string.c`; `stderr` and the printf helpers go through the domain's hostcall write, which the SQLite domain has |
| the trace | 57 MB for pgbench's default script, handed over as a host region the way `memsys5`'s pool is, which needs `cma=` on the guest's command line |
| the link | the globals offset sized to `.text` and the descriptor delivered into `dom_data`, which `build-sqlite-capstone.sh` does for a larger program |

Keeping the host arm working is what makes a fault in the freestanding arm a
fault about capabilities rather than about the port.
