# PostgreSQL's memory manager, outside PostgreSQL

PostgreSQL the program does not run under capabilities and is not meant to: it
wants an operating system, a file system, sockets and processes. Its memory
manager is another matter. The seven files of `src/backend/utils/mmgr` are a
hierarchical allocator with reset and delete semantics, the only one in this
project's set that exercises a tree, and they come out of the tree whole.

This port is those seven files, unchanged, plus what the backend would have
provided, plus a driver that replays a recording of a real workload against
them. There is no `adapted/` and no `patches/`: the manager is compiled as it
ships, and the two patches under `port/` are applied to a copy at build time.

    bash run-pg-gate.sh          # does it still run under capabilities
    bash run-pg-sublet-gate.sh   # does the discipline still hold
    bash census-capstone.sh      # what a capability build costs

Every script says at its head what it does and why. The two gates need no
recording, generate their own workload and are what the nightly runs. For a
run against a real recording, `run-pg-replay.sh` and `run-pg-sublet.sh` take
one as their argument. `sublet/README.md` says why the protected arm looks the
way it does.

## Three things a reader would otherwise get wrong

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

**The capability-roundtrip warning is not ours.** The compiler flags
`DatumGetPointer`, an inline function in `postgres.h` that casts an integer
back to a pointer. Of the files in the memory manager, only `dsa.c` calls it,
and `dsa.c` is not one of the seven.

**The allocator had the bug the compiler exists to find.** It aligned its
region by masking the address and casting it back, which discards the
capability and faults on the first dereference
(`-Wcapstone-pointer-roundtrip`). It aligns by moving the pointer now. That is
the class that made eighteen of MicroPython's twenty-one patches, met on the
first file written for this port.

## Why the host arm is the gate and not a convenience

The manager is built twice from one configured tree, once against glibc and
once for capstone64. The host build exists to be checked against the backend
itself: the recording carries what the real manager asked of `malloc` over the
same workload, and the host arm must ask for exactly that. If it does not, the
recording is not the manager's workload and nothing downstream means anything.

The domain build cannot be checked that way, for the reason the first section
gives: its size classes differ, so its block counts differ by construction.
What it checks instead is that it gave back what it took, and that every object
the recording freed still held what was written into it.
