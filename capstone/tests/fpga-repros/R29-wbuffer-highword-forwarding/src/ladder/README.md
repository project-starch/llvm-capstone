# The silicon distance ladder (boot sw49, 2026-09-10)

Three variants of `../s06agg_kernel.h` differing in ONE thing: the number of filler instructions
between the plain `sd` of the granule's high word and the capability-grained `ldc` of that granule.
Built with `tests/runtime-qemu/silicon-ladder/build-ladder-domain.sh` at distinct `DOMAIN_BASE_VA`
(0xA0000 / 0xB0000 / 0xE0000; preflight C15 refuses a collision) against the unmodified rung at
0x10000.

    arm         gap   entry VA   image sha256[:16]   sw49 reading
    s06agg        0   0x10000    249118220f8cf37a    66   <- the defect
    s06agg_d1     1   0xA0000    48a6f8b5b0947a35    64   <- the turn point
    s06agg_d2     2   0xB0000    4010922474b15363    64
    s06agg_d4     4   0xE0000    f0fc280bdb706a9a    64

**The gap is verified in each image's DISASSEMBLY, never in its source.** The filler is
`__asm__ volatile("nop"… ::: "memory")`, and the memory clobber is what stops the compiler moving the
store past it — but a clobber is a compiler barrier, not a guarantee, so the check that matters is
counting the instructions between `sd a0,0x18(a2)` and `ldc a3,0x10(a2)` in the built `.dom`. An arm
whose filler had been optimised away would return 64 and read as "the defect stopped at this
distance", which is the same failure that produced two retractions in this investigation already.
`../../run.sh verify` gates the shipped rung on exactly that adjacency property.

Rebuilding these is a fresh entry-stall draw per image (see `../../../R16-entry-stall/`), so prefer
the recorded hashes above; if you must rebuild, `sha256sum` the set and abort if any two match.
