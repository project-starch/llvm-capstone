# Eight cases on stock CheriBSD, revocation on, through the port, 2026-09-22

**All eight complete under libc revocation, and the control beside them
faults.** Stock CheriBSD purecap -- the platform's own `malloc` under every
APR node, its own kernel-defaulted revocation on and read back in the guest
(`CHERI_ABI pointer_bytes=16 runtime_revocation=1`) -- does not see a stale
bucket or a stale pool holder. The same run's control frees a block, sweeps,
and reads through the old pointer at the same labelled load,
`apr_defect_read`, and faults exactly there (signal 34, code 2). The
mechanism is active; it is never asked, because neither recycling level
between a bucket and `malloc` ever calls `free()` on the paths these cases
take.

    matrix.tsv    one line per case, the control's outcome beside it
    inputs.json   binaries, platform fingerprint, the ABI-control line

This is the day's second run through the port (the first, before case 4's
handback became the lender's epoch, is in the archive) and the second
CheriBSD record of this corpus. The first, 2026-09-21,
ran the cases against the census's freestanding build with `free()`
interposed and counted (`freed_to_malloc=0` on every arm); this one runs
them through the port's CheriBSD build, `bucket-pointers.c` over
`node-malloc.c`, the same allocators under the platform's `malloc`, so that
the four columns of the corpus come from one source tree. The verdicts are
the same. The platform is the PoisonCap work tree's image with the local
libc fix the CPython corpus documents (`libc 6726fdb0…`); the guest default
is preserved.
