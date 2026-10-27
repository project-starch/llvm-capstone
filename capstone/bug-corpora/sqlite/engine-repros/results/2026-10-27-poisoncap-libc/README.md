# PoisonCap on the corpus with the C library's revocation on (2026-10-27, stopped)

Question: can PoisonCap's two layers, the C library's revocation and its memsys5, run together on the
corpus, in the driver configuration of `../2026-10-26-poisoncap-heap256/` (256 MiB heap from malloc,
minimum 16, lookaside on)? The speedtest1 reuse runs did (`ports/sqlite/poisoncap/results/2026-10-25-speedtest1/`, E3c).

Method: `scripts/libc-run.sh` on p13. The case binaries are driver-la's of 2026-10-26, bit for bit
(`logs/driver-libc-la-compare.out`); only the controls `libc_revoke` and `uaf_libc` are new.
`run-cases.py --libc-revocation run` sets `security.cheri.runtime_revocation_default` to 1 for each
measured process only.

## Controls

| control | run (`smoke-driver-libc-la`) | off (`smoke-off-driver-libc-la`) |
|---|---|---|
| `libc_revoke`: the process's `malloc_revoke_enabled()` | 1 | 0 |
| `uaf_libc`: use-after-free of one C library block | poison fault | returns, reads the old value |
| `uaf_read`: use-after-free of one memsys5 block | poison fault | poison fault |

The switch works and reaches the measured processes. The controls use the 1 MiB static heap.

## Cases (`driver-libc-la`, stopped after five)

Cases 00 to 04 all fault in the C library's `memcpy` (`libc.so.7` offset 0x1b0a5e), on the poisoned
32-byte C library block at 0x40c0e000, right after memsys5's initialisation and before the case reaches
its bug. With the C library's revocation off (2026-10-26), 00, 02 and 03 fault inside the case
program and 01 and 04 return. The run was stopped by hand at case 04 (no `run.json`); the repetition never started.

Not established: why the block is poisoned. A candidate is the defect recorded in
`bug-corpora/cpython/pymalloc-repros/platform/README.md` (their C library never removes poison from a
block it hands out again); their own speedtest1 run R1 stopped at start-up on the same address.

Consequence: with their published C library, the corpus cannot be run with both layers on. The
comparison stays with the C library's revocation off, where every case's object is a memsys5 object.
