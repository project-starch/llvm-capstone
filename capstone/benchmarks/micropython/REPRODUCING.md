# How to re-run every measurement

One index over both corpora and the model pairs, because the commands were
previously spread across evidence files, case READMEs and commit messages, and
three measurements had none written down anywhere. If a row is listed here it can
be re-run from this file alone.

Set up first:

    source capstone/tests/capstone-test-env.sh          # bash, not zsh
    export CAPSTONE_REPO_ROOT=/path/to/llvm-capstone    # zsh needs this explicitly

Paths below are relative to the repository root.

## The three model pairs

Each is a matched pair differing in ONE constant, and the pairing is the point: a
quiet arm alone cannot be told apart from a domain that never started. Run the arm
expected to RETURN first.

    L=capstone/tests/runtime-qemu/silicon-ladder

| what | command | expected |
|---|---|---|
| temporal, in region | `bash $L/run-ladder-qemu.sh nestalloc` | PASSED, retval 170 — untrapped |
| temporal, past region | `bash $L/run-ladder-qemu.sh nestoob` | fault, cause 7, bounds `0x400` |
| spatial, into next block | `DOMAIN_EXTRA_CFLAGS=-DNEST_SPATIAL_OFFSET=64 RUNG_NAME=nestspat_in bash $L/run-ladder-qemu.sh nestspat` | PASSED, retval 170 — untrapped |
| spatial, past region | `DOMAIN_EXTRA_CFLAGS=-DNEST_SPATIAL_OFFSET=1088 RUNG_NAME=nestspat_oob bash $L/run-ladder-qemu.sh nestspat` | fault, bounds `0x400` |
| global, in bounds | `DOMAIN_EXTRA_CFLAGS=-DNEST_GLOBAL_OFFSET=0 RUNG_NAME=nestglob_in bash $L/run-ladder-qemu.sh nestglob` | PASSED, retval 34 |
| global, into next global | `DOMAIN_EXTRA_CFLAGS=-DNEST_GLOBAL_OFFSET=64 RUNG_NAME=nestglob_oob bash $L/run-ladder-qemu.sh nestglob` | **fault, bounds `0x40` = 64 B** |

`RUNG_NAME` is not decoration. Without it every arm overwrites the same `.dom` and
the marker vouches for whichever build ran last.

The three bounds widths are the whole finding: `0x60000` / `0x400` for an
allocator-carved object, `0x40` for a linker-assigned one.

## MicroPython in the domain: the common recipe

    B=capstone/benchmarks/micropython/build-micropython-silicon.sh
    SH=$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share
    ROM="-DMICROPY_CONFIG_ROM_LEVEL=MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES"

Build the guest runner once. It is **not** `capstone-test.user`; that one takes
`<dom> <times> [<second-elf>]` and will read the suite runner's arguments as a
second ELF, fail to create the domain, and produce monitor-side faults that look
like results:

    GCC=capstone/caplifive-buildroot/build/host/bin/riscv64-buildroot-linux-gnu-gcc
    U=capstone/caplifive-buildroot/package/modcapstone/userspace
    $GCC -O2 -I capstone/caplifive-buildroot/package/modcapstone/include -I $U \
      -o $SH/mpy-resume-guest capstone/benchmarks/micropython/tools/mpy-resume-guest.c $U/lib/libcapstone.c

Then, for any script-driven row: put the scripts in
`$CAPSTONE_TMP_ROOT/micropython/tests/<dir>/`, always with a `00_sanity.py`, and

    MPY_TESTS=all MPY_TEST_BASE_DIR=<dir> MPY_TEST_INCLUDE_UNSUPPORTED=1 \
    MPY_FLOAT_CORE=1 DOMAIN_EXTRA_DEFS="$ROM" DOM_NAME=<name> bash $B
    cp $CAPSTONE_TMP_ROOT/micropython-silicon/<name>.dom $SH/
    python3 capstone/benchmarks/micropython/tools/run-resumable-suite.py \
      --domain $SH/<name>.dom \
      --expected $CAPSTONE_TMP_ROOT/micropython-silicon/obj/mpy_tests.expected \
      --guest-runner $SH/mpy-resume-guest --out-dir <out> --capture-output

`MPY_TEST_INCLUDE_UNSUPPORTED=1` is load-bearing for any row whose script is
supposed to crash: the table generator derives each expectation by running the
script on the HOST and silently drops any that exits non-zero. Without it an image
builds with the interesting tests MISSING, behind a line reading `1 tests kept, 3
skipped`.

Always check `Created domain ID = 0` in the round log before believing anything.
`18446744073709551615` is -1 and means no domain existed.

## Per-row commands

| row | scripts | extra |
|---|---|---|
| `MPY-T09` `MPY-T10` `MPY-T11` `MPY-T12` `MPY-T13` `MPY-T25` | `temporal-corpus/repros/domain/` | none |
| `MPY-S01` | `spatial-corpus/cases/MPY-S01_*/` | none |
| `MPY-S31` | `spatial-corpus/cases/MPY-S31_*/` | none |
| `MPY-S05` | `spatial-corpus/cases/MPY-S05_*/` | apply `revert-the-fix.patch` to the MicroPython tree first, **and reverse it afterwards** — that tree is shared by every other build |

Rows measured through the C glue rather than a script use `DOMAIN_EXTRA_DEFS` and
`run-domain-smoke.py`, one flag per row, all in `port/mpy_domain.c` behind their own
`#ifdef` so the production glue is byte-identical with them off:

| row | flag | expected retval |
|---|---|---|
| `MPY-T07` | `-DMPY_T07_LEXER_UAF` | `0x70000001` |
| `MPY-T16` | `-DMPY_T16_DEINIT_AFTER_SWEEP` | `0x16005a01` |
| `MPY-T29` | `-DMPY_T29_HIDDEN_ROOT` | `0x29007701` |
| spatial on the real allocator | `-DMPY_SPATIAL_OVERFLOW=1` | `0x5A004001` |
| the same, past the heap (control) | `-DMPY_SPATIAL_OVERFLOW=2` | fault, bounds `0x60000` |

    DOMAIN_EXTRA_DEFS="-D<FLAG> $ROM" MPY_FLOAT_CORE=1 DOM_NAME=<name> bash $B
    cp $CAPSTONE_TMP_ROOT/micropython-silicon/<name>.dom $SH/
    python3 capstone/tests/runtime-qemu/run-domain-smoke.py $SH/<name>.dom

Build the two arms of a pair separately and **check their md5 differs** before
running. A `-D` that fails to reach the compiler gives two identical images and a
sweep that measures one thing twice.

## The filesystem stack

    MPY_VFS=1 MPY_TESTS=all MPY_TEST_BASE_DIR=capstone-vfs \
    MPY_TEST_INCLUDE_UNSUPPORTED=1 MPY_FLOAT_CORE=1 \
    DOMAIN_EXTRA_DEFS="$ROM" DOM_NAME=mpy_vfs_suite bash $B

with `temporal-corpus/evidence/vfs-smoke-domain.py` as the script. Expected:
`VFS hello 32768` — a FAT filesystem formatted, mounted, written and read back over
a RAM block device written in Python. With `MPY_VFS` unset the image is
byte-identical to a build without any of that work.

## Host-side parent builds

For rows fixed at the pin, where the domain cannot show the defect because the
defect is gone:

    git -C $CAPSTONE_TMP_ROOT/micropython worktree add /tmp/capstone/mpy-<name> <commit>
    make -C /tmp/capstone/mpy-<name>/mpy-cross -j16 CC=gcc-12
    make -C /tmp/capstone/mpy-<name>/ports/unix -j16 CC=gcc-12 MICROPY_MPYCROSS_DEPENDENCY=

`CC=gcc-12` and the empty `MICROPY_MPYCROSS_DEPENDENCY` are both required; the
reasons are in `temporal-corpus/evidence/parent-build-attempt-2026-08-17.txt`.
Do NOT add `-fsanitize=address`: every combination tried ends in `FATAL: uncaught
NLR` on any input at all, including `import sys`.

## What has no reproduction path, and will not get one

`MPY-T01` `MPY-T04` need POSIX file descriptors, `MPY-T06` needs berkeley-db
(not even checked out in the pinned tree), `MPY-T19` needs NimBLE. The reasoning is
in `temporal-corpus/REMAINING.md` and was re-checked against the sources rather
than asserted.
