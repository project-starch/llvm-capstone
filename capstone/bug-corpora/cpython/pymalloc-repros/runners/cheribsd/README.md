# Stock CheriBSD: the `cheribsd-revocation` arm

The twenty cases as ordinary CheriBSD purecap programs with pymalloc stock, under the platform's
own libc revocation as it ships. pymalloc returns a freed block to its own pool free list, not to
libc, so the expected result is that nothing is caught; see the header of
`run-cheribsd-revocation.sh` for why that is the measurement.

    source capstone/tests/capstone-test-env.sh
    export CHERI_SDK=<CheriBSD SDK> CHERI_SYSROOT=<its purecap rootfs>
    bash capstone/bug-corpora/cpython/pymalloc-repros/shared/build-cases.sh cheribsd BUILD
    bash capstone/bug-corpora/cpython/pymalloc-repros/runners/cheribsd/run-cheribsd-revocation.sh BUILD OUT

The runner refuses a guest whose `runtime_revocation_default` is not 1, and records the guest's
`security.cheri` settings before and after. `observe/supervise.c` observes each case's fault from
outside the program. The recorded run is `results/20261008-cheribsd`.
