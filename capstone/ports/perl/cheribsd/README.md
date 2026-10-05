# Perl 5.36.3 on CheriBSD purecap

`build.sh` builds the complete interpreter with the existing seven
pointer-layout patches shared with the Capstone port. It pins Perl and
perl-cross, verifies the release archive, and keeps all fetched sources and
artifacts under `$CAPSTONE_TMP_ROOT`. Configure/build helpers run on the host;
the resulting interpreter targets RISC-V purecap with 16-byte pointers and
alignment, `-O1`, no dynamic loading, no interpreter threads, and libc malloc.
Time::HiRes is omitted because its build step executes target probes.

```sh
source capstone/tests/capstone-test-env.sh
CHERI_SDK=/path/to/sdk CHERI_SYSROOT=/path/to/purecap-rootfs \
  PERL_CHERI_ROOT=/tmp/capstone/perl-cheribsd \
  bash capstone/ports/perl/cheribsd/build.sh
```

The build root must be fresh. `PERL_ARCHIVE` and `PERL_CROSS_MIRROR` allow
cached inputs. Outputs are `perl`, `manifest.json`, the patched source tree
and build logs. Perl's pure-Perl library remains under `src/perl-5.36.3/lib`;
stage it and set `PERL5LIB` for workloads needing modules. The core smoke test
uses builtins and needs no separately installed library.

The first cross-build failed because perl-cross left `d_nanosleep` unset,
generating the invalid directive `# HAS_NANOSLEEP`. The recipe supplies the
CheriBSD answer explicitly, together with the OS name and capability alignment.
No additional interpreter source patch was needed for the checked smoke path.
The CheriBSD ports catalog alone did not establish that this exact version
would work on the artifact's RISC-V purecap ABI; this recipe was built and run.

The [qualification evidence](results/2026-09-28/) records the existing
`../musl/scripts/smoke.pl`: all 17 sections plus `SMOKE_DONE` match native
5.36.3 byte-for-byte under both `_RUNTIME_REVOCATION_DISABLE=1` and
`_RUNTIME_REVOCATION_ENABLE=1`. Arrays, hashes, 20,000-object churn, recursion,
regexes, closures, methods, files and packing complete. These are two smoke
processes, not the complete Perl test suite or a memory benchmark.

This qualification binary keeps Perl's upstream nested allocators.
`PERL_CHERI_SV_HEADS=1` builds the study variant instead: patch 0001 and the
PoisonCap backend of [`../sv-heads`](../sv-heads/README.md) replace the SV-head
arenas, and the common phase observer is linked in. One binary carries the
spatial control and PoisonCap (`PERL_POISONCAP_MODE=0/1`, which the CheriBSD
runner sets from the arm). Its [four-arm campaign](../../../experiments/study/results/perl-reuse-four-arm-20260928/README.md)
admits Perl's SV heads to the cross-application reuse figure; SV bodies, hash
entries and OP slabs remain upstream's.
