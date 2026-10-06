# Virtual Perl qualification

`2026-10-07.json` records Perl 5.36.3 rebuilt with `PERLD_PROFILE=virtual`.
The image uses virtual C startup, Linux virtual mappings, the virtual allocator
and delegated ABI v2 services. Its hashes identify the QEMU, Linux image,
launcher, module, Perl image and smoke source used by the run.

The combined gate passes the existing virtual application, SQLite and mruby
regressions plus Perl startup, file I/O and the Perl smoke workload. The virtual
smoke uses 2,000 live-object iterations through `CAPSTONE_PERL_SMOKE_OBJECTS`
to keep the QEMU qualification bounded; the script's physical default remains
20,000. The profile is one hart with private anonymous mappings and no target
fork, POSIX threads or file-backed `mmap`.

Rebuild and run:

```sh
CAPSTONE_LLVM_BUILD_DIR=/path/to/llvm-build \
RUNTIME_REPO=/path/to/llvm-capstone-runtime-r0 \
PERLD_PROFILE=virtual PERLD_OPT=-O1 \
  bash capstone/ports/perl/musl/build-perl-domain.sh
```

Use the virtual application runner with `--perl` and
`--perl-smoke capstone/ports/perl/musl/scripts/smoke.pl` to regenerate the
record.
