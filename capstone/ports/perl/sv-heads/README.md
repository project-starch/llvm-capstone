# Perl SV heads under Sublet

Perl 5.36.3 keeps every value's head, the 48-byte `SV` on a capability
target, in 4080-byte arenas. A freed head goes on `PL_sv_root`, a list
threaded through the freed head itself, and the next `new_SV` hands it out
again. A stale `SV*` therefore reaches the next value, and ASan cannot see
it: the memory never returns to `malloc`. This directory moves that
allocator's lifetimes into an adapter, so that a released head can be
revoked (Capstone Sublet), and records each issue in the common reuse observer
(`experiments/study/reuse-gap-observer.h`). The SV bodies, hash entries and
OP slabs keep their upstream allocators; they are not measured here.

## What changes in Perl

[`patches/5.36.3/0001-sv-heads-through-a-lifetime-adapter.patch`](patches/5.36.3/0001-sv-heads-through-a-lifetime-adapter.patch)
applies after the port's seven pointer-layout patches (`../musl/patches`)
and is inert without `-DPERL_SV_HEAD_ADAPTER`. Its header lists every hunk:

- `Perl_more_sv` takes each head from `perl_svh_new`; `PL_sv_root` is never
  set, so every `new_SV` arrives there, including the inline copy that the
  `-DDEBUGGING` extension `re` compiles. `plant_SV` returns the head with
  `perl_svh_del`. An `SVf_BREAK` head (global destruction) is released for good,
  as upstream keeps it off the free list.
- A freed head's flags are `SVTYPEMASK`, and the core reads them through
  pointers that may be stale by design: `SvIS_FREED` (sv.h: "there are some parts
  of the core that have pointers to already-freed SV heads"), and type probes on
  a COP's stash, `PL_curstash`, `PL_last_in_gv` restored by `SAVESPTR`, or
  `GvIO`'s own test. `SvIS_FREED` and `SvTYPE` ask the adapter first and read the
  head only while it is live; a released head reads as `SVTYPEMASK`, as upstream.
- `S_visit`, which global destruction uses, walks the adapter's live heads in
  upstream's order (newest arena first, ascending within one).

## The adapter

[`sv-heads.h`](sv-heads.h) is the platform-independent core. It keeps upstream's
policy exactly: a LIFO free list and a new page of `PERL_ARENA_SIZE /
sizeof(SV) - 1` heads (84) only when the list is empty. A spatial build therefore
reissues the same slot identities in the same order as upstream Perl. The free
list, each head's state and its current alias live in a sidecar, never in a head.

| Backend | Mode 0 (control) | Mode 1 (protected) |
|---|---|---|
| [`capstone.c`](capstone.c), `PERL_SUBLET_MODE` | each head its own slot and bounds, alias kept across lifetimes | release revokes the slot (`sublet_give`) and takes a fresh alias before reissue |
| [`native.c`](native.c), `PERL_SVH_NATIVE_MODE` (test only) | reissued at once | ASan-poisoned and delayed by 4,096 releases |

Capstone heads come from the program's region 1: `experiments/applications/
build.py --nested perl` grants 32 MiB and maps the SDK's grant to that index.
Each backend prints one `PERL_REUSE_GAP` histogram and one `PERL_SV_HEADS` ledger at exit.
Every snprintf in the report formats at most two values: on Capstone a variadic
argument spilled past the argument registers arrives misplaced with compilers
that lack C-48 (`ISSUES.md`).

Until 2026-10-10 a CheriBSD PoisonCap backend (`cheribsd.c`, `PERL_POISONCAP_MODE`)
sat beside these; it was removed, and the four-arm campaign below that used it
stays as recorded. Patch 0001's comment still names it, because recorded build
manifests pin the patch's hash.

## Build

```sh
# Capstone: the study variant, then the measured image with its grant.
CAPSTONE_LLVM_BUILD_DIR=<llvm build with C-46 and C-48> PERLD_SV_HEADS=1 PERLD_OPT=-O1 \
  PERLD_ROOT=$CAPSTONE_TMP_ROOT/perl-svh-capstone bash ../musl/build-perl-domain.sh
python3 capstone/experiments/applications/build.py --app perl --nested perl \
  --root $CAPSTONE_TMP_ROOT/perl-svh-capstone --libc-root $CAPSTONE_TMP_ROOT/perl-svh-capstone \
  --toolchain <same llvm build> --input-revision HEAD --out <new directory>
```

## Native gates

`bash test-native.sh` builds the same Perl natively with ASan and `native.c`,
then checks the core's unit test, the study workload's oracle in both modes, and
a positive control; `--suite` adds upstream's complete test suite in both modes.

## Results (2026-09-28)

- [Native suite](results/2026-09-28/native-suite.txt): of 2,630 files, mode 0
  (immediate reuse) fails only `lib/perlbug.t` test 21, exactly as the unpatched
  ASan build does. Mode 1 (poisoned, delayed) additionally stops in three files,
  each at one read of a freed head by upstream Perl that no adapter path covers:
  `S_glob_assign_glob` reads its source glob after `LEAVE` may have freed it
  (`op/gv.t`; `$x = *foo; *x = $x` reproduces it, and 5.38.2 has the same code),
  `pp_ftrowned` reads the flags of a freed SV on the argument stack
  (MakeMaker's `INSTALL_BASE.t`), and B's `make_sv_object` reads an SV it
  addresses by number (`lib/B/Deparse.t`). Under Sublet such a
  read faults instead of reading stale flags.
- [Capstone smoke](results/2026-09-28/capstone-smoke.txt): `../musl/scripts/smoke.pl`
  prints the native oracle byte for byte in both modes, with 167,854 heads live
  at once and 256,290 revocations under Sublet. It needs more than the campaign
  VM's 262,144 nodes (high-water 592,192 under Sublet), so it ran in a separate
  boot with 1,048,576; its observer table overflows and is not a measurement.
  The [`$x = *foo; *x = $x` control](results/2026-09-28/capstone-glob-control.txt)
  prints `done` in mode 0 and ends in SIGSEGV from the capability fault in mode 1.
- [Four-arm campaign](../../../experiments/study/results/perl-reuse-four-arm-20260928/README.md):
  `records.pl 512 3 0` on both platforms, 12/12 processes.
