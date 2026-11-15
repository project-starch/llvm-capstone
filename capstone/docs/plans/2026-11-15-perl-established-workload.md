# Reuse distance at Perl's SV heads on 500.perlbench_r's suns.pl: pre-registration, 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards. It follows `2026-10-11-nested-reuse-distance.md` (definition,
observer, arms, report) and `2026-11-15-postgres-established-workload.md` (platform).

## The question

October measured Perl 5.36.3 on `smoke.pl`, a script that we wrote. This run asks the same question on
an input of SPEC CPU2017's 500.perlbench_r: does the lifetime patch change when Perl's SV-head
allocator reuses an address? It also measures the noise floor between two runs of one arm.

## Why only suns.pl (addendum to the stream's brief)

The run uses only the train script that runs on stock Perl. SPEC's harness and the modules it ships
in `data/all/input/lib` require perlbench's own Perl 5.22.1: `lib/Config.pm` stops unless the version
is 5.22.1 and the OS name is `spec-cpu`. The scripts that call `rand` (refrate's checkspam, diffmail
and splitmail, and test's makerand) cannot reproduce SPEC's expected output on stock Perl, because
perlbench replaces `rand` with SPEC's own generator; on the host's Perl 5.40, diffmail and splitmail do
not even compile with Perl's core library in place of SPEC's. Three train scripts use no `rand` and
match SPEC's expected output on stock Perl: perfect, scrabbl and suns. A probe on the run-3 stock image
under QEMU ran suns.pl in 447 s with SPEC's expected output, while scrabbl.pl had not finished after
36 minutes (2.5 s natively; perfect.pl takes 2.8 s natively). So suns.pl is the one input that fits the
60-minute budget.

## Definition and boundary (October's)

Reuse distance and buckets as in October. The boundary is "sv-heads": an issue when `new_SV` takes a
head from the arena free list (`uproot_SV`), a release when `del_SV` returns it (`plant_SV`), at the
head's address. Hooks: `reuse-hooks/perl/sv-heads-reuse-gap-ARM.patch` and `reuse-gap-hook.h`.

**Change to the observer table**: 2^19 slots instead of 2^18 (`EXP_REUSE_GAP_SV_LOG2=19` to
`reuse-hooks/perl/build.sh`, default unchanged at 18). The probe's suns.pl run filled 213,016 of the
262,144 slots (81 %); a full table sets error 1 and rejects the run.

## Platform and images

The trap route of run-3 (kit `~/asr-work/eval/kit-run-3`, as in the PostgreSQL plan). Images:
`reuse-hooks/perl/build.sh stock|protected` from the tree export of `trap/eval` ed8cc375c2 with this
branch's hook files, perl-5.36.3.tar.gz sha256 f2a1ad88116391a176262dd42dfc52ef22afb40f4c0e9810f15d561e6f1c726a
(https://www.cpan.org/src/5.0/perl-5.36.3.tar.gz), perl-cross c2d8f8b702. The protected arm adds the
port's lifetime patch `ports/perl/sublet/patches/5.36.3/0001`.

## Workload

- SPEC CPU2017 1.0.2 (p13 `~/spec2017/SPEC17`), `benchspec/CPU/500.perlbench_r/data/train/input/suns.pl`
  (sha256 7545b37d...) with its dictionary `WORDS` (f32a0a4b...), and SPEC's modules
  `data/all/input/lib` on the include path, as SPEC stages them.
- Invocation as SPEC's `Spec/object.pm` runs a script without an `.in` file: `perl -I. -I./lib suns.pl`,
  standard output to `suns.out`, in a copy of the input set under `/tmp/run` (suns.pl writes its
  `validate` file into the working directory).
- `PERL_HASH_SEED=0` in both arms: perlbench fixes its hash seed at build time (`spec_config.h`
  defines `PERL_HASH_SEED`), and the stock Perl takes a fixed seed from the environment.
- **Output oracle**: `suns.out` and `validate` byte-identical to SPEC's expected
  `data/train/output/suns.out` (01688ea0...) and `validate` (96611d1c...), compared in the guest.
- Runner: `reuse-hooks/perl/run-perlbench-suns-reuse.py`, one boot per run.

## Arms and repetitions

**stock** and **protected**, two repetitions each (stock-1, stock-2, protected-1, protected-2), results
under `~/asr-work/eval/run-5/reuse/perl-ARM-REP/`, collected by `reuse-hooks/cpython/collect-reuse.py`.
A run is rejected if it did not complete, exited nonzero, logged a fault, reports a nonzero observer
error, or an output file differs from SPEC's.

## Predictions

1. **Output**: both files equal SPEC's expected files in every run.
2. **Noise floor near zero**: with a fixed hash seed nothing in suns.pl depends on time or chance, so
   within-arm reusediff is at most 0.001 in both arms (identical bins expected).
3. **Stock against protected equal**: the Perl patch hands out the head before the free list moves and
   keeps the free-list order and the head layout; its arenas are larger malloc objects (a table of
   head references follows the heads), which moves arenas but not heads within them (run-3 on
   smoke.pl: identical bins). Predicted: equal
   issue, release and reuse counts, and stock-against-protected reusediff at most 0.001.
