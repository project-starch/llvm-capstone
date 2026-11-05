# Perl: matched work slices and a completed sub-workload

The unchanged SPEC `makerand.pl` workload now has a short paired instruction
comparison. It uses a fixed seed; `Perl_pp_rand` entries identify corresponding
random draws in both binaries. No source, input, or frozen src.alt changes.
The SDK and compiler/library binaries are the same as the
[previous baseline](../2026-11-05-minimal-mallocng/README.md), including its
compiler-revision and libc optimization-level limitations.

[Evidence](measurements.json) records stop counts, complete-window checks,
output and executable hashes, function/ISA attribution and source audits.
Program instruction attempts include libc/runtime but exclude outside-program
work. These are not hardware cycle or wall-time measurements.

| Scope | Plain attempts | Capstone attempts | Difference |
|---|---:|---:|---:|
| 10000 | 15,716,538 | 23,664,008 | +50.57% |
| 20000 | 31,416,547 | 47,308,616 | +50.59% |
| completed-makerand | 2,984,247,290 | 4,494,385,318 | +50.60% |

The slices start at entry 1 of `Perl_pp_rand` and stop at entry 10,001 or 20,001.
All four runs reach the requested stop and pass window/byte attribution checks.
The Capstone profiles finish in approximately 61 and 65 host seconds including
VM boot and reporting; the deadline is 180 seconds. Those times describe the
workflow, not platform performance. Doubling the sampled work changes the
relative overhead by less than 0.02 percentage points.

## Correctness scope

Before the longer runs were cancelled, **both complete makerand invocations
had already exited successfully**. Their 43,841-byte output files match the
original SPEC expected output byte for byte, SHA-256
`539d02dc10fabf7096fdc8331ec144d1e42f813913ec48e4f611bce3f840950e`.
Their first profiling windows are complete and reconcile to their recorded
footers. Attribution above was recovered from those existing windows, not by
rerunning the full workload or treating cancellation as successful completion.

This validates the makerand sub-workload, **not the complete Perl test harness**.
The latter remains unvalidated. The slice ratios describe this loop-heavy
workload and must not be generalized to every Perl program. Initial parsing
and interpreter startup are outside the slices but included in the completed
makerand measurement. The three ratios agree closely for this workload.

## Remaining overhead

For 10,000 draws, the total difference is 7,947,470 attempts:

| Location | Extra attempts |
|---|---:|
| Interpreter/application functions | 7,943,838 |
| Allocator | 1,271 |
| Other libc/runtime | 2,361 |

Thus more than 99.9% of the difference is in interpreter functions. The largest
positive function deltas are `Perl_pp_nextstate` (+1,620,540), `Perl_pp_padsv`
(+760,381), `Perl_pp_unstack` (+600,000), `Perl_pp_next` (+519,480) and
`Perl_pp_and` (+459,980). This points toward compiler output for frequently
called opcode handlers and their stack/global accesses. It does not justify
another heap-policy change or edits to the benchmark. Attribution is by
containing function; inlining and ABI-induced workload differences remain
relevant when interpreting individual function deltas.

Before/after checks cover all 1,704 original Perl files in the installed
selected tree, with zero mismatches. Baseline SDK hashes are unchanged, and no
long-run or Perl-slice QEMU guests remain active. Private raw evidence is under
`/var/tmp/capstone-spec-php-20261105/perl-slices` and the recorded run paths on
p12; licensed sources and outputs are not committed here.

## Workflow for further instruction comparisons

Use a matched work endpoint, not an hours-long complete run or equal wall time:

```sh
source capstone/tests/capstone-test-env.sh
PROFILE_STOP_AT=Perl_pp_rand:10001,Perl_pp_rand:1 \
  bash capstone/ports/spec-cpu2017/tools/profile.sh \
  arms.json cap test 500.perlbench_r "$OUT"
```

Run the plain arm at the same start/stop counts. The identical frozen binary,
input, seed and logical work unit are required; compare longer slices when
checking representativeness. Full application qualification is a separate
activity and must not be claimed from a slice or started merely to obtain an
instruction comparison.
