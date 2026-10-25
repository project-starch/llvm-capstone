# Perl 5.36.3: port patch 0011 (glob assign reads its source before LEAVE), validation (2026-10-11)

Patch `ports/perl/musl/patches/5.36.3/0011-glob-assign-reads-source-before-leave.patch` (sha256
297200543564b3f980dd312a41dfbe1d7c4a266caff30fe74b9759c6499927b5, uncommitted when this ran) moves
`S_glob_assign_glob`'s two reads of its source glob, `SvTAINTED(ssv)` and `GvAV(ssv)`, to before
the `LEAVE` that may free it. Before it, `$x = *foo; *x = $x`, a correct program, faulted on the
protected image (cause 25) and ran on the stock one: the false positive recorded in
`bug-corpora/perl/release-differential/results/2026-10-11-virtual/README.md`.

Everything below ran on one build host from one source snapshot: `git archive 16d2171a3edc
capstone` plus the uncommitted 0011, plus `compiler-rt/lib/builtins` of 9bc6f9adf1c9, committed
as a local snapshot before building. Data: [`results.json`](results.json). N = 1 per arm.

## Result

| check | outcome |
|---|---|
| native gate: 85 upstream test files, pristine vs pristine + 0011 | **same result in every file**; the only TAP difference is the process ID printed in `op/magic.t` tests 102-103 |
| control `$x = *foo; *x = $x; print "done\n"`, new images | **`done`, exit 0, on stock and protected** |
| the same control, recorded images without 0011 (positive control) | stock `done`; protected **fault cause 25 in `S_glob_assign_glob`**, exit 139 |
| t/base (9 files) + smoke.pl at 2,000 objects | protected = stock in all 10, byte for byte; = native in 9 (`base/term.t` is the known port fault on both); **identical to the recorded same-answers results** |
| op/gv.t under the port (`qualify.py`) | runs on both arms to test 237 and ends identically (fault 24 in `Perl_mro_package_moved`, the port's own, before the file's `*x = $x` block); RT #125840 block run standalone: correct on both new images |
| security corpus, 11 cases x 2 arms | **every verdict as recorded**; table cells unchanged: system allocator 3 / 3 / 3, nested allocator 2 / 0 / 2 |

## Builds

Script: [`build.sh`](build.sh) (all four in parallel, `JOBS=24`).

| build | how | sha256 | wall |
|---|---|---|---|
| stock image | `build-perl-domain.sh`, `PERLD_PROFILE=virtual`, `PERLD_SUBLET=0`, port patches 0001-0011 | a5efdd17817fa36024b5c0cfd672f224b1dcf983d294f8cfab6c613971095db1 | 361 s |
| protected image | the same with `PERLD_SUBLET=1` (+ sublet 0001) | 09e87eae3297b61dc4ebd5113766ff67b17d735dd3ad88519000d8c904cf9abd | 362 s |
| native, without 0011 | pristine perl-5.36.3.tar.gz, the port's native Configure line (`-des -Uusedl -Uusethreads -Uusemymalloc -Dusenm=false -Dinc_version_list=none`), gcc, `-O2` | 6f4afa79f9ea05aec653c8dacc9363bdbf7ba627608123291ffb9c247dd8b0f7 | 115 s |
| native, with 0011 | the same tarball plus 0011 alone (applies at fuzz 0) | 567bae4e6440f866414311c724359b0e6afa7507595751e29d07c20a88dc244e | 115 s |

Both Capstone build logs show `applied 0011-glob-assign-reads-source-before-leave.patch`. The
port's build script builds its own native reference from the pristine tarball (no port patch);
the two native builds above use its Configure line, so they differ in 0011 only.

## 1. Native gate

Script: [`native-gate.sh`](native-gate.sh), comparison [`native-compare.py`](native-compare.py).
In each native build tree, `make test-prep`, then each file as `./perl harness -v FILE` from `t/`
(upstream's harness, which applies the `#!` switches, e.g. `-T` for taint.t). Compared per file:
the plan, every `ok` / `not ok` line with its text and directive, and the harness's `Result:`.

| files | without 0011 | with 0011 | identical |
|---|---|---|---|
| op/gv.t | 304/304, PASS | 304/304, PASS | yes |
| op/taint.t | 1058/1058, PASS | 1058/1058, PASS | yes |
| op/magic.t | 208 planned, 1 `not ok` (194, a TODO), PASS | the same | yes, except the PID in the text of tests 102-103 (the comparison flags this file; the PIDs are the only difference) |
| mro/*.t, 73 files | every file PASS | every file PASS | yes, all 73 |
| base/*.t, 9 files | every file PASS | every file PASS | yes, all 9 |

Per-file plans and counts: `results.json` → `native_gate.harness`. The same-answers reference
outputs (each base file run as `perl base/X.t`, smoke.pl at 2,000 objects, the control) are
byte-identical between the two native builds; smoke stdout sha256 a4863fc7bf3ca182..., control
stdout `done` (sha256 d117fa006ba92085...). The whole native gate took 31 s.

## 2-3. The images: control, same answers, op/gv.t

**Control and same-answers**, one boot ([`run.sh`](run.sh), `runtime/virtual/run-staged.py
--timeout 3600 --disk-mib 2048`, own QEMU lock, fresh `--work`), as
`../2026-10-11-same-answers/` (t/ copied to tmpfs, lib linked, `PERL5LIB`, each file as
`capstone-vexec perl.dom base/X.t` from t/; smoke from a tmpfs directory with
`CAPSTONE_PERL_SMOKE_OBJECTS=2000`; `CAPSTONE_EXEC_DIAGNOSTICS=1`). Image hashes rechecked in the
guest. stdout hashed in the guest and again on the host from the serial text (equal in all 22).

| workload | native + 0011 | stock | protected | protected = stock | = native | = recorded same-answers |
|---|---|---|---|---|---|---|
| control | `done`, exit 0 | `done`, exit 0 | `done`, exit 0 | yes | yes | (new) |
| base/cond.t | 49a8eb79c930 | exit 0, same | exit 0, same | yes | yes | yes |
| base/if.t | f3e52b93ba9c | exit 0, same | exit 0, same | yes | yes | yes |
| base/lex.t | 6c379691d968 | exit 0, same | exit 0, same | yes | yes | yes |
| base/num.t | 0979daee0acf | exit 0, same | exit 0, same | yes | yes | yes |
| base/pat.t | 4b0ceccd8cce | exit 0, same | exit 0, same | yes | yes | yes |
| base/rs.t | 54116ee14af8 | exit 0, same | exit 0, same | yes | yes | yes |
| base/term.t | 954eb67d78c6 | `ok 1`, then fault cause 25, exit 139 (f432ad69c1aa) | the same | yes | no (port: backticks) | yes |
| base/translate.t | c1268208f88f | exit 0, same | exit 0, same | yes | yes | yes |
| base/while.t | a57799323a51 | exit 0, same | exit 0, same | yes | yes | yes |
| smoke.pl, 2,000 objects | a4863fc7bf3c | exit 0, same | exit 0, same | yes | yes | yes (follow-up run) |

(stdout sha256 prefixes; "same" = the native hash.) `base/term.t` faults at its backtick on both
images exactly as recorded (the virtual profile starts no subprocesses).

**Positive control** ([`control-old.sh`](control-old.sh), one boot): the same one-liner on the
recorded images built without 0011 (same-answers stock e4c27891..., protected 20856abd...): stock
prints `done`, exit 0; protected ends with `capstone-exec: domain fault cause=25` (QEMU: "Cap mem
access on revoked capability"), exit 139, pc offset 0x170be4 in the code region, which with the
image's load base 0x10000 is `S_glob_assign_glob`+0x704 in that image's symbols. So the
control can fail, and fails without 0011.

**op/gv.t under the port.** [`corpus.sh`](corpus.sh) runs the corpus's `qualify.py` (unchanged at
16d2171a3edc: child-perl helpers stubbed, as the corpus harness does) on both images after the
corpus, with op/gv.t, op/taint.t, op/magic.t and the 73 mro/*.t. All 76 files end the same way on
both images (outcome, cause, function, ok and not-ok counts; `results.json` → `qualify`), and the
TAP of op/gv.t, op/taint.t and op/magic.t is identical between the arms:

| file | stock | protected |
|---|---|---|
| op/gv.t | fault 24 `Perl_mro_package_moved`, ok 236, not ok 1 (94, "DESTROY was called") | the same |
| op/taint.t | exit 255: qualify.py runs `perl FILE` without `-T` ("-T is on the #! line"); not run under the port | the same |
| op/magic.t | fault 25 `fileno`, ok 60 (as recorded in `qualify-v2.txt`) | the same |
| mro/*.t | 25 files complete, 48 fault with cause 24 in the mro functions (the port's own) | the same, file for file |

op/gv.t reaches test 237 (the recorded pre-0010 run reached 23) and stops at the port's own cause-24
fault, before line 1139, the RT #125840 block that exercises `*x = $x`. That block (lines
1139-1155) and the perl #121242 glob_assign/gp_free block (lines 960-976) were therefore run
standalone ([`gv-blocks.pl`](gv-blocks.pl), `is`/`pass` replaced by prints; [`gv-blocks.sh`](gv-blocks.sh),
one boot):

| image | output |
|---|---|
| native without / with 0011 | 7 correct lines + `GV_BLOCKS_DONE` (sha256 1553b5dc102960b4..., both) |
| new stock, new protected (0011) | identical to native, exit 0 |
| recorded stock without 0011 | identical to native, exit 0 |
| recorded protected without 0011 | nothing printed; fault cause 25 at the same pc offset 0x170be4 (`S_glob_assign_glob`), exit 139 |

## 4. Security corpus

`bug-corpora/perl/release-differential/runners/virtual/run-virtual.py`, unchanged (runner
ca6e88f87401, judge c5e5872372b3, the hashes recorded in the 2026-10-11 rerun's `inputs.json`),
`--timeout` default 600 s, on two persistent VMs of the registered platform brought up for this
run ([`vm-up.sh`](vm-up.sh): QEMU af37cc32, launcher 3975314a, module a1c6cb6b, exact bounds):
stock as `virtual-malloc`, protected as `virtual-nested-pools`. Both controls passed on both arms
(`eval`, `shim`).

| case | virtual-malloc recorded / now | virtual-nested-pools recorded / now |
|---|---|---|
| 01 | fault 24, Perl_pp_iter (site) / **same** | fault 25, Perl_pp_iter (site) / **same** |
| 02 | exit 255, Perl's "attempt to copy freed scalar" / **same** (only the two addresses in the panic text differ) | fault 25, Perl_sv_setsv_flags / **same** |
| 03 | completed, [PASS] / **same** | completed, [PASS] / **same** |
| 04 | fault 25, memmove (site) / **same** | the same / **same** |
| 05 | completed, [FAIL] 1 of 2 / **same** | fault 25, Perl_cv_undef_flags / **same** |
| 06 | completed, [FAIL] 5 of 29 / **same** | the same / **same** |
| 07 | fault 25, S_regcppop (site) / **same** | the same / **same** |
| 08 | fault 28, Perl_pp_gvsv / **same** | the same / **same** |
| 09 | fault 24, Perl_newATTRSUB_x (site) / **same** | fault 25, Perl_newATTRSUB_x (site) / **same** |
| 10 | fault 25, memmove (site) / **same** | the same / **same** |
| 11 | completed, [FAIL] 3 of 3 / **same** | the same / **same** |

Outcome, cause, function and site flag equal the recorded `matrix.tsv` in all 22 rows; the harness
line too, except case 02's panic text, whose two heap addresses differ. Read by the registered
rules, the cells stay: system allocator 04, 07, 10 caught by the Sublet arm (3 of 3); nested
allocator 02 and 05 caught by `virtual-nested-pools` (2 of 2). The CheriBSD column (3 and 0) was
not rerun here.

## Wall times

| step | wall |
|---|---|
| Capstone builds (parallel) | 361 s, 362 s |
| native builds (parallel) | 115 s each |
| native gate, both trees | 31 s |
| control + same-answers boot (both images) | 476 s; in the guest per file: stock 14.2-23.1 s, protected 12.4-23.0 s, smoke 31.5 s / 36.4 s |
| corpus | virtual-malloc 320 s, virtual-nested-pools 335 s (in parallel) |
| qualify.py, 76 files | 2,135 s, 2,144 s (in parallel, after the corpus) |
| positive-control boot / gv blocks boot | 55 s / 85 s |

Not a performance comparison (other sessions' VMs ran on the same host).

## Deviations and limits

- `op/taint.t` did not run under the port: `qualify.py` invokes `perl FILE` without the `-T` its
  `#!` line requires. Natively it ran under the harness with `-T` (identical, PASS).
- The native gate compares pristine against pristine + 0011, built with the port's native Configure
  line, not the port's Capstone patch set (0001-0010 are for the domain build).
- op/gv.t under the port does not reach its `*x = $x` tests on either image (port fault 24 in
  `Perl_mro_package_moved` after test 237); those blocks were run standalone instead, as above.
- The CheriBSD arm of the corpus was not rerun.
- N = 1 per arm. Raw serial logs and VM state not committed.
