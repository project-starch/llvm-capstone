# The virtual arms: predictions

Registered 2026-10-11, before any corpus case ran on either image. A result that differs is a
finding; this file keeps the prediction that was made.

## The arms

Both run Perl 5.36.3 as ported (`ports/perl/musl/build-perl-domain.sh`, virtual profile) on the
Sublet-lifetime QEMU (af37cc32, exact bounds), one process per case
(`runners/virtual/run-virtual.py`). The system allocator in both is musl 1.2.5's mallocng on the
virtual profile, which bounds each object to its request and retires its lifetime on free.

| arm | image | what differs |
|---|---|---|
| `virtual-malloc` | stock (`PERLD_SUBLET=0`) | Perl's SV head arena as released: a freed head goes on `PL_sv_root` and is handed out again, never reaching `free()` |
| `virtual-nested-pools` | `PERLD_SUBLET=1` | patch `ports/perl/sublet/patches/5.36.3/0001`: each SV head a CDERIVE child of its arena's lifetime, revoked by CREVOKE when freed |

Nothing else differs between the two images: same tree, same SDK, the patch inert without
`-DPERL_CAPSTONE_SUBLET`. Both carry the port's new patch 0009, which works around compiler issue
C-77 (`docs/ref/ISSUES.md`): before it, every program containing a bareword that starts with `::`
faulted at compile time in `memcpy` on both images.

## How a row is read

The runner writes, per case, the outcome (completed, exitN, fault, timeout), the fault's cause
and function, whether that function is one of the case's `fault_sites`, and the harness line.
Two controls must pass first (`6*7` evaluates, the harness loads from the staged library).

The cases' `fault_sites` are host ASan's frames #0-#1 at the pin, from the 2026-10-06 run
(`results/20261006-host-differential/sites.tsv`), declared before any virtual run. Three cases
have none (02, 05, 11: ASan silent).

* **System allocator** (03, 04, 06, 10; and 07, whose stale access ASan places in the regexp
  program, malloc memory): `virtual-malloc` catches the case when it faults with cause 25 (a
  revoked lifetime, the system allocator's free) in one of its `fault_sites`. A fault elsewhere,
  or with another cause, is reported and not counted: the stock port faults with cause 24 on its
  own (below), so a cause-24 fault cannot be told from the port's.
* **Nested allocator** (01, 02, 05, 09: an SV head): the stale reference is into Perl's own head
  arena, where ASan sees nothing, so its SEGV frames (01, 09) lie where a corrupted value is
  followed later, not at the stale access. These are read as a matched pair instead:
  `virtual-malloc` is the baseline and must reach the case and complete, with any exit status
  and no fault; `virtual-nested-pools` then catches it when it faults with cause 25 (a
  revoked lifetime). A case whose baseline faults has no baseline and is reported as such, not
  counted, as in the mruby and SQLite rows.
* 08 is spatial and 11 is an in-bounds read of a live object; both run for completeness and are
  not temporal readings.

The stock port is known to fault on its own, identically on both images
(`runners/virtual/qualify.py`, 2026-10-11): cause 24 in `ck_builtin_func1` (`op/ref.t`), and the
virtual profile starts no subprocesses (`system()` returns -1, a piped open is ENOSYS, backticks
fault with cause 25 in `fileno`), after which several core test files fault in `Perl_sv_clear`.
The corpus harness starts no subprocess (`harness/shim.pl` stubs `runperl` and `fresh_perl*`).
Assigning `$0` faults too (cause 24 in `memchr`); no trigger assigns it. A fault outside a case's
sites can therefore be the port's, which is why sites, the cause and the matched pair decide.

Qualification of the patch, the false-positive check, compares Perl core test files on the two
images file by file with the child-perl helpers stubbed (`qualify.py`); a file that ends
differently on the protected image is a freed-head read the patch does not route through its
arena. It runs before the corpus and its result is committed with the corpus run.

## Predictions

From the physical arms of 2026-10-06 (`results/20261006/matrix.tsv`), where `sysalloc-sublet`
had a revoking system allocator and `sublet-svheads` the old SV-head adapter on top:

| case | layer | virtual-malloc | virtual-nested-pools | reading |
|---|---|---|---|---|
| 01 | nested | fault, cause 24 (as on every physical arm) | fault | no baseline |
| 02 | nested | Perl's own panic, exit 255, no fault | the same: `SvIS_FREED` reads the head through its arena, as upstream does | not caught by Sublet; Perl's own check |
| 03 | system | completes | completes | the trigger does not reproduce here |
| 04 | system | fault, cause 25, in a site (memmove, Perl_sv_catpvn_flags) | the same | caught |
| 05 | nested | completes ([FAIL]: Perl's "Attempt to free unreferenced scalar") | fault, cause 25 | caught by the patch |
| 06 | system | fault, cause 25, in a site (Perl_SvREFCNT_dec_NN, S_free_codeblocks) | the same | caught |
| 07 | system | fault, cause 25, in a site (S_regcppop, S_regcp_restore) | the same | caught |
| 08 | spatial | fault, bounds (cause 28) | the same | not temporal |
| 09 | nested | fault (as on every physical arm) | fault | no baseline |
| 10 | system | fault, cause 25, in a site (memmove, PerlIOScalar_write) | the same | caught |
| 11 | -- | completes, wrong bytes | the same | not reachable by any arm |

So the predicted table cells: system allocator 4 cases (04, 06, 07, 10), all caught on
`virtual-malloc`; nested allocator 1 case with a baseline (05), caught by
`virtual-nested-pools`. 01 and 09 are expected to fault without the patch; if either completes
on `virtual-malloc` instead, it enters the nested denominator and its `virtual-nested-pools` row
decides it.

## Amendment, after the first run: patch 0010 and a rerun

The first run (`first-run/`, runner ced629f24256, images stock e4ca6fb06359 and protected
d8fa2b671553, both carrying patch 0009) ended as follows; every control passed on both arms.

| case | virtual-malloc | virtual-nested-pools |
|---|---|---|
| 01 | fault 24, Perl_pp_iter (site) | fault 25, Perl_pp_iter (site) |
| 02 | exit 255, Perl's "panic: attempt to copy freed scalar" | fault 25, Perl_sv_setsv_flags |
| 03 | completed, [PASS] | completed, [PASS] |
| 04 | fault 24, ck_builtin_func1 (not a site) | the same |
| 05 | completed, [FAIL] (no "Attempt to free unreferenced scalar") | fault 25, Perl_cv_undef_flags |
| 06 | fault 24, ck_builtin_func1 (not a site) | the same |
| 07 | fault 25, S_regcppop (site) | the same |
| 08 | fault 28, Perl_pp_gvsv | the same |
| 09 | fault 24, Perl_newATTRSUB_x (site) | fault 25, Perl_newATTRSUB_x (site) |
| 10 | fault 25, memmove (site) | the same |
| 11 | completed, [FAIL] | the same |

**04 and 06 never reach their defects, on either platform.** Both triggers load `overload.pm`,
which calls `builtin::blessed`. Perl hands each `builtin::` call checker its descriptor as
`newSVuv(PTR2UV(builtin))` and turns it back with `NUM2PTR` (`builtin.c`). Where a pointer is a
capability that round trip loses the tag, so any `builtin::` call or `use overload` faults: cause
24 in `ck_builtin_func1` here, and on CheriBSD SIGPROT at one instruction in
`Perl_sv_2uv_flags`, with revocation on and off, for both cases. `perl -e
'builtin::blessed(...)'` and `perl -e 'use overload ...'` reproduce it on all three images
with no defect involved. Port patch 0010 keeps the descriptor's index instead of its address.
With it, both one-liners complete on all three rebuilt images.

The corpus is rerun on images rebuilt with patch 0010 (stock, protected, and the CheriBSD
interpreter), with the same runners, rules and predictions, all registered above before the
first run. The first run stays recorded.

Two further readings of the first run, so the rerun cannot be read as fitted to them:
* 02 on the protected image faults at `stype = SvTYPE(ssv)`, `sv_setsv_flags`' first read of the
  freed SV the defect hands it, one step before upstream's own `SvIS_FREED` panic. By the rule
  above its baseline ended without a fault (exit 255), so a cause-25 fault on the protected image
  counts. The prediction (Perl's panic on both) was wrong.
* `$x = *foo; *x = $x`, a correct program, completes on the stock image and faults on the
  protected one (cause 25 in `S_glob_assign_glob`): upstream reads the source glob after `LEAVE`
  may have freed it. The old adapter recorded the same read (`ports/perl/sv-heads/README.md`). It
  is a read of a freed head that the patch does not route, by choice: it is a read after free.
  The qualification table (`qualify-v2.txt`, 47 core test files, every one ending the same way
  on both images) does not exercise it.
