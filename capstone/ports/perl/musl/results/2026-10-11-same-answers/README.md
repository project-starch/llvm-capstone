# Perl 5.36.3: protected build, same answers? (2026-10-11)

The Perl row of the pre-registration
[`docs/plans/2026-10-11-protected-same-answers.md`](../../../../../docs/plans/2026-10-11-protected-same-answers.md)
(9bc6f9adf1c9, committed before this run). Prediction: `t/base` (9 files) and `scripts/smoke.pl`
give the same per-file TAP results and byte-identical smoke output on native Perl, on the stock
virtual image and on the protected one (`PERLD_SUBLET=1`). N = 1 per arm.

## Outcome

| file | native | stock (virtual) | protected (`PERLD_SUBLET=1`) | protected = stock | = native |
|---|---|---|---|---|---|
| base/cond.t | 4/4 ok, exit 0 | 4/4 ok, exit 0 | 4/4 ok, exit 0 | yes | yes |
| base/if.t | 2/2 ok, exit 0 | 2/2 ok, exit 0 | 2/2 ok, exit 0 | yes | yes |
| base/lex.t | 120/120 ok, exit 0 | 120/120 ok, exit 0 | 120/120 ok, exit 0 | yes | yes |
| base/num.t | 56/56 ok, exit 0 | 56/56 ok, exit 0 | 56/56 ok, exit 0 | yes | yes |
| base/pat.t | 2/2 ok, exit 0 | 2/2 ok, exit 0 | 2/2 ok, exit 0 | yes | yes |
| base/rs.t | 41/41 ok, exit 0 | 41/41 ok, exit 0 | 41/41 ok, exit 0 | yes | yes |
| base/term.t | 7/7 ok, exit 0 | **1 of 7, fault cause 25 in `fileno_unlocked`, exit 139** | **the same** | yes | no (port) |
| base/translate.t | 257/257 ok, exit 0 | 257/257 ok, exit 0 | 257/257 ok, exit 0 | yes | yes |
| base/while.t | 4/4 ok, exit 0 | 4/4 ok, exit 0 | 4/4 ok, exit 0 | yes | yes |
| smoke.pl | 18 lines (P1-P17, SMOKE_DONE), exit 0 | **P1-P4, then "Out of memory!", exit 1** | **the same** | yes | no (port) |

"= stock" and "= native" compare the whole stdout byte for byte (sha256 in `results.json`), which
for the TAP files also means the same plan line and the same `ok`/`not ok` per test number.

Read by the registered rule:

- **Protected equals stock on all ten workloads**, stdout byte for byte, exit status for exit
  status. No protected-only fault; the known false positive (`$x = *foo; *x = $x`) was not reached.
- **Eight `t/base` files equal native** on both Capstone arms: 486 assertions, byte-identical TAP.
- **Two workloads differ from native on stock already** -- port differences, not effects of the
  patch; for them the claim narrows to protected = stock, which holds:
  - `base/term.t` test 2 runs `` `$^X -le "print 'hi there'"` ``. On both images the guest shell
    first tries to run the image as a script (binfmt is not registered in this gate), then the
    domain faults with cause 25 in `fileno_unlocked` (pc offset 0x72ca9c stock, 0x727a30
    protected; symbolized against each image with load base 0x10000), the file ends after
    `ok 1`. This is the known "backticks fault on the virtual port" of the corpus README.
  - `smoke.pl` at its default `CAPSTONE_PERL_SMOKE_OBJECTS` (unset, so 20,000 objects) prints
    P1-P4 and then Perl's "Out of memory!" on stderr in P5's SV churn, exit 1, on both images.
    The port's earlier virtual qualification (`../virtual/README.md`) ran smoke with 2,000 objects
    "to keep the QEMU qualification bounded"; this run used the script as is, as registered. So
    the registered smoke comparison covers only P1-P4 (identical on all three arms up to the
    point the Capstone arms stop). The unregistered follow-up below, at 2,000 objects, completes
    on both images with output identical to native.

## Follow-up run: smoke.pl at 2,000 objects (not registered)

A separate boot after the registered run, on 2026-10-11 at about 08:42 UTC (the remote host's own
clock read 08:16:24Z; it runs about 26 minutes behind). Why: the registered default of 20,000
objects exhausts the virtual heap on both Capstone arms ("Out of memory!" above); 2,000 is the
setting the port's earlier virtual run used (`../virtual/README.md`). The registered 20,000-object
result above stands as recorded; this run does not replace it.

Same two images (sha256 e4c27891... stock, 20856abd... protected, rehashed in the stage), same
platform, same launcher and module, one boot, fresh `--work`, own QEMU lock.
`CAPSTONE_PERL_SMOKE_OBJECTS=2000` on all three arms, otherwise as above. Script:
[`smoke2k.sh`](smoke2k.sh).

| | native | stock | protected |
|---|---|---|---|
| exit | 0 | 0 | 0 |
| stdout | 360 bytes, 18 lines (P1-P17, SMOKE_DONE), sha256 a4863fc7bf3ca182... | identical | identical |
| stderr | empty | empty | empty |
| wall | 0.045 s | 28.34 s (guest) | 32.02 s (guest) |

Stdout equality holds on both checks: the sha256 computed in the guest (the quoting slip of the
first run fixed) and a byte comparison of the serial text on the host. Native output at 2,000
objects has the same hash as at 20,000: the printed lines do not depend on the count. The boot
took 67 s (run-staged.py, start to end).

## What ran

Source: a `git archive` of 9bc6f9adf1c9 (`capstone/`, plus `compiler-rt/lib/builtins` from the
same commit), snapshotted on the remote host. Both images built by
`ports/perl/musl/build-perl-domain.sh` (sha256 5124d552...) with `PERLD_PROFILE=virtual`, default
`PERLD_OPT` (`-O2`), default `PERLD_VIRTUAL_BLOCKS` (65536), `JOBS=24`; protected adds
`PERLD_SUBLET=1` (patch `sublet/patches/5.36.3/0001`, sha256 328902a9..., applied 11th after
port patches 0001-0010; `-DPERL_CAPSTONE_SUBLET`). Script: [`build.sh`](build.sh). Inputs:
perl-5.36.3.tar.gz f2a1ad88116391a1..., perl-cross c2d8f8b7027e, musl-1.2.5.tar.gz a9a118bb....

| | sha256 |
|---|---|
| stock image | e4c27891c419ebd92a157745a7174b98d97c4e5b52098839ca8fe0104d998e7a |
| protected image | 20856abd9cfbb5c40a249f28c06f753bbfe85b370c2bc73f408a74e3c15045b8 |
| native perl (the stock build root's; used) | c79b1b3b2a1c270a758609dd6ea4e780eafbf0c8154e532d201d12eb20e5450e |
| port patches 0001-0010, concatenated | 28631728363ecf1e2c5d1ae982ee2a0992122503bb39695a6ca197b07ca64823 |
| `t/base/*.t` (sha256sum list) | 97366809...: identical in the native, stock and protected trees |
| smoke.pl | 15bf158879a2f5432b183ed944bd52d928e0e4bbf40a5f93a8d9d0f2c6af1f44 |
| run-staged.py | d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c |

These images are not the corpus's (296cb582 / 317ae110): same patch set, rebuilt here from this
snapshot.

Platform (as registered: Sublet-lifetime QEMU af37cc32, launcher 3975314a, module a1c6cb6b, exact
bounds, one hart):

| | sha256 |
|---|---|
| qemu-system-riscv64 | af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d |
| fw_jump.elf | 44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116 |
| Image | e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b |
| rootfs.ext2 | 62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845 |
| capstone-vexec | 3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4 |
| capstone_vm.ko | a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512 |
| clang | 4c6135feafae20c23edf01a4964d51459d7acb85469cd50166704e1f9a3365d4 |

**Native reference**: the build script's own native build of the same tarball (Perl's
`Configure -des -Uusedl -Uusethreads -Uusemymalloc -Dusenm=false -Dinc_version_list=none`,
gcc 15.2.0, `-O2`, x86_64-linux), run on the host as installed `native/bin/perl` from a copy of
its build tree's `t/` with `../lib` linked to its build tree's `lib/` and `PERL5LIB` set to it.
smoke: `perl smoke.pl DIR` in a directory holding a copy of smoke.pl.

**Capstone arms**: one boot, `runtime/virtual/run-staged.py --timeout 3600 --disk-mib 2048`, own
QEMU lock, fresh `--work`. Staged per arm: the image, the build tree's `t/` and `lib/` (as
`qualify.py` stages them, without its child-perl stub, which `t/base` does not load). In the
guest, per arm: `t/` copied to tmpfs (rs.t writes `./foo`), `lib` linked beside it, each file run
as `PERL5LIB=<lib> capstone-vexec perl.dom base/X.t` from `t/`, stdout and stderr to separate
files; smoke as `capstone-vexec perl.dom smoke.pl <dir>` from a tmpfs directory holding it,
`CAPSTONE_EXEC_DIAGNOSTICS=1`. Script with the generated gate: [`run.sh`](run.sh).

## Wall times (measured)

| | native | stock (guest) | protected (guest) |
|---|---:|---:|---:|
| each t/base file | 0.029-0.043 s | 16.02-19.45 s | 8.95-20.26 s |
| 9 t/base files, sum | 0.30 s | 154.8 s | 138.3 s |
| smoke.pl | 0.069 s (complete) | 27.48 s (to OOM) | 33.77 s (to OOM) |

Guest times are `/proc/uptime` around each `capstone-vexec`, so they include each launch; they
are not a performance comparison (one sample, startup-dominated). Builds: stock 331 s, protected
338 s. The boot (run-staged.py, start to end): 369 s.

## Deviations and limits

- The guest-side sha256 of each stdout in the `SA_END` marker is empty: a quoting slip in
  `run.sh` (`cut -d\" \"` in an unquoted heredoc). Capstone stdout hashes are therefore computed
  on the host from the serial log with CRLF normalised to LF; native output contains no CR, and the
  eight equal files compare equal under that normalisation.
- smoke ran at the script's default 20,000 objects and both Capstone arms ran out of memory (see
  above). A second, unregistered boot at 2,000 objects followed at the coordinator's request; it
  is recorded separately above.
- `base/term.t` reached its backtick on both images; no binfmt registration in this gate, as in
  the port's virtual runs.
- N = 1 per arm. Raw serial log not committed.
