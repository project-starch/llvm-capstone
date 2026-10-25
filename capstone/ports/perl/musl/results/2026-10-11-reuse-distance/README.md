# Perl 5.36.3: reuse distance at the SV-head allocator, stock against protected (2026-10-11)

The Perl row of the pre-registration
[`docs/plans/2026-10-11-nested-reuse-distance.md`](../../../../../docs/plans/2026-10-11-nested-reuse-distance.md)
(16d2171a3edc, pushed before any instrumented build or run). Boundary: **sv-heads**, Perl's
SV-head hand-out and free. Workload: `scripts/smoke.pl` at 2,000 objects
(`CAPSTONE_PERL_SMOKE_OBJECTS=2000`), as in the memory-behaviour run. Arms: **stock** (Perl's SV
arenas as released) and **protected** (`PERLD_SUBLET=1`, patch `sublet/patches/5.36.3/0001`),
both with the same observer hooks behind `EXP_REUSE_GAP`. N = 3 per arm, one VM per repetition.
Data: [`reuse.json`](reuse.json) (bucket k = observer `bins[k-1]`, empty buckets omitted;
`buckets` and the counters beside it are repetition 1, `repetitions` lists all three).

## Outcome

| | stock (3 reps) | protected (3 reps) |
|---|---|---|
| hand-outs (`attempts` = `issues`) | 80,204 (all 3) | 80,204 (all 3) |
| releases | 76,294 (all 3) | 76,294 (all 3) |
| reuses | 56,345 (all 3) | 56,345 (all 3) |
| distinct start addresses | 23,859 (all 3) | 23,859 (all 3) |
| observer `error` | 0 (all 3) | 0 (all 3) |
| median bucket | 2 (distance 2-3) (all 3) | 2 (all 3) |
| exit, stdout | 0, sha256 a4863fc7bf3ca182... (all 3) | 0, the same (all 3) |

Reuse histogram (every repetition of both arms; bucket k holds distances 2^(k-1) <= d < 2^k):

| bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reuses | 15,648 | 13,014 | 9,377 | 7,772 | 2,600 | 1,452 | 362 | 520 | 241 | 330 | 527 | 1,036 | 7 | 2 | 3 | 3,454 |

**Share of reuses whose bucket differs between the arms** (sum over k of
|stock_k/N_s - prot_k/N_p| / 2): **0** for repetition 1 against repetition 1, and 0 for all nine
stock x protected pairs. The six REUSE-GAP lines are identical, character for character.

## Verdicts

- **Prediction 1 (equal histograms), Perl: holds.** Stock and protected give the same histogram
  bucket for bucket, and the same issues, releases, reuses and distinct starts. The repetitions did
  not vary at this boundary (all three identical within each arm), so the clause "the arms vary
  alike" is met trivially: every per-bucket range is a single value, the same on both arms.
  The memory-behaviour run saw its lifetime counts vary between repetitions; the SV-head sequence
  did not vary here.
- **Prediction 3 (the figure), Perl panel:** the two cumulative curves are the same curve (the
  histograms are equal). The figure itself is drawn by the paper script from `reuse.json`, not here.
- Prediction 2 concerns SQLite only.

## Hook placement

Patches and wrapper: [`experiments/study/reuse-hooks/perl/`](../../../../../experiments/study/reuse-hooks/perl/).
The protected patch 0001 rewrites both hooked macros (it puts `uproot_SV` and `plant_SV` under
`#ifdef PERL_CAPSTONE_SUBLET`, with the upstream macros in `#else`), so there are two hook
variants with identical placement:

| | stock (`sv-heads-reuse-gap-stock.patch`) | protected (`sv-heads-reuse-gap-protected.patch`) |
|---|---|---|
| hand-out | `sv_inline.h`: `uproot_SV`, right after `++PL_sv_count` | `sv_inline.h`: `uproot_SV` (both branches), right after `++PL_sv_count`, i.e. before `Perl_sv_arena_issue` derives the child |
| release | `sv.c`: `plant_SV`, right before `--PL_sv_count` | `sv.c`: `plant_SV` (both branches), right before `--PL_sv_count` (after `S_sv_arena_retire`'s CREVOKE) |
| address | the head pointer the free list holds (`PL_sv_root`'s head / `del_SV`'s argument) | the arena's own pointer to the head (free-list head / `plant_p`), the same address |
| observer | `sv.c`: `Perl_reuse_gap_sv_issue` / `_release`, one static table of 2^18 slots, `atexit` report | the same code |

`uproot_SV` is the one path to a new head (`new_SV` takes a free head with it; `Perl_more_sv`
adds an arena, then takes the first head with it), so every hand-out is one
`reuse_gap_attempt` + `reuse_gap_issue(address, sizeof(SV))`. Every release (`del_SV` expands to
`plant_SV`) is `reuse_gap_release`, including an `SVf_BREAK` head, which upstream keeps off the free
list and which is therefore never handed out again. Perl never frees an SV arena while running
(only `sv_free_arenas` at destruction), so the hook does not count drops for this boundary. The
report is one line on stderr at exit (`REUSE-GAP sv-heads attempts=.. issues=.. releases=..
reuses=.. distinct_starts=.. capacity=262144 error=.. bins=..`), formatted without printf.

The protected image carries the lifetime patch and the stock one does not: `Perl_sv_arena_issue`
is defined in the protected image's symbol table and absent from the stock one's.

## Builds

Source: `git archive 16d2171a3edc capstone` plus the uncommitted `experiments/study/reuse-hooks/
{perl,ffmpeg}` files, plus `compiler-rt/lib/builtins` from 9bc6f9adf1c9, committed as a snapshot
on the remote host before building (snapshot commit f21690b; the hook files there have the
sha256 below). Driver: [`build.sh`](build.sh) (both programs, both arms, in parallel), which runs
`experiments/study/reuse-hooks/perl/build.sh ARM ROOT` with `PERL_MIRROR`, `PERL_CROSS_MIRROR`,
`JOBS=24`. The wrapper does not edit the port; per arm it:

1. runs `ports/perl/musl/build-perl-domain.sh` unchanged (`PERLD_PROFILE=virtual`,
   `PERLD_SUBLET=0|1`, default `-O2`) and keeps that image as `uninstrumented.dom`;
2. `make clean`, copies `reuse-gap-observer.h` and `reuse-gap-hook.h` into the source root, applies
   the arm's hook patch at fuzz 0, and runs the port's build again from its perl stage with
   `PERLD_OPT="-O2 -DEXP_REUSE_GAP"` (perl-cross's `-Doptimize`: the target compile only; the
   wrapper checks the define is in `CFLAGS` and not in `HOSTCFLAGS`);
3. checks: every archive member of `libperl.a` and `lib/auto/**.a` that references `PL_sv_root`
   (44 of them; each `new_SV` expansion does) calls or defines `Perl_reuse_gap_sv_issue` after
   step 2 (44 of 44) and none does after step 1 (0 of 44); the image defines both hook functions.

| | stock | protected |
|---|---|---|
| image (`perl.dom`, instrumented) | 2e533bf2f477cf6a4220a05856560f526b0f5cbe4e9a3abb0d9925052c8ff164 | b978c3528881656c99011281014a62a48a8f6d17fb7da23dc89db62f9ebe5ae4 |
| step 1 image (uninstrumented) | bf0ed9cba9807ebe941b41b8f0d205d906339898c4d0eb985986c83d239dc13b | 5b97009a7dd97021d342ac8dc3a1b197b7b30d8c740b279e8102addfc0794aa5 |
| hook patch | e78aea4fddce024c... | eefe894c4dad55af... |
| build wall (step 1 + step 2) | 559 s (357 + 202) | 558 s (356 + 202) |

The step-1 images are not byte-identical to the memory-behaviour images (e4c27891... /
20856abd...): same size, and the only difference in their string content (`strings -n 6`, sorted
in the C locale) is Perl's embedded `Compiled at <date> <time>` line. Inputs: hook header
f95a0717be512aba..., observer 627b0fe3ac30d75b..., sublet 0001 328902a9641a0587..., port patches
0001-0010 concatenated 28631728363ecf1e....

## Runs

Script: [`run.sh`](run.sh) (8 VMs in parallel: these 6 and the FFmpeg row's 2), parser:
[`parse.py`](parse.py) (`python3 -I parse.py M TAG OUT`; it exits 1 without writing if any
observer `error` is nonzero, a REUSE-GAP line is missing, the bins do not sum to `reuses`, the
output check fails, or the guest's image hash differs from the host's). Each VM boots through
`runtime/virtual/run-staged.py --timeout 3600` with its own `CAPSTONE_QEMU_LOCK` and a fresh
`--work`. In the guest: `insmod`, then from a tmpfs directory holding smoke.pl
(15bf158879a2f543..., the memory run's)

    CAPSTONE_PERL_SMOKE_OBJECTS=2000 CAPSTONE_EXEC_DIAGNOSTICS=1 \
      /mnt/vm/capstone-vexec /mnt/vm/prog.dom smoke.pl /tmp/run > out 2> err

then `rmmod` (exit 0 in all runs). stdout is hashed in the guest and again on the host from the
serial text; both equal a4863fc7bf3ca182... (the same-answers smoke hash at 2,000 objects) in
all six runs. stderr holds only the REUSE-GAP line.

Platform (as registered: Sublet-lifetime QEMU af37cc32, launcher 3975314a, module a1c6cb6b,
exact bounds, one hart, 2 GiB):

| | sha256 |
|---|---|
| qemu-system-riscv64 | af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d |
| fw_jump.elf | 44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116 |
| Image | e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b |
| rootfs.ext2 | 62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845 |
| capstone-vexec | 3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4 |
| capstone_vm.ko | a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512 |
| run-staged.py | d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c |
| clang | 4c6135feafae20c23edf01a4964d51459d7acb85469cd50166704e1f9a3365d4 |

**Wall times**: `capstone-vexec` in the guest (`/proc/uptime` around it): stock 37.99-38.08 s,
protected 43.92-44.25 s. Boots (run-staged.py start to end): stock 49-50 s, protected 56 s; the
batch of 8 VMs 56 s. Not a performance comparison (8 VMs plus other sessions' QEMUs on one host).

## Cross-reading with the memory-behaviour run (not registered)

The memory-behaviour run could not check "about one lifetime per hand-out" because Perl exposes
no hand-out count. This run counts 80,204 SV-head hand-outs on the same workload; the protected
arm there allocated 80,490 to 80,568 more lifetimes than stock. The remainder (286 to 364) is not
attributed here. Different images (the hooks), so this is a reading across runs, not a check.

## Deviations

- Two build attempts before the builds of record failed in the wrapper's own checks, not in the
  build: the first counted `globals.o` (which defines `PL_sv_root`) as a user of it; the second
  ran `llvm-nm | grep -q` under `pipefail`, which fails when grep exits early. Both checks were
  fixed (snapshot commits 2283dbb, f21690b) and all four images rebuilt from the fixed wrappers.
  No run used an image from a failed attempt.
- `CAPSTONE_EXEC_DIAGNOSTICS=1` in the guest environment, as in the memory-behaviour run.
- N = 3 per arm as registered. Raw serial logs not committed.
