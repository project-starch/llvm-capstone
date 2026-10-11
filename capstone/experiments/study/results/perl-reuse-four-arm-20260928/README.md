# Perl 5.36.3: four-arm SV-head reuse

All 12 processes pass the complete interpreter's `records.pl 512 3 0` oracle
(`EXP-OK perl 2357760`, which the unmodified native 5.36.3 also prints) and
the nine phase checks: three per arm, from one fresh Capstone boot and one
CheriBSD boot. These are application executions, not replay. The workload is
the study's record-churn script run by the full interpreter; it is not a
standard benchmark score.

| Arm | Processes | Issues/process | Observed reuse share | Median observed gap |
|---|---:|---:|---:|---:|
| Capstone spatial | 3/3 | 49,185 | 63.74% | 2–3 |
| Capstone + Sublet | 3/3 | 49,185 | 63.74% | 2–3 |
| CheriBSD spatial adapter | 3/3 | 49,192 | 63.74% | 2–3 |
| CheriBSD PoisonCap adapter | 3/3 | 49,192 | 57.27% | 4,096–8,191 |

The measured boundary is Perl's SV-head allocator: every value's 48-byte head,
which upstream carves from 4080-byte arenas and chains through the freed head
itself. SV bodies, hash entries and OP slabs keep their upstream allocators and
are not measured. The [adapter and its Perl patch](../../../../ports/perl/sv-heads/README.md)
keep upstream's policy (LIFO free list, a new 84-head page only when it is
empty), so each platform's spatial control reissues the same slots in the same
order as upstream Perl. Every repetition within an arm reports identical counts
and bins.

Compare protection against its own platform's control:

- **Capstone Sublet** revokes each released head before its slot can be issued
  again: 47,764 revocations per process, one per release, with 17,892 slots
  carved (213 pages). Reuse order is unchanged, so the histogram equals its
  spatial control bin for bin. The VM's node high-water mark reached 83,618 of
  262,144 in the Sublet processes and 35,854 in the spatial ones.
- **CheriBSD PoisonCap** poisons each released head and publishes it only after
  a revocation sweep, with the published SQLite policy transferred: 4,096
  entries, or at least 16 MiB held with a quarter quarantined. Each process
  swept 12 times: 11 full-queue drains and one teardown; the percentage trigger
  never fired, and nothing was left quarantined. Quarantined heads hold their
  slots, so the adapter carved 251 pages instead of 213 and issued 21,019
  distinct heads instead of 17,839.

The CheriBSD arms share one binary, kernel and corrected libc, with process
revocation **enabled** and the guest-wide default disabled, as in the CPython
and PostgreSQL campaigns. The Capstone arms share one image on the platform of
the CPython campaign (same QEMU, kernel, firmware, rootfs, launcher and
262,144 nodes). Both platforms stage the same `records.pl`, `strict.pm` and
`warnings.pm`.

The figure measures the conditional distribution of **observed same-start
reuses**, indexed by successful new lifetimes, including interpreter startup
and exit. It does not measure failed allocation attempts, physical working set,
elapsed time or total memory.

## Evidence and checks

Run `python3 validate.py`. It re-derives `reuse-summary.json` from the two raw
archives and checks, per process: the runner's own verdict, the oracle, the
phases, the platform and build identities, the staged input hashes, and each
closing `PERL_SV_HEADS` ledger against its histogram (issues, releases and live
heads; one revocation per release under Sublet and none in the control; poison,
clear and zero bytes equal, and sweeps equal to their causes, under PoisonCap).
A tampered histogram, summary or sweep count makes it fail.

- `capstone-raw.tar.gz`, `cheribsd-raw.tar.gz`: runner, points, manifest,
  `runs.jsonl` and every process's command, stdout and stderr. The CheriBSD
  archive leaves out the guest's private key and its serial console log, and
  the `uname` line in its manifest has the kernel builder's `user@host` replaced
  by `<builder>@<host>`; nothing else is changed.
- `capstone-build-manifest.json`: the relinked image (`build.py --nested perl`)
  with its compiler, runtime and every input object's hash.
- `cheribsd-build-manifest.json`: the binary, compiler, configuration and every
  source input's hash.

Both builds are `-O1`. The Capstone compiler is the one the CPython campaign
used (binary `2790a0c1…`); it has the C-46 and C-48 fixes. The compiler of the
earlier Perl port builds (`f7b50f08…`) lacks C-48, which misplaces the second
and later variadic arguments spilled to the stack, and it also fails an
assertion compiling `pp_ctl.c` at `-O1`.

## Qualification

The adapter's native gates (`ports/perl/sv-heads/test-native.sh`) build the same
Perl for the host with ASan, poisoning each released head while delaying its
reuse. Upstream's complete test suite, 2,630 files, then differs from an
unpatched ASan build only where upstream Perl itself reads a freed head: `op/gv.t`,
`lib/B/Deparse.t` and ExtUtils-MakeMaker's `INSTALL_BASE.t`. With released heads
reused at once, as upstream does, it shows no difference at all. The first of
these, `$x = *foo; *x = $x`, reads the freed source glob after `LEAVE` in
`S_glob_assign_glob`, in 5.36.3 and 5.38.2: on Capstone the measured image's
spatial arm prints `done`, and its Sublet arm stops it with SIGSEGV from the
capability fault. The interpreter's 17-section smoke script also prints the
native oracle in both Capstone modes, in a separate boot with a larger node
pool. The [adapter's results](../../../../ports/perl/sv-heads/README.md#results-2026-09-28)
record all three.
