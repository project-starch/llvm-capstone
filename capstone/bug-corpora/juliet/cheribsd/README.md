# Juliet on CheriBSD purecap

The same Juliet cases as the virtual-profile harness one directory up (`../select.py`'s two manifests: 366 temporal,
9,138 spatial), built for stock CheriBSD purecap with the CheriBSD SDK's compiler and run under CHERI-QEMU, so the
platform's own protection -- the compiler's stack bounds, `malloc`'s per-object bounds, and `free`'s quarantine with
a later revocation sweep -- can be set beside `../results/2026-10-19-virtual/`. Nothing in the cases or the manifests
changes; what changes is the compiler, the vehicle, the observer and therefore the judge.

The vehicle is the CheriBSD built from pinned sources on 2026-10-01 (cheribuild `9224aad4`, CHERI-LLVM `7e122876`
= clang 17.0.0, cheribsd `88f39900`, CHERI-QEMU `d0bb921c`); every record carries the compiler's version and hash,
its purecap configuration file, the libc, QEMU, firmware and kernel hashes, and the disk image's size and mtime,
read from the tools rather than from this paragraph.

## How a reading is made

- **Build** (`build.py`): every case twice, exactly as `../build.py` does -- a bad image (`-DOMITGOOD`) and a good
  image (`-DOMITBAD`), both `-DINCLUDEMAIN` with `io.c`, `-O0` -- with `<sdk>/bin/clang --config
  cheribsd-riscv64-purecap.cfg -std=gnu11 -fuse-ld=lld`. The configuration file is the SDK's own and carries
  `-cheri-tgot-tls`: this CheriBSD's dynamic linker refuses an image with traditional TLS relocations, and `io.c`'s
  `<ctype.h>` use references libc's thread-local rune locale (found on the 2026-11-05 smoke, every image exit 1 with
  `Traditional TLS not supported`). Also built: `supervise`, the ABI probe, `sicode-selftest`, the uaf control, the
  type-size and malloc-bounds probes, the seed finder and `ok`.
- **Batches** (`run.py`): one boot each, up to 64 cases (128 images) per boot, several boots at once when asked
  (`--guests`), each on its own qcow2 overlay over the shared disk image, which is never opened for writing. The
  batch's images reach the guest over virtio-9p and are copied to its `/tmp` before they run. One shell script runs,
  in this order: `ok`, the ABI probe, the controls (bounds, revoked, uaf), the type-size probe, the malloc-bounds
  probe, the seed finder, the images (a case's good image before its bad image), the controls, the ABI probe and
  `ok` again. Every program runs as `timeout -k 5 10 ./supervise ./<image>`.
- **Observer**: `supervise` (`cpython/pymalloc-repros/observe/supervise.c`) traces the program and prints, from
  outside it, the signal, `si_code`, address and pc the kernel reports for the fault that ended it, and the mapped
  object (the image, or `/lib/libc.so.7`) the pc and the return address lie in. The program under test prints
  nothing that decides anything: a fault leaves its own prints in a stdio buffer, so `Calling bad()...` is not
  required for a catch and `Finished bad()` is read only from a completion.
- **Three configurations**, each a run of the same build with the kernel's revocation knobs set by sysctl and read
  back into the record: **default** (`runtime_revocation_default=1`, `every_free=0`, the platform as shipped),
  **off** (`default=0`), **every-free** (`every_free_default=1`: `free()` sweeps before it returns). Only the default
  run decides a verdict; off is the attribution pair (a fault that persists with revocation off was not revocation's),
  every-free the sensitivity reading that separates a block that was quarantined and not yet swept from one the
  mechanism never saw. The temporal set runs all three; the spatial set runs the default only, since bounds do not
  depend on revocation. This is the three-boot design of `bug-corpora/php/engine-repros`' CheriBSD arm.
- **A batch counts** (`judge.py`) only if the boot completed without a panic, every knob read back as the
  configuration requires, the steps are exactly the sequence derived from the build, no step timed out, `ok`
  completed at both ends, the ABI probe read `pointer_bytes=16` with the configuration's `runtime_revocation` at
  both ends, and the controls read as they must at both ends: `ctl-bounds` (the ABI probe's one-past-exact-bounds
  read) dies on SIGPROT `PROT_CHERI_BOUNDS` in every configuration; `ctl-revoked` (`sicode-selftest revoked`: free,
  forced sweep, read) dies on SIGPROT `PROT_CHERI_TAG` at the default and every-free and completes with revocation
  off; `ctl-uaf` (free, no forced sweep, read) dies on `PROT_CHERI_TAG` at every-free and completes with revocation
  off -- at the default it is recorded, not required, because what the quarantine does with one free is itself a
  reading (`uaf_control` in the summary). The seed finder must find all four patterns, the same in every batch.
- **Verdicts**: a good image is CORRECT when it printed `Finished good()` and exited 0; any fault there is a
  FALSE-POSITIVE? to examine. A bad image is CAUGHT when SIGPROT killed it with the class's `si_code` --
  `PROT_CHERI_TAG` (2) for 415/416, `PROT_CHERI_BOUNDS` (1) for 121-127, or `PROT_CHERI_TAG` on a spatial case whose
  pointer moved far enough out of bounds to lose its tag (reported in the `sicode` column) -- it did not print
  `Finished bad()`, and the case's good image is CORRECT. It is MISSED when it printed `Finished bad()` and exited 0
  without a fault. It is **STOPPED:<what>** when something that is not the class's capability fault ended it:
  `STOPPED:SIGABRT` is this libc's `free()` refusing a second free of a block it already quarantined or revoked
  (`mrs.c`, `abort_on_validation_failure`), an allocator consistency check that is recorded and never scored as a
  capability catch; other signals and other SIGPROT codes likewise. Anything else is NO-READING with a reason.
- **Attribution is by site**, and weaker than the virtual harness's strict check: the `site` column names the
  function holding the fault pc (in the image, or in the library `supervise` named, resolved with `--rootfs`) and
  `ra_site` the function holding the return address (the caller only when the site is a leaf such as libc's
  `memcpy`). CheriBSD's kernel reports no capability bounds with the fault, so the size-based attribution of
  `../judge.py` has no counterpart here, and the summary says so.
- **rand-dependent cases** (flow variant 12) take their verdict from the seeded run, as in `../judge.py`, with the
  seeds the guest's own `rand()` gives: the census run's seed finder reports the first seed for each pattern of the
  first two `rand()%2` results under this libc, `build.py --seeded` builds with those, and every batch of every run
  must read the same four. CAUGHT if any seed is caught; STOPPED if no seed is caught and the seeds that ran the
  defect were all stopped the same way (the others MISSED); MISSED only if all four completed. The off and
  every-free columns of such a case are marked `:unseeded`, since those configurations run the census image only.
  The STOPPED clause was added after the temporal run: its six CWE-415 rand cases read SIGABRT on the one seed
  pattern that takes the double free and MISSED on the three that do not, which the first rule left NO-READING.
- **MISSED cases are relabelled only on evidence from this platform**: `NOT-TRIGGERED:sizeof-ptr` from the type-size
  probe read in the guest (a capability is 16 bytes, so `malloc(sizeof(T*))` holds one `double`, `int64_t` or
  two-int struct); `NOT-TRIGGERED:wide-snprintf` from the case's own twin printing `wcslen=1` after the copy on this
  libc (what `swprintf` writes is deterministic per libc, and FreeBSD's need not do what musl's did). CWE-170 twin
  readings are reported only. Intra-object cases are reported apart: the SDK's default bounds are per allocation,
  not per field.

## Running it

```sh
export PATH=/home/miniconda/miniconda3/envs/cheri-deps/bin:$PATH      # python 3.11 with pexpect
J=$(capstone/bug-corpora/juliet/fetch-juliet.sh <dir>)
capstone/bug-corpora/juliet/cheribsd/run-set.sh temporal <out> "$J" <sdk> <rootfs> <image> [guests]
capstone/bug-corpora/juliet/cheribsd/run-set.sh spatial  <out> "$J" <sdk> <rootfs> <image> [guests]
```

with `<sdk>` = `cheri/output/sdk`, `<rootfs>` = `cheri/output/rootfs-riscv64-purecap`, `<image>` =
`cheri/output/cheribsd-riscv64-purecap.img` of the CheriBSD build above. `run-set.sh` runs, per set: select, build,
the default run (and for temporal the off and every-free runs), the seeded build and run, a first judgement, the
twins of the MISSED cases, and the final judgement. Results land in `<out>/<set>/results/`.

Do not start Python from `capstone/bug-corpora/juliet/`: its `select.py` shadows the standard library's `select`,
which pexpect imports (`shared.py` loads `../common.py` by path for that reason).

## Results

`results/<date>-cheribsd/` holds, per set, `<set>-verdicts.tsv` (one row per case: flags, the verdict, the good
image's verdict, the fault's `si_code`, site and return-address site, the off and every-free readings, the twin
reading) and `<set>-summary.json` (counts per class and configuration, each run's batches counted and why any was
void, the controls' readings, the seeds, the probes' readings, the toolchain and the vehicle), plus `PROVENANCE`
(the vehicle's source pins and the harness commit). No raw logs.

`results/2026-11-05-cheribsd/`, harness commit `ad8d0502b189` on the vehicle above (clang 17.0.0 `7e122876`,
QEMU 7.1.0), every batch of every run counted (temporal: 6 default + 6 off + 6 every-free + 1 seeded; spatial: 143
default + 9 seeded + 3 twin), every good image CORRECT (no false positive in 9,504 good images). Verdicts are at the
platform default; the temporal rows carry the off and every-free readings beside them.

| CWE | cases | CAUGHT | MISSED | STOPPED | not triggered | intra-object | virtual profile (2026-10-19) |
|---|---|---|---|---|---|---|---|
| 415 double free | 228 | 0 | 0 | 228 SIGABRT | | | 228 caught |
| 416 use after free | 138 | 0 | 138 | 0 | | | 138 caught |
| 121 stack overflow | 3,846 | 3,630 | 0 | 0 | 144 wide-snprintf | 72 | 3,630 caught |
| 122 heap overflow | 2,314 | 1,634 | 418 | 0 | 76 wide-snprintf, 114 sizeof-ptr | 72 | 2,052 caught |
| 124 buffer underwrite | 1,098 | 1,098 | 0 | 0 | | | 1,098 caught |
| 126 buffer over-read | 782 | 725 | 57 | 0 | | | 722 caught |
| 127 buffer under-read | 1,098 | 1,098 | 0 | 0 | | | 1,098 caught |

The temporal set under its three configurations (the off and every-free columns are the census image's reading;
the 7 + 6 rand-dependent cases are seeded at the default only, so there they are marked `:unseeded` and counted
apart):

| CWE | cases | default | every-free | off | virtual profile |
|---|---|---|---|---|---|
| 415 double free | 228 | 0 caught, 228 STOPPED:SIGABRT | 222 STOPPED:SIGABRT, 6 unseeded | 228 MISSED | 228 caught |
| 416 use after free | 138 | 0 caught, 138 MISSED | 131 CAUGHT + 3 unseeded caught, 4 unseeded missed | 138 MISSED | 138 caught |

**Temporal.** At the platform default nothing is caught. Every CWE-416 case completes: it frees one or a few
blocks, far below the quarantine's flush threshold, and uses the block before any sweep, so the capability is still
tagged -- the uaf control (free, no forced sweep, read) completed in 12 of 12 batches while the forced-sweep control
faulted `PROT_CHERI_TAG` in 12 of 12. Every CWE-415 case is `STOPPED:SIGABRT`: `mrs_free` refuses the second free
and aborts, which is the allocator's consistency check and not a capability fault. With **every-free** revocation
134 of 138 CWE-416 cases are CAUGHT `PROT_CHERI_TAG` (57 inside libc's `strlen`/`wcslen` under `printf`, 77 in the
case's own code; the 4 others are rand-dependent cases whose unseeded image took the defect-free branch) and
CWE-415 is unchanged. With revocation **off** there is no `PROT_CHERI_TAG` fault anywhere: CWE-416 138 MISSED, and
CWE-415 228 MISSED because the underlying allocator accepts the second free. The virtual profile's 366 of 366 and
CheriBSD's 0 of 366 differ in the mechanism's timing, not its coverage: the virtual profile ends a lifetime at
`free`, CheriBSD's sweep is batched and asynchronous.

**Spatial.** 8,185 CAUGHT, all `PROT_CHERI_BOUNDS` (no spatial pointer lost its tag first), at the copying function
(`memcpy` 1,822, `memmove` 1,602, `wcscpy` 575, `strcpy`/`strncpy`/`wcsncpy` 500 each, the `cat` variants 220 each)
or in the case's own `badSink` (106). The 418 CWE-122 MISSED are exactly the eleven families whose access stays
inside `malloc`'s rounding slack, 38 flows each: `CWE129_large` writes `buffer[10]` of `malloc(10 * sizeof(int))`,
40 bytes into a 48-byte capability, and the ten `CWE193` off-by-one families write the eleventh byte of
`malloc(10)`, a 16-byte capability (`capbounds`: 4 -> 16, 24 -> 32, 100 -> 112). No CAUGHT family has a MISSED
member. The 57 CWE-126 MISSED are CWE-170 cases (unterminated copy, then printed): all 54 `wchar_t` ones and the 3
rand-dependent `char` ones, whose twins read `dest[99] = 0` -- the copy happened to be terminated. The other 51
`char` CWE-170 cases are CAUGHT at `strlen`: on this ABI the stack word after `dest` was not zero, so the print read
past the buffer. The 220 wide-snprintf cases are NOT-TRIGGERED here too: FreeBSD's `swprintf` also writes one
character for a narrow `%s` (twins `wcslen=1`), and the 114 sizeof-ptr cases hold their one element in a 16-byte
capability.

**Intra-object: the pre-registration predicted MISSED and was wrong.** All 144 `type_overrun` cases fault
`PROT_CHERI_BOUNDS`, reported as `INTRA-OBJECT:CAUGHT`; the fault is not on the write inside the struct. The 72
`char` cases fault inside `memcpy` reading the SOURCE: `sizeof(struct)` is 48 bytes with two 16-byte pointer
fields and `SRC_STR` is a 32-byte literal. The 72 `wchar_t` cases complete the copy (96 bytes from a 128-byte
literal), which leaves no terminator inside the struct, and fault at `wcslen` under `printf` reading past the
struct. Both are real out-of-bounds accesses the case makes under this ABI, caught by allocation bounds; neither is
sub-object protection, which the SDK's default bounds do not provide.

What these numbers are not: a measurement on hardware (QEMU, TCG), of C++ cases, of other optimisation levels than
-O0, of a CheriBSD configured with sub-object bounds or with every-free revocation as its default, or of attribution
by the faulting capability's bounds (site-level only, see above).

## Pre-registration, before the first run (2026-11-05)

Written before any set ran, after a two-case smoke on the same vehicle (one CWE-121 and one CWE-416 case, bad and
good, under `supervise` at the platform default: the stack overflow died on SIGPROT `PROT_CHERI_BOUNDS` in libc's
`memcpy` with the return address in `main`; the use-after-free completed and printed the freed buffer). Each
prediction names what it rests on, so a surprise is a surprise.

**Spatial, default configuration** (9,138 cases; the virtual profile caught 8,600, missed 60, 394 not triggered or
intra-object):

- **CAUGHT, `PROT_CHERI_BOUNDS`**, for the large majority: every stack overflow and under-write/read (the compiler
  bounds an address-taken stack object exactly), and every heap overflow whose access leaves `malloc`'s capability.
  Rests on the smoke, on `ffmpeg/plain-heap-repros`' CheriBSD run, and on the mechanism.
- **MISSED, by rounding slack**, for heap cases whose access stays inside the size class `malloc` rounded the request
  up to (`capbounds` records the slack per request size in the summary): a `malloc(10 * sizeof(int))` written at
  index 10 lands 40 bytes in on a 48-byte capability. `ffmpeg/plain-heap-repros` measured exactly this absorption on
  this vehicle (cases 1, 4, 5). No count is predicted: which Juliet variants cross the class boundary is what the
  run measures, and a MISSED heap row with a small overflow is a reading of CHERI's allocation granularity, not a
  defect of the harness.
- **NOT-TRIGGERED:sizeof-ptr** for the 114 `sizeof_double/int64_t/struct` cases: a capability is 16 bytes.
- **wide-snprintf (220 cases, NOT-TRIGGERED on the virtual profile): no firm prediction.** musl's `swprintf` wrote
  one character for a narrow `%s`; FreeBSD's converts the multibyte argument, so the copy may be complete and the
  overflow real. The twin decides; a CAUGHT here is a difference between libcs, not between protections.
- **CWE-170 (60 cases, all MISSED on the virtual profile)**: the unterminated copy is read until a zero; whether one
  lies at `dest[99]` is the stack's content, which this ABI need not share with the virtual profile's. The twin's
  `dest99` reading is reported; no verdict is predicted.
- **Intra-object (144 cases)**: MISSED -- the SDK's default bounds are per allocation. Reported apart, as
  `INTRA-OBJECT:`.
- **Good images: 0 false positives** in 9,138.

**Temporal, default configuration** (366 cases; the virtual profile caught all 366):

- **CWE-416 (138): MISSED for nearly all.** A Juliet case frees one or a few blocks, far below the quarantine's
  flush threshold (`mrs.c`, a quarter of the heap by default), and the use follows the free at once, so no sweep
  has run and the capability is still tagged. The smoke read exactly this on `malloc_free_char_01`. A CAUGHT here
  would be a case whose use happens to follow a sweep; none is expected.
- **CWE-415 (228): STOPPED:SIGABRT for all.** `mrs_free` validates every pointer against its bitmap and aborts when
  the block is already quarantined (`abort_on_validation_failure` is true unless `_RUNTIME_ABORT_DISABLE` is set).
  Rests on reading `mrs.c`, not on a measurement: the smoke ran no CWE-415 case. A completion here (MISSED) would
  mean the validation did not fire; a SIGPROT would mean the second free dereferenced a revoked capability, which is
  not expected at the default.
- **Good images: 0 false positives** in 366.

**Temporal, every-free**: CWE-416 **CAUGHT `PROT_CHERI_TAG`** for nearly all 138 -- `free()` sweeps before it
returns, and every use in these cases goes through the revoked capability, directly or inside libc (`strlen` under
`printLine`); a use that goes through a copy made *after* the free, or through memory rather than the capability,
would be the exception. CWE-415 stays **STOPPED:SIGABRT**: the second free presents an untagged capability, which
the same validation refuses. `ctl-uaf` must die on `PROT_CHERI_TAG` at both ends of every batch.

**Temporal, off**: **no `PROT_CHERI_TAG` fault anywhere** -- one would refute the attribution of the every-free
catches to revocation. CWE-416 MISSED throughout. CWE-415: with quarantining off `mrs_free` hands the pointer
straight to the underlying allocator, and what that allocator does with a second free is not predicted here; the
column is reported. `ctl-revoked` and `ctl-uaf` must complete at both ends of every batch.

**Controls, every run**: `ctl-bounds` dies on `PROT_CHERI_BOUNDS` at both ends of every batch; the ABI probe reads
`pointer_bytes=16` and the configuration's `runtime_revocation`; the seed finder reads four seeds, identical across
batches and runs.
