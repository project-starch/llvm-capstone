# ffmpeg/subobject-repros on cheribsd-subobject, the buggy arms supervised -- 2026-10-10 (R3c)

Pre-registered at `9a314a3473c2` (docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md, R3 and
R3c). Every case built with `-Xclang -cheri-bounds=subobject-safe`; each buggy arm runs under
`cpython/pymalloc-repros/observe/supervise.c`, which reports the fault pc and, since 2026-10-10, the mapped
object holding it and the return address. As predicted, 9 of 9:

- 00, 01, 02, 03: SIGPROT si_code 1 in `ff2_case_run`, a site declared before the run (the case body's
  defective member access).
- 05, 06, 08: SIGPROT si_code 1 AT `write_probe`, the labelled probe, called from `ff2_case_run`.
- 07: SIGPROT si_code 1 in libc's `memcpy`, a declared site, called from `ff2_strlcpy+0x7e`, the case's
  own bounded copy. This is the weaker form of attribution, stated as such in the case's `fault_sites_why`.
- 04: completes (exit 0) -- the 2026-10-09 MISSED, again.

The first supervised attempt (R3a, same day) resolved 8 of these; its case 07 pc lay in libc, which the
attribution tool could not yet read. That is why the supervisor and tool were extended and this run made.

    CHERI_EXTRA_CFLAGS="-Xclang -cheri-bounds=subobject-safe" \
      bash capstone/bug-corpora/ffmpeg/subobject-repros/runners/run-cheribsd.sh <out>

The suite's own exit is 1 because its recorded statuses expect the buggy arms to complete; the
attribution exit is 0 (every fault at a declared site or probe).

Platform (stock CheriBSD, revocation on): qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf.

| arm | exit | |
|---|---|---|
| cheribsd-abi | 0 | as recorded |
| cheribsd-bounds | 162 | as recorded |
| subobj-control | 162 | as recorded |
| so-00-fixed | 0 | as recorded |
| so-00-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-01-fixed | 0 | as recorded |
| so-01-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-02-fixed | 0 | as recorded |
| so-02-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-03-fixed | 0 | as recorded |
| so-03-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-04-fixed | 0 | as recorded |
| so-04-buggy | 0 | as recorded |
| so-05-fixed | 0 | as recorded |
| so-05-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-06-fixed | 0 | as recorded |
| so-06-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-07-fixed | 0 | as recorded |
| so-07-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |
| so-08-fixed | 0 | as recorded |
| so-08-buggy | 162 | faulted (the buggy arm; see attribution.tsv) |

Program sha256: so-00 17066c29d9e216c8, so-01 3dba58950bafae85, so-02 4bb915dc7c3883e6, so-03 3727d59d13cfa71c, so-04 f3275202bd81f69d, so-05 e36e30a3189bcd52, so-06 faf546d2e9c37cc3, so-07 f5907d8464da5d93, so-08 f35353ea2fda6499

`attribution.tsv` beside this file is tools/attribute-cheribsd-faults.py's table (with `--sysroot`) for the buggy arms.
