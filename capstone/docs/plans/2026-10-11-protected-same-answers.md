# Protected builds give the same answers: pre-registration, 2026-10-11

Committed and pushed before any build or run below. A result that differs from what is written
here is a finding, recorded as such; this file is not edited afterwards.

## The question

Does protecting a program's nested allocator with the two lifetime instructions change what the
program computes? Each program runs one workload three ways, from one source snapshot:

- **native**: the program built for the host, the oracle;
- **stock**: the program on virtual Capstone (`CAPSTONE_APPLICATION_PROFILE=virtual`, musl
  mallocng), its nested allocator as released;
- **protected**: the same with the nested allocator's lifetime patch, the one the security table
  measures.

Platform for both Capstone arms: the Sublet-lifetime QEMU (af37cc32), launcher 3975314a, module
a1c6cb6b, exact bounds, one hart, as the 2026-10-11 corpus runs.

## The rule

The prediction for every program is the same: **protected output equals stock output equals
native output**, compared exactly on the oracle named below, and both Capstone arms exit as native
does. A protected arm that faults, differs from stock, or differs from native where stock does not
is a finding. Where stock already differs from native, that difference is the port's, recorded
separately, and the claim narrows to protected equals stock.

| Program | Protected patch | Workload | Oracle |
|---|---|---|---|
| mruby 4.0.0-rc2 | 0008 (GC object slots) | full `mrbtest` | per-test results (already measured 2026-10-11; not rerun) |
| FFmpeg 9.0.1 | buffer-pool 0003 (pools), `FFAPP_SUBLET=1` | decode of the run-of-record clip and its flipped control | 30 frame MD5s each (already measured, `ports/ffmpeg/app/results/2026-10-11-virtual-pools`) |
| memcached 1.6.45 | 0006 (slabs, cache.c) | protocol transcripts, one and four workers | normalised bytes against native (already measured, commit bd6a7dcac627) |
| PostgreSQL 17.5 | memory-contexts 0003, `PGSU_NESTED=sublet` | `work.sql` in the single-user backend | all 22 result rows and the final count |
| CPython 3.13.7 | 0014 (obmalloc), `CPY_SUBLET=1` | `python -m test`, sequential in one process, on: test_dict test_list test_set test_tuple test_bytes test_str test_float test_long test_json test_gc test_weakref test_collections test_itertools test_sort test_re | per-test outcome; protected equals stock test for test |
| Perl 5.36.3 | sublet 0001 (SV heads), `PERLD_SUBLET=1` | `t/base` (9 files) and `smoke.pl` | per-file TAP results; smoke output byte for byte |
| SQLite 3.22.0 | sublet-3220000-memsys5 (memsys5 and lookaside) over a memsys5 heap | speedtest1 `main --size 1` | row count and FNV hash of each of the 32 phases |
| Wireshark 4.6.8 | wmem 0001 (block, block_fast) | the port's capture set, `-V -n` | stdout byte for byte against native stock tshark |

SQLite and Wireshark need their protected configuration wired into the application build first;
until today it existed only in their component and corpus builds. That wiring is a separate commit
and does not change these predictions. SQLite's stock arm for this comparison is the same memsys5
heap without the patch, so the two arms differ only in the patch.

## Known in advance

- Perl: the protected image faults on one correct program, `$x = *foo; *x = $x`
  (`bug-corpora/perl/release-differential/results/2026-10-11-virtual/README.md`). It is not part
  of `t/base` or `smoke.pl`; if a selected file reaches it, that is reported, not excluded.
- CPython: tests that need `fork`, `dlopen` or sockets skip or fail on both Capstone arms. Those
  are port differences from native; the prediction for them is protected equals stock.
- N = 1 per arm.
