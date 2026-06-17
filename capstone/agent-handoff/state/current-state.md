# Current Capstone state

Minimal snapshot. Read first in every session.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, and
  `run-nullblk-split-rmmod.sh`
- QEMU runtime smoke tests use snapshot mode, so repeated runs do not mutate `rootfs.ext2`
- Buildroot getty is pinned to `ttyS0`, avoiding intermittent boot-to-login hangs through `/dev/console`
- QEMU runtime smoke tests force `-smp 1`, avoiding intermittent boot stalls under the current OpenSBI/QEMU setup
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` - third BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` - fourth BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` - fifth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-prime.sh` - sixth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` - seventh BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` - eighth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` - ninth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cover.sh` - tenth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-duff.sh` - eleventh BEEBS benchmark runs
  end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` - twelfth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` - thirteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` - fourteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-strstr.sh` - fifteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ndes.sh` - sixteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraybinsearch.sh` - seventeenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-queue.sh` - eighteenth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listinsertsort.sh` - nineteenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listsort.sh` - twentieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-expint.sh` - twenty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-compress.sh` - twenty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-md5.sh` - twenty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh` - twenty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-matmult.sh` - twenty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc32.sh` - twenty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-mergesort.sh` - twenty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-stringsearch1.sh` - twenty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bs.sh` - twenty-ninth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fir.sh` - thirtieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-lcdnum.sh` - thirty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ns.sh` - thirty-second BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ud.sh` - thirty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nsichneu.sh` - thirty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraysort.sh` - thirty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayheapsort.sh` - thirty-sixth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayquicksort.sh` - thirty-seventh
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-dllist.sh` - thirty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-hashtable.sh` - thirty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-aes.sh` - fortieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-picojpeg.sh` - forty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-sha256.sh` - forty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-huffbench.sh` - forty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-rijndael.sh` - forty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc.sh` - forty-fifth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-statemate.sh` - forty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-arcfour.sh` - forty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-des.sh` - forty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-mont64.sh` - forty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-dijkstra.sh` - fiftieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-stack.sh` - fifty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-vector.sh` - fifty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-edn.sh` - fifty-third BEEBS benchmark
  runs end to end and validates its correctness marker

Most BEEBS correctness-marker wrappers now share `beebs_simple_domain.c` and
`beebs_simple_host.c`. Keep separate per-benchmark domain/host files only when
the marker ABI or host behavior is genuinely different; currently the older
`fac`, `fibcall`, and `insertsort` wrappers keep custom markers.

All Capstone-specific benchmark source adaptations now live in explicit `.c`
files under `capstone/benchmarks/beebs/adapted/`. Shell scripts orchestrate
fetch/build/link/run only; no C code is embedded in `.sh` heredocs. Full-
replacement adapted files (bubblesort, prime, cnt, duff, janne_complex, tarai,
levenshtein, recursion) are compiled directly. Tail-append files (strstr,
insertsort, jfdctint, fdct, aha-compress, nettle-md5, nettle-cast128) are
concatenated with the stripped upstream source at build time.

`build-beebs-simple-capstone-common.sh` now supports `BEEBS_EXTRA_DEFINES`
(array of `-D` defines, e.g. `BEEBS_EXTRA_DEFINES=(QUICK_SORT)`),
`BEEBS_EXTRA_SED_EXPRS` (array of sed expressions applied to each source file
after include-stripping, e.g. to add `static` to local array declarations that
would otherwise trigger the stc-copy bug), and includes `-fno-jump-tables`
unconditionally (jump tables use raw integer addresses which fault on Capstone
since loads require capabilities).

## Resolved blocker

The 2026-06-09/10 split `null_blk` unload blocker is resolved. The hang was
diagnosed as lost timer progress after split-domain activity: QEMU traces showed
that the final timer H-interrupt was taken while `mie.MTIP` was disabled, after
which OpenSBI did not reprogram the timer and RCU/percpu-ref progress stopped.

The fix is in `capstone/capstone-qemu`:

- Capstone H-interrupt selection in `riscv_cpu_local_irq_pending()` now considers
  only interrupts enabled by `env->mie`.
- `rmw_mie64()` calls `riscv_cpu_check_interrupts()` after `mie` changes so a
  pending H-interrupt becomes deliverable when software reenables it.

The split null_blk package also keeps the safer fixes found during investigation:
metadata is borrowed per domain call instead of permanently shared, and
`null_validate_conf()` copies back only validated scalar configuration fields.

All temporary Linux/OpenSBI/QEMU trace and printk diagnostics were removed before
the verified run.

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Four remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
