# The embedded suites on P1 (bitstream d83d3a633, kit bkit91): T1 and T3

The 103 programs (BEEBS, RV8, CoreMark) rebuilt with the two SDKs of P1's SPEC arms, for silicon kits on the native
route. The run is the bitstream-3 run's (`../results/2026-10-27-t3all-bitstream3/`, `../results/2026-10-28-t3heap-bitstream3/`)
with T2 dropped. What changes, and why:

- **SDKs.** T1 = the FRCDKMSspa3 SDK, T3 = the FRCDKMSa3 SDK (perf/cclever/arms/*/sdk; the SPEC images' compiler, llvm-c9
  29027126f39b). `check-sdks.py` over their roots (perf/alean/sdksp13b vs sdkm13b) reads INTEGRITY OK: only the allocator
  objects and the sdk.json heap switches differ.
- **Launch.** bkit91's kernel has the native adapter built in, with no /dev/capstone-vm, so capstone-vexec cannot run. Each
  process is a direct exec of its .dom through binfmt, as the SPEC native stages run.
- **Counters.** A native exec is confined to its cwd grant (resource-dom.h), so the perf module's sysfs counters and
  /proc/interrupts are out of reach. The harness's `-DBENCH_CONTEXT_COUNTERS` (`build.py --context-counters`) reads
  rdcycle/rdinstret in the context. These counts include the kernel's work at exits inside the region; the bitstream-3
  CSRUNV-window counts did not. The run script brackets each process with the module's `native_counters` (run_cycles,
  runs), the per-process window-only reading. Without the define, the harness objects are byte-identical to before
  (default, count and both plain modes compared).
- **Layout.** `layout_script` accepts gp's back margin (`. += 2048;`) at the end of .bss, which the fold-global-lo SDKs'
  link.ld carries. The layout gate (`check-arms.py build/T1 build/T3`) passes 148 of 148 pairs.

Files:
- `build-p1.sh` runs heap-layout.py over both SDKs, then build.py for each arm.
- `mk-p1kits.py` makes the seven kits from perf/main3b/t3k1..3 and perf/main5/t3a1..4. It keeps each source kit's process
  order and W/I without the T2 steps, and builds each board stage with bkit91's own make-train-stage.sh.
- `p1-check.py` is the transcript check (exit 0 PASS, 1 FAIL, 2 NO DATA). It was negative-tested on a twin transcript
  with, in turn: a cycles=0 line, a check= mismatch across arms, a check= mismatch with the bitstream-3 reference,
  rdcycle_rdinstret=static, a missing NATIVE-CNT end, rc=1 and an empty input.
- `qpass.sh` is the QEMU functional pass of the seven native stages (faultesc/run.sh, slots 29-31).

Readings, 2026-11-17:
- QEMU functional pass (qpass.sh, natio images, icount shift 0). All seven kits PASS p1-check.py with the bitstream-3
  references (runs-t3k1..3, runs-t3a1..4):
  - every step rc 0;
  - 4,840 BENCH lines, each with cycles, instret and ns above 0;
  - check= is one value per program and iteration across T1 and T3, and equals bitstream 3's for 103 of 103 programs.
- Each board stage's train/ is byte-identical to the native stage that passed (diff -r; the kit adds only plan.json).
- QEMU instret (exact instruction counts, the kernel's at exits included):
  - T3/T1 geomean 0.9998, range 0.9962-1.0000.
  - Against bitstream 3's silicon instret it is not like-for-like: matmult-int reads 25.88 M on QEMU, close to
    240 x 20^3 x 13 = 24.96 M from its 13-instruction inner loop, while bitstream 3's silicon window read 11.97 M with the same
    iteration count and the same inner loop. Which instrument differs is not established, so instret is compared
    silicon to silicon only.
