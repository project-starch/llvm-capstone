# FPGA monitor checkpoints C3 and C5 on silicon -- pre-registered 2026-10-03 20:05:04, committed and pushed BEFORE the boots

Bitstream caplifive_supcall_36a641e0b.bit (identified by csnodefree 0xFFCD). Both boots use the same payload Image (sha256
33b63f59113e) and the same stages, and differ only in the OpenSBI build (firmware/SHA256SUMS):
- monitor capstone-sbi monitor/supcall-fpga 1fd1bbe: the board line 472990f, merged with the context-slots line 4674ab6,
  plus the silicon CSR-event form;
- wrapper caplifive-opensbi wrapper/supcall-fpga 849c8e1: carves the 1 KiB save area under the define.
Stages, in this order: k800 (stock b2d60e52, the control); then P1 cell 5 -O0 speedtest (e6ee5255c896aa21, its
recorded oracle) with `--speedtest1 --testset main --size 1 --verify` through sqlite_host_rr.user 2c9e82d1; then k800.

C3 = fw-c3 b317668bc7f8: the MERGED monitor with supervision compiled out (no defines beyond TARGET_FPGA).
  Predicted: k800 retval=4 twice; speedtest Verification Hash 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0,
  SPEEDTEST1-CYCLES within 1 % of the record 2,551,483,818; one boot banner; no stall.
  This shows the merge itself is sound on silicon.

C5 = fw-c5 49864994aa64: + CAPSTONE_SUPERVISED_CALL + CAPSTONE_SUPERVISOR_CSR_EVENTS + CAPSTONE_SUPERVISE_CLASSIC_TEST.
  Every classic call runs under supervision, with quantum 2,000,000 cycles, and the monitor resumes each preemption.
  Predicted:
  - k800 retval=4 twice, each with SUPN 0 and SUPK 0. k800 is ~4,500 cycles, well under one quantum.
  - speedtest Verification Hash 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0, exactly as C3. The workload's
    result is unchanged by ~1,275 preempt/resume cycles.
  - SUPN about 1,275 (2,551M / 2M). This is a timing quantity, reported. Its invariants are SUPN > 0 and SUPK 0 at
    the end.
  - Every SUPA line 0 (armed).
  - SPEEDTEST1-CYCLES is REPORTED, not predicted. The domain's mcycle bracket includes the monitor's resume loop and
    its two UART trace lines per resume.
  REFUTED if:
  - the hash or HEAP differs;
  - the call ends in SUPK 2 (a fault event);
  - SUPN is 0 on the speedtest (supervision never preempted);
  - any SUPA is non-zero;
  - the boot wedges (classified by ENT1/ENT2 and the SUP tags, not as an R-16 stall).

## ADDENDUM 2026-10-03 20:08:30, before either boot: the image was re-baked, and the control relinked
The first attempt never booted. The runner's preflight refused it before any upload, on two grounds:
- C15: speedtest1.dom and the stock k800 both enter at 0x10000, which is R-3;
- the image carried unused files.
The image was re-baked with the k800 RELINKED at 0x20000 (589ceee3853c6092; its silicon record is 4 at both ends
of both P1 boots on R-42) in place of the stock one, and with speedtest1_baseline and sqlite_host.user retired.
The new payload Image is 9f66fb53af98. Firmware rebuilt from the same sources on it (firmware/SHA256SUMS):
- C3 96884501d098;
- C5 eec16b773423.
Preflight now reads GO.
Predictions are unchanged, except that the control is the relinked k800 (retval 4).

## C5 RESULT, and the diagnostic boot C5d, pre-registered 2026-10-03 20:31:34 before it
- C3 (96884501d098) PASSED as predicted:
  - k800 4 and 4 (4,524 / 4,531 cycles);
  - 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0;
  - 2,551,615,035 cycles (+0.005 % against the record).
  The merged monitor with supervision compiled out is sound on silicon.
- C5 (eec16b773423) did NOT boot Linux. OpenSBI's banner printed through "Boot HART MEDELEG", then silence: no
  monitor tag, no Linux line. The runner timed out on login and released the board. In C3 Linux's first line follows
  the banner directly, so C5 hangs inside the Capstone monitor's init, before the handoff.
- Static checks hold: no VM-only opcode in any assembly input of C5 (dom.c.S, int_handler.c.S, sbi_capstone.S,
  init.S). The layout is identical to C3's: sup_save_region 0x8008a780..0x8008ab80 in both.
- C5d (d27587e95654) = C5 + CAPSTONE_BOOT_TRACE (wrapper f59a2b3): BT00 once the UART capability is minted, BT01
  before the save-area carve, BT02 after it, BT03 at the end of cap_env_init. Predictions, one boot:
  - no BT tag at all: the hang precedes cap_env_init's UART mint (CAPENTER, or the supervision build's global setup);
  - BT00 and BT01 only: the carve hangs (split_out_cap of sup_save_region);
  - BT00..BT02 and no BT03: a later step of cap_env_init;
  - BT03 and still no Linux: after cap_env_init.

## C5d RESULT and ROOT CAUSE; the fixed C5 pre-registered 2026-10-03 20:45:23 before its boot
- C5d printed NO boot-trace tag. Its own OpenSBI banner ends at "Boot HART MEDELEG", then silence, so the hang
  precedes cap_env_init's UART mint.
- The cause is in the generated code. capstone-c's dom_init, called from sbi_capstone_init_cap before cap_env_init,
  carves EVERY monitor global out of the top of dom_stack:
  - the board build's globals take 4,960 B of the 8 KiB FPGA dom_stack;
  - the CAPSTONE_SUPERVISED_CALL context-slot tables take 21,552 B (4 x 2 KiB descriptor pools among them).
  At 8 KiB the carve runs past the stack's base and faults before any trap vector exists. QEMU's 64 KiB hid it.
- The fix is in wrapper/supcall-fpga:
  - 36c5607: dom_stack is 32 KiB under CAPSTONE_SUPERVISED_CALL; FPGA define-off builds keep 8 KiB.
  - 882892f: CAPSTONE_PLATFORM_DEFS carries the defines to OpenSBI's assembly. Without it the first fix never took
    effect.
- A build gate (build-fpga-fw.sh) refuses a firmware whose dom_init carves + 2 KiB exceed dom_stack, or whose RW +
  36 KiB crosses 0x800A0000. It was controlled both ways: it refuses the 8 KiB supervision build and passes the C3
  build that booted. C3 rebuilt through the new wrapper is byte-identical (96884501d098), so the hook is inert when
  unused.
- Fixed C5 = da369481cacf (+ CAPSTONE_BOOT_TRACE): dom_init carves 21,552 B of 32,768 B, and _fw_end is 0x80093000.
  Predicted:
  - BT00, BT01, BT02 and BT03 print, then Linux boots;
  - then the original C5 predictions: k800 4 twice, SUPN 0, SUPK 0; speedtest 112006 38bb59fd, HEAP 2097152 DROPPED
    0 RC 0; SUPN about 1,275 (reported), SUPK 0, every SUPA 0.

## Fixed C5 RESULT (boot supmon-c5fix, fw da369481cacf), and C5q pre-registered 2026-10-03 21:07 before its boot
- **The dom_stack fix holds on silicon.** BT00, BT01, BT02 and BT03 printed, then Linux booted. C5d hung at the same
  point with 8 KiB of stack, and the only other difference is the define plumbing that carries the stack size.
- **k800 retval=4 (4,511 cycles), but it did NOT run under supervision.** The ladder host calls through
  DPI_CALL_WITH_CAP (call_domain_with_cap), and the hook covers call_domain only. No SUP or ENT tag printed for it.
  The prediction "k800 ... SUPN 0, SUPK 0" was a wrong premise of this pre-registration, not a result.
- **The speedtest ended in the pre-registered refutation, SUPK 2.** In order:
  - The arm stood: SUPA 0.
  - The domain faulted before its first preemption (SUPN 0): MCAU 2, MEPC 0x82400240 (DBAS + 0x240 = VA 0x10240),
    MTVL 0x34229073 = `csrw mcause, t0`.
  - That instruction is in the glue's `__test_reentry`. The domain entered there because the two region shares had
    already called it plainly (SHA6 twice).
  - The supervision gate forbids every plain CSR with addr[9:8] != 0, by design: supervised-call-silicon.md:65,
    csr_regfile.sv:2879-2886 at 36a641e0b.
  - The fault came back as an EVENT with the trap stripped. The monitor returned -1 to the host (SQ: X/fail), and
    the shell came back (DN_1): a real domain's kind-2 path works under the monitor on silicon (N=1).
  - The driver stopped at that stage, as designed, so the closing k800 did not run.
- **Statically, every silicon-ABI domain image does this.** sup-static-audit.py, with positive controls for every
  rule (21/21) and negative controls (0/15), flags 6 words in each of speedtest1 e6ee5255 and k800 589ceee3:
  - the glue's mcause/mtval restore at re-entry, and save at return;
  - the cycle bracket's `csrr mcycle` / `minstret`.
  Its first flag is exactly the board's MEPC and MTVL. So the C5 premise was false as written: existing domains do not
  run under supervision unchanged. Their glue and cycle bracket must avoid M-level CSRs.

**C5q** (fw bdabef34243c, Image a9e838663d64; monitor 134258f, wrapper 882892f):
- Firmware defines: SUPERVISED_CALL, CSR_EVENTS, CLASSIC_TEST, QUIET (no per-invoke lines inside the cycle bracket),
  CLASSIC_MASK = 2 (only the 2nd call_domain of the boot is supervised), BOOT_TRACE.
- The image is the speedtest PATCHED by patch-sup-glue.py: e6ee5255 -> 7291218eab669695. Exactly those 6 words
  change, and nothing else:
  - mcause/mtval restore -> nop;
  - mcause/mtval save -> `li t0, 0`;
  - `mcycle` -> `cycle`, the same counter. It is legal under supervision: sup-guards stubs 10/11 run it supervised,
    84/84 on this bitstream, and RVZicntr = 1 in the board config.
  The image's embedded initramfs was checked to carry the patched words.
- Stages: k800 (589ceee3, plain); speedtest x3; k800.
- Predicted:
  - BT00..BT03, Linux; k800 retval=4 at both ends.
  - Each speedtest run: Verification Hash 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0.
  - SUPM 0, 1, 0 for runs 1, 2, 3.
  - Run 2 (supervised): final SUPK 0, SUPN > 0 and about 1,275 (reported), no SUPA line (QUIET prints only a
    refused arm).
  - Runs 1 and 3 (plain): SPEEDTEST1-CYCLES within 0.1 % of each other and of the C3 record 2,551,615,035.
  - Supervision overhead = run 2 - mean(runs 1, 3): positive and below 1 %. Point estimate ~0.2 %, from ~4k cycles per
    preempt/resume (four 64-slot walks, the 32-CCSRRW CPMP swap, the monitor loop) x ~1,275. The value is reported.
- Refuted by:
  - any hash other than 112006 38bb59fd;
  - run 2 ending in SUPK 1/2/3 or SUPN 0;
  - an overhead of 1 % or more;
  - a wedge.
- If run 1 fails, the patch itself is broken and run 2 says nothing.
