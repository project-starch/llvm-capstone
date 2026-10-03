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

## C5q RESULT (boot supmon-c5q, fw bdabef34243c), a correction, and C5t pre-registered before its boot (pushed 2026-10-03 21:35:42 in 5569e2c4b0fe; an earlier draft of this heading said 21:45)
- **Boot and controls held.** BT00..BT03 printed and Linux booted. k800 returned 4 (4,589 cycles).
- **Run 1 (plain, SUPM 0) PASSED:**
  - 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0;
  - 2,550,641,195 cycles, -0.038 % against the C3 record.
  So the 6-word glue patch is neutral under a plain call.
- **Run 2 (supervised, SUPM 1) WEDGED.** ENT1:2, then no monitor output for the 600 s stage budget. The driver's
  wedge read put the commit pc at 0x80021188, `li sp, 0` in supervised_invoke: the instruction just before the
  supervised CALL at 0x8002118a. Runs 3 and the closing k800 did not run.
  - The two bursts of binary on the console (~1156 s and ~1186 s into the run) came after the stage timeout, during
    the driver's own wedge reads: switch 209 is odd, so it hands the console TX to the tracer. They are not monitor
    output.
  - **The C5q predictions for runs 2 and 3 are refuted.**
  - One commit-pc sample cannot tell "the resume CALL hangs" from "it escapes again forever".
  - The monitor's resume path had never run on silicon: escape kind 1, re-arm with csupctl = 1, then CALL on the
    returned seal. C5fix's call faulted before its first quantum.
- **Ruled out by reading:**
  - a timer (MTIP) livelock: create_domain zeroes the seal and writes only mstatus with MIE = 0, so mie = 0. sup-mtip
    measured that a masked timer does not escape.
  - a seal region overlapping domain memory: it is DOMAIN_DATA_SIZE = 1,536 B rounded to the granule, above the 944 B
    full save, and split apart from dom_data.
  - a lost seal: the generated code stores the CALL's rd into domains[id] before anything else.
- **CORRECTION (claim-auditor, 2026-10-03).** Under "Fixed C5 RESULT" above, "the ladder host calls through
  DPI_CALL_WITH_CAP (call_domain_with_cap)" is WRONG and withdrawn.
  - k800's only entry is the annotated region share: shared_region_annotated's plain `__domcallsaves`
    (sbi_capstone.c:2238 at 1fd1bbe). ladder_perf_ctl.c:5-6 says "that share IS the domain entry".
  - The board log agrees: k800 printed SHA0..SHA6 and no ENT0.
  - The conclusion stands: k800 never ran supervised.
- **AUDIT-TOOL GAP (same auditor).** sup-static-audit.py missed custom-0 `debug.print`, which decodes as a CSR_WRITE to
  0x800 (decoder.sv:499-513), and it skipped 16-bit words.
  - Both are added now, with controls: 24/24 flagged, negatives clean.
  - The auditor's independent raw scan of the patched speedtest finds no gated word. The re-run tool agrees: stock 6,
    patched 0.

**C5t** (fw 393210e96691; the same Image a9e838663d64; monitor f45f366):
- Defines: SUPERVISED_CALL, CSR_EVENTS, CLASSIC_TEST, CLASSIC_MASK = 1 (the first call_domain, i.e. the speedtest),
  TRACE_EVENTS (MCAU/MEPC for the first 4 preemptions), MAX_RESUMES = 8, BOOT_TRACE.
- NOT QUIET: SUPA after every arm and SUPK after every CALL print.
- Stages: k800; the patched speedtest; k800.
- Readings, and what each one means:
  - A: `SUPA:0 SUPK:1 MCAU MEPC SUPA:0`, then silence. The first RESUME CALL never returns.
  - B: kind 1 repeating up to the bound, then `SUPN:8 SUPK:1b`. The call returns -1 and the closing k800 returns 4.
    The resume works mechanically, but the loop would never have ended. The causes and epcs of the first four
    preemptions (MCAU/MEPC) say why: an immediate re-escape repeats one epc.
  - C: `SUPA:0`, then silence. The FIRST supervised CALL neither escapes nor completes.
  - D: the call completes, SUPK 0. Not expected: the speedtest needs ~1,275 quanta and the bound is 8.
  - The expected first-escape cause is the quantum, 0x8000000000000010, with MEPC inside the domain image
    [DBAS, DBAS + 0x16a388).
- Each of A, B and C names the failing step. That is the purpose of this boot, which spends no other board time.

## C5t RESULT (boot supmon-c5t, fw 393210e96691): reading B, so the resume works mechanically; and C5u pre-registered before its boot
- k800 returned 4.
- **The speedtest, supervised (SUPM 1, loud), read B.** Every cycle went `SUPA:0 SUPK:1` (each re-arm stood, each
  resume came back as kind 1):
  - MCAU 0x10, the quantum cause (the report prints only the low word of 0x8000000000000010);
  - MEPC 0x824A27B0, 0x82549E98, 0x82431A10, 0x82549E9C, all inside the domain image and different each time, so
    the domain advanced;
  - after 8 preemptions: SUPN 8, SUPK 0x1b, and a clean return of -1. The domain's own output had reached
    "100 - 500 INSERTs". The driver's HARD STOP on obs = -1 is the bound's expected value.
  - So the monitor's resume path, on silicon and through the compiler's __domcallsaves, works for at least 8 cycles.
    C5q's wedge needs something C5t did not have: more preemptions, QUIET (no ~5 ms of UART between an escape's SAVE
    walk and the resume's RESTORE walk), or being the second call of the boot.
- **The candidate mechanism class** is a store/tag ordering hazard that time hides. ISSUES.md lists the S-07/S-10b
  composed liveness ("never observed, because no test has yet opened that window") and the TAG_WAIT stall of the
  s06sec header. Both leave a CALL that never commits, which is C5q's wedge-read signature.

**C5u** (fw d07761ded4ce; the same Image a9e838663d64; monitor 78151e4):
- Masks, by bit n = call_domain n: CLASSIC = 14, QUIET = 12, FENCE = 4, TRACE_EVENTS for every preemption,
  MAX_RESUMES = 4000. SUPM reports supervised | quiet << 4 | fence << 8.
- Stages: k800; then four speedtest runs; then k800. The runs, by call:
  - call 0: plain (SUPM 0x000);
  - call 1: A, loud, every event traced (0x001). This is the SAME position as C5q's wedged run 2;
  - call 2: C, quiet with a `fence` before every arm (0x111);
  - call 3: B, quiet with no fence (0x011). This is C5q's configuration, and it is last.
- Predicted:
  - BT00..BT03, Linux, k800 4. Call 0 gives the oracle 112006 38bb59fd at ~2,550.6M cycles (within 0.1 %).
  - Every run that returns gives 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0.
  - A supervised run that completes has SUPK 0 and SUPN about 1,275 (reported). In A, every MCAU is 0x10 and every
    MEPC lies inside the domain image.
  - A's SPEEDTEST1-CYCLES includes ~13 s of UART printing in the resume loop, so it is not an overhead reading. C's
    and B's cycles, if they return, are the overhead readings against call 0, reported.
- Outcomes:
  - **A, C complete; B hangs:** store/tag ordering. A fence before the re-arm avoids it. That goes to the RTL lane as
    a bare directed test: a domain with many tagged register slots, an immediate resume.
  - **A completes; C hangs** (B is then lost): time, not ordering. The switcher-busy class.
  - **A hangs:** not a QUIET effect. A rare point after k > 8 preemptions; A's last MCAU/MEPC name it.
  - **All complete:** C5q's hang did not reproduce at N = 1. The next step is a repeat of C5q's exact configuration,
    not another variant.

## C5u RESULT (boot supmon-c5u, fw d07761ded4ce): outcome "A hangs". It is not a QUIET effect; the RESUME CALL never completes
- k800 returned 4. Call 0 (plain) gave 112006 38bb59fd, HEAP ok, 2,551,482,753 cycles (+0.03 % against C5q run 1).
- **A (loud, every event traced, C5q's position) wedged after 213 preemptions.**
  - Every one of the 213 events is MCAU 0x10 (the quantum), with its MEPC inside the domain image.
  - The tail is `SUPK:1 MCAU:10 MEPC:82851864`, then `SUPA:0` (the re-arm stood), then nothing for the 600 s budget.
  - So the resume CALL after preemption 213 neither returned nor escaped. The quantum always yields an escape once
    the domain commits anything.
  - The driver's wedge read: commit pc 0x800211b6, `li sp, 0` immediately before the CALL in supervised_invoke. This
    is the same signature as C5q, where it was 0x80021188 in that layout.
  - C and B were lost behind it, as pre-registered.
- **The escape before the hung resume** landed at VA 0x61864 in sqlite3VdbeExec (DBAS 0x82800000). It came directly
  after a capability store-to-load on the domain stack:
  - `stc a3, 0(s0-0x80)`;
  - `sb` into the adjacent granule;
  - `ldc a2, 0(s0-0x80)`;
  - `lbu` through it.
  No earlier preemption in A landed at 0x61864 (N = 1 for this sequence).
- **What it rules out:** QUIET timing. A had ~10 ms of UART between every SAVE walk and RESUME walk, and hung all the
  same.
- **What stands:** over 8 + 212 resumes worked. The hang is a property of the domain state at one escape, or of
  accumulated state, not of the resume path as such.
- **Next instrument, not a board boot of the speedtest:** a bare directed test that creates the condition. A supervised
  domain loops on STC to a granule, SB to its neighbour, LDC of the same granule, LBU through it, with a small quantum,
  so that escapes land at every offset of the sequence. Its matched control is the same loop with the capability
  store/load replaced by integer ones. It runs on silicon in the ladder harness, and the RTL lane can run it in
  simulation.

## C5f, pre-registered 2026-10-04 before its boot: C5u with the S-16 workaround (a fence before every domcall)
- **S-16 is localised and has a workaround in bare**
  (`../sup-resume-2026-10-03/`, `../../fpga-repros/S16-supervised-switch-never-finishes/`):
  - the switch's idx-7 walk write waits behind a store-buffer commit queue that never drains;
  - the RTL lane's simulation traces it to a dom-switch push into a full commit queue, which overwrites the head;
  - a `fence` right before the CALL drains the queue first, and the bare repro then COMPLETES, 8,551 escapes.
- **C5f (fw bc9206e7ccb3)** is C5u's exact configuration: the same masks, the same Image a9e838663d64 with the patched
  speedtest, monitor 78151e4. It differs in one thing: `FW_PRECALL=fence` inserts a `fence` before every one of the
  8 generated `domcall`s. Artifact gate: 8 CALL words, all preceded by a fence. Negative control: the C5u firmware
  reads 8 of 8 without one.
- Stages, as in C5u: k800; speedtest x4 (call 0 plain; A loud and traced; C quiet with a fence before each arm;
  B quiet); k800.
- **Predicted, if S-16 is C5u's hang and the fence masks it:**
  - every run gives 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0;
  - A, C and B end with SUPK 0 and SUPN about 1,275 (reported);
  - the closing k800 returns 4.
  - C5u hung at A's 213th resume.
- **Reported:** the supervision overhead from C's and B's SPEEDTEST1-CYCLES against call 0's. Each fence costs the
  drain of about 26 swap-out stores per CALL.
- **Refuted by:** any hang (read with the stages driver's apertures), or any hash other than the oracle.
- **Preflight override, recorded before the boot.** The first launch (04:47:03) was BLOCKED by the preflight and spent
  no boot. The preflight inspects the SHARED overlay, which was restored to the stock k800 (b2d60e52, at 0x10000)
  after C5u. C5f's payload carries its own saved Image a9e838663d64.
  - Its embedded initramfs, extracted and hashed: k800 589ceee3 at entry 0x20000 (the image with the published
    QEMU-pass record) and speedtest1 7291218e at 0x10000. These are C5u's staged images exactly, with no entry-VA
    collision between the run's domains.
  - The boot runs with `PREFLIGHT=0` for that reason only; no check was weakened.
- **The second launch (04:48:23) was refused by the stages driver's own freshness gate.** That gate finds the
  REFERENCE overlay's bytes inside the payload's initramfs, and its default reference is the restored shared overlay.
  - The boot runs with `SQLITE_STAGE_OVERLAY` pointing at a private directory holding the ORIGINAL intended files: the
    relinked k800 artifact 589ceee3 (`~/capstone-artifacts/k800-relinked-0x20000/k800.dom`) and the glue-patched
    speedtest 7291218e (`/tmp/capstone/mon-c0/supglue/speedtest1.dom`).
  - So the gate still verifies, by content, that the payload carries exactly the images this run intends.
  - Not from the payload's own extraction, which would be circular.

## C5f RESULT (fw bc9206e7ccb3, 04:49-05:10): the fence before every domcall DELAYS the hang but does not close it. S-16 also hits the ESCAPE switch
Lines are in `results/board-c5f.result-lines.txt`.
- k800 returned 4.
- Call 0 (plain) gave 112006 38bb59fd at 2,551,695,601 cycles.
- **A (loud) made 552 preemptions (C5u: 212), every re-arm standing, then hung after a re-arm** with signature A:
  224 = 0x1f, 225 = 0x88.
- **The stale slot-0 commit pc is 0x828d56bc, inside the domain** (DBAS 0x82800000, so VA 0xe56bc in `lookupName`).
  - Corrected the same day on the RTL lane's reading: for an escape-side wedge, slot 0 holds the resume point of the
    PREVIOUS preemption, not where this quantum landed. 0xe56bc is inside an -O0 local-init burst, but the stuck
    escape came up to one quantum later, at an unrecorded point.
  - What stands: the last instruction issued before the stuck switch was the domain's, so the stuck switch was a
    quantum ESCAPE's SAVE walk, i.e. S-16 entered from the escape side.
  - That the queue was full of the domain's own stores is the mechanism's requirement, not an observation.
- **The pre-registered prediction "A, C and B all complete" is REFUTED.** A fence before the CALL removes only the
  CALL-side trigger; no software placement covers an arbitrary preemption point.
- **Preemptive supervision on this bitstream needs the RTL fix.** The RTL lane's store-path fix (room check keyed on
  the request being decided) applies to any switcher write.
- C and B did not run.
