# Stage 2 on Line B: the CRVC cycle band and the go/no-go for a stage-2 seal (pre-registration, 2026-11-27)

The plan is plans/2026-11-25-stage2-trap-route-decoder-spec.md, section (d) of the Line B part, with the mode's home
converged with rtl (bit 0 of the srevroot write). Written after step 4's CRVCB twin reading, before any synthesis of
the stage-2 RTL (capstone-ariane stage2-crvc 796af23d9), and before the placement control's board session.

**It decides two things:**
- whether stage 2 is worth a seal: rtl's synthesis-only gate, then a full build, may proceed on it;
- what a board pair on that seal must read for stage 2 to be proposed to the lead.

It cites:
- tests/stage2-lineb-step4 (PREREG ebbf6d221cb0) for the fetch readings;
- tests/stage2-placement-b42 (f4a6996c72c9, Amendments 1 and 2: 97f4a5f07f3f, 4758c2241785) for P_s;
- the two board results that showed a relink alone moving perlbench on b42c8c644:
  - LSRH, tests/spec8-lsrh-board RESULTS e09ecb0844a4 on compiler/imagick-loop-shapes: FAIL, perlbench +1.27 %, nab
    +0.31 %;
  - ZCB, tests/stage2-zcb-b42 RESULTS d0c5af96e033: the ZCBB image FAILS, perlbench +0.41 %, nab +0.14 %.

## The image: CRVCB, and only CRVCB

- **CRVCZB is not a candidate.** ZCB does not enter (d0c5af96e033), and the placement control cannot reopen that
  verdict: its use there (Amendment 1 of stage2-zcb-b42) decides attribution, not entry. Step 4's ZCBB and CRVCZB
  readings are graded in step 4's RESULTS and enter nothing here.
- **Per-image choice under option B.** CRVCB's fetch saving is negative on nab (-3.532 %) and lbm (-2.982 %), so both
  ship without +cap-crvc.
  - On the board stages that means the ON image carries CRVCB's perlbench_r, cpugcc_r and imagick_r natives, and
    **LSHOFFL's own nab_r, byte-identical**.
  - So nab is an in-session control: its binary is the same in both images. Its band is [0, 0].
  - imagick (+0.379 %) is positive, so it ships with the feature. Its band lies below one 0.01 s tick.

## The input: step 4's CRVCB reading (twin, shift 1)

The reading is the c4 spec8 boots S4-LSHOFFL and S4-CRVCB, on capstone-qemu c42ff84d0799 (sha256 457477d73dde1a2d),
under step 4's PREREG.
- Fetch bytes come from perf/cclever/fetchbytes.py (sha256 24d05fb3bace9e7d), per native command window, summed per
  stage.
- Each window's load delta must equal its enters line's base minus 0x10000.

| stage | LSHOFFL insns | CRVCB insns | diff | LSHOFFL bytes | CRVCB bytes | B/insn | fetch saved |
|---|---:|---:|---:|---:|---:|---|---:|
| perlbench (makerand) | 3,094,620,533 | 3,094,618,527 | -2,006 | 10,619,819,362 | 8,781,923,040 | 3.432 → 2.838 | **17.306 %** |
| gcc (t1.c) | 21,710,898 | 21,709,575 | -1,323 (-0.006 %) | 77,255,052 | 70,249,950 | 3.558 → 3.236 | **9.068 %** |
| imagick | 94,485,851 | 94,485,454 | -397 | 351,237,504 | 349,907,162 | 3.717 → 3.703 | 0.379 % |
| nab | 7,154,811,027 | 7,154,811,027 | 0 | 25,574,707,108 | 26,478,095,328 | 3.575 → 3.701 | -3.532 % |
| x264 | 506,620,171,647 | 506,619,389,450 | -782,197 | 1,857,395,105,930 | 1,817,243,110,776 | | 2.162 % |
| mcf | 26,644,261,998 | 26,644,261,994 | -4 | 87,378,218,004 | 81,600,027,966 | | 6.613 % |
| xz | 49,619,823,268 | 49,619,576,139 | -247,129 | 173,345,885,300 | 170,122,591,336 | | 1.859 % |
| lbm | 6,092,401,820 | 6,092,401,820 | 0 | 23,242,341,846 | 23,935,411,038 | | -2.982 % |

- **Outputs:** EQUAL on all eight stages. Every instruction difference is within 0.006 %, which is step 4's
  prediction 1, so the twin gate for the pair holds.
- **perlbench is the makerand window alone,** the command the board times. The stage's other command (test.pl
  through natperl) is not a native window and is not counted.
  - User instructions outside the image in that window: 61,982 in both arms. The uncompressed libc and runtime
    therefore carry almost none of makerand's fetch.
  - Kernel instructions are not counted.
- **The reader was repaired before this reading.** On a small image, the search's 2,000-PC check had counted user
  PCs outside the image, and it read "0 deltas fit" on lbm. The check now applies only to PCs that a candidate
  delta puts inside the image.
  - Controls: the old and new readers agree on nab window 2 and gcc window 15, and a wrong ELF exits 1.
- **perlbench, cross-checked by an independent reader.** Instruction lengths came from llvm-objdump -d (the bytes it
  prints for each instruction), and the delta from the enters line, with no search.
  - It agrees with fetchbytes to the byte on both arms: 10,619,819,362 and 8,781,923,040.
  - With LSHOFFL's ELF under CRVCB's profile, it reads 1,236,996,935 instructions at non-starts, so the check fires.

**Open risk: the perlbench reading is twice P1's.** Step 4 predicted CRVCB within ±25 % of P1's 8.47 %, and that
prediction is refuted on perlbench.
- **The workloads differ:**
  - P1's row was 500.perlbench_r-test-sw, not makerand;
  - gcc's row was train01, not t1.c, which step 4 already excluded for that reason.
- **The six stages that share P1's workloads read P1's figures to three digits:** x264 2.162 against 2.155, mcf
  6.613 against 6.613, xz 1.859 against 1.859, imagick 0.379 against 0.375, nab -3.532 against -3.532, lbm -2.982
  against -2.982.
- Why makerand compresses about twice as well is not identified, so no per-class census is claimed. The figure is
  the campaign's own flags on the board's own workload, so it is the right input either way.

## The band

time saved_s = fetch saved_s × r_s, with r_s in [0, 0.37] (the plan; ZCB9 on P1 silicon, N=1). Scale is the three
S-LSHOFFL boots of the ZCB pair on b42c8c644 (d0c5af96e033): perlbench 278.38 s, gcc 5.56 s, imagick 11.23 s, nab
727.50 s. The plan's 275.70 s for perlbench has no source found on this branch, and it is not used.

| stage | fetch saved | band | at scale | 1 % of OFF |
|---|---:|---:|---:|---:|
| perlbench | 17.306 % | 0 to -6.403 % | 0 to -17.83 s | 2.78 s |
| gcc | 9.068 % | 0 to -3.355 % | 0 to -0.187 s | 0.056 s |
| imagick | 0.379 % | 0 to -0.140 % | 0 to -0.016 s | |
| nab | 0 (LSHOFFL's binary) | 0 | 0 | |

- **The 0.37 was calibrated at 0.889 % fetch saved.** 17.3 % is about 20 times further out, and nothing says r is
  constant across that range.
- **The decision quantity is r_s, measured:** r_s = (-Δ_s / mean(OFF_s)) / fetch saved_s, with Δ_s = mean(ON) -
  mean(OFF) on the new seal.

## The board, after the placement session's RESULTS

**The seal N:** b42c8c644 + the CRVC decoder + the mode (rtl's stage2-crvc, or its successor), sealed.
- rtl's synthesis-only gate and lint numbers come first, as CLAUDE.md requires for a change on a loop cone.
- Its readback name is recorded in an amendment before the first boot.

**The kernel K:** trapk's final tree with the patch in the plan's "The mode's home" section.
- The patch is the binfmt.h bind of table_pa | crvc, plus the boot probe.
- Its twin checks (plan (b) item 3: a flagged image runs in the mode; the leak mutant trips the guard; with the guard
  also off, an ordinary process still decodes standard RVC) pass before any board image is built.

**The images**, both on K and built by c4/spec8/board/make-board.sh as S-LSHOFFL was:
- S-LSHOFFL-K: S-LSHOFFL's natives;
- S-CRVCB-K: CRVCB's perlbench_r, cpugcc_r and imagick_r, with LSHOFFL's nab_r.
- They differ only in the three CRVC natives.
- Their sha256 go in an amendment before the first boot.

**The boots, on board 2:**
1. **K0:** S-LSHOFFL-K on b42c8c644, before the reflash. This is the seal control's reference.
2. The reflash to N, read back.
3. **Six alternating boots on N:** OFF1, ON1, OFF2, ON2, OFF3, ON3 (OFF = S-LSHOFFL-K, ON = S-CRVCB-K), as in the
   ZCB pair. No reflash and no other bitstream's boot inside them.
- **Stages:** the S bracket's four, with t = up1 - up0 on each stage's @@SPEC-END run line.
- **This session does not boot before tests/stage2-placement-b42's RESULTS is pushed,** because every rule below
  reads P_s from it.

## What decides (fixed here, before any reading)

- **The inputs:** P_s is the placement spread on stage s from tests/stage2-placement-b42's RESULTS.
  spread_s = max(P_s, range(OFF_s)) over the three OFF boots on N.
- **A stage is UNDECIDED** if any of these holds:
  - that RESULTS leaves P_s UNDECIDED on it;
  - the drift control fires;
  - the stage faults, is cut, or fails specdiff;
  - its output hashes differ between boots.

**GO for proposing stage 2 to the lead requires all three:**
1. **A gain on perlbench or gcc, beyond placement:**
   - max(ON) < min(OFF) - 0.02 s;
   - and -Δ_s - P_s ≥ 0.01 × mean(OFF_s).
   - Only the part of the saving beyond P_s counts toward the 1 %.
2. **No stage worse:** on every decided stage, Δ_s ≤ 0.0005 × mean(OFF_s) + spread_s + 0.02 s (the LSRH board form,
   with spread_s in place of the boot spread).
3. **nab holds both ways,** its binary being identical: |Δ_nab| ≤ 0.0005 × mean(OFF_nab) + range(OFF_nab) + 0.02 s.
   - If it fails, the two images differ by more than their CRVC natives, and the session is UNDECIDED, not NO-GO.

**Otherwise NO-GO.** The stage-2 seal is then not proposed, and stage 2 stops on Line B.

**The drift control** is imagick's val phase, the same plain binary in all six boots on N. It fires if
|mean(val, ON) - mean(val, OFF)| > range(val, OFF) + 0.02 s, as in the ZCB pair.

**The seal control: seal_s = mean(OFF on N) - t(K0).**
- It is reported for every stage. It says "the seal costs time on s" if seal_s > 0.0005 × t(K0) + range(OFF_s) +
  0.02 s.
- It does not change the ON-against-OFF verdict, which is read within N.
- It goes to the lead beside it, because the campaign's other programs would pay it.
- N never replaces the campaign seal unless the verdict is GO, the seal control holds, AND the lead agrees.

## Predictions

At P_s = 0, for scale only, since P_s is not read yet: GO on perlbench needs r_perl ≥ 0.058 (2.80 s), and on gcc
r_gcc ≥ 0.150 (0.076 s). A P_s of a few seconds on perlbench, the size of LSRH's 3.55 s excess, would raise the
perlbench threshold to about r = 0.13.

1. **perlbench reads a saving (Δ < 0),** with r_perl in (0, 0.37].
   - At P1's perl calibration (0.37), it reads about -17.8 s, which is a GO.
   - At P1's xz calibration (0.05), it reads about -2.4 s, which is a NO-GO even at P_s = 0.
   - So the reading separates the two calibrations P1 left open.
2. **gcc reads flat or a saving** (Δ ≤ 0), no more than 0.19 s. Whether it clears r ≥ 0.150 is not predicted.
3. **imagick reads flat,** |Δ| < 0.03 s.
4. **nab reads flat** under rule 3.
5. **The seal control holds on every stage.** The decoder is out of every path a process without the flag takes.
6. **The predicted verdict is GO, carried by perlbench.** This is a prediction, not a band. The band itself rejects
   only r > 0.37 or a slowdown.

## What this does not decide

- **Which cache or front-end effect carries any saving.** I$ misses are not read.
- Whether the libc and runtime should be rebuilt with the feature. On makerand they carry almost no fetch.
- x264, mcf, xz and lbm on silicon: they are not in the S bracket.
- The choice between option A and option B, which stays with the lead.

## Amendment 1 (2026-11-27, before any K twin boot and before any board image is built): the seal N, the kernel K, the firmware, and K's twin checks

### The seal N (rtl)

- **The bitstream:** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2n-crev4b-wap-flu-r80-r77-crvc_796af23d9.bit`.
  - sha256 0149657ee39f6ff8cf744d9081f9e8ed55362f81400174de4af9c523dd56790b, 11,443,722 B.
  - Recomputed here from the bitstreams directory.
  - The resident readback name is the same name.
- **rtl's keys, all passed:** the synthesis-only gate (3/3 keys) and the full build (llvm-capstone adc455e93544).
  - **Lint:** equal to b42c8c644 on every counter (LATCH 44, UNOPTFLAT 39).
  - **Synthesis-only gate:**
    - id_stage_i +190 LUT, +10 FF (retiming);
    - csr_regfile_i +1 FF, which is crvc_q;
    - no other module beyond ±2,500 LUT;
    - timing loops 0.
  - **The routed build:**
    - WNS +1.796 ns, inside b42c8c644's null band;
    - 0 routing errors;
    - check_timing loops 0, TIMING-23 0 and LUTLP-1 0 on the routed checkpoint, each with its toy control firing.
  - **rtl's own prediction for id_stage MISSED.** It said "near 0" of the worst 500 paths; id_stage_i is on 244 of them. That count is not a key, and its b42c8c644 baseline is pending.

### The kernel K, on copies of trapk's trees (assigned by runtime, 2026-11-27)

**Source identity, before the patch:**

| copy | of | Image sha256 | .config sha256 |
|---|---|---|---|
| K-twin | trapk/linux-6.1, the twin kernel | 749fed2606e3ebca966c3b22a95dd95271ed11ad9fe27c72bf7baf3fe96d23f3 | 8edfe80d9e0f9221701181b82c30354374611237328a77145517ec1715f1ab04 |
| K-fpga | trapk/fpga-final/linux-6.1, the board kernel (copied under its make-payload lock) | 8ae543cb8475e401975c8c848f656ea9299778b8b4d14c23f46cb42363f32724 | 200589b7523c210a1ea1d2deac6df430a5a59362df052f5ec88e5ecfea935b48 |

- The five files the patch touches are byte-identical in the two trees.
- trapk itself is untouched.

**The patch** is K/apply-K.py, whose output is K/K.diff. It is the same diff on both copies. Per the plan's "The mode's home" section:
1. **The ELF loader** (sublet/binfmt.h):
   - It binds `c->table_pa | crvc`, with crvc = 1 when e_flags carries EF_CAPSTONE_CRVC (0x100).
   - A flagged process prints "binds in the CRVC mode".
   - When the boot probe found no mode, it refuses a flagged image with -ENOEXEC and prints "CRVC image refused".
2. **The boot probe** (capstone_task.c, core_initcall):
   - It writes 1 to srevroot, reads it back, then writes 0.
   - It prints "sublet: CRVC probe read 0x... after writing 1: ...".
3. **select_root** compares `(old & ~1UL)` against table_pa (runtime: yes).

**One twin-only mutant, SUBLET_MUTANT_CRVC_LEAK** (mm/context.c), added like the series' other mutants, by one define in
asm/sublet-test.h.
- For an ordinary mm whose task is named crvcleak*, switch_mm writes 1: the mode bit with no table.
- It prints once, "mutant CRVC_LEAK wrote 1".
- The series' LEAK_ROOT mutant cannot serve here: it leaves a protected table selected, so the mode would really be on.

**The twin builds** (c4/kernels/build-K-twin.sh, incremental in the K-twin copy, config unchanged at 8edfe80d…):

| variant | defines | Image sha256 |
|---|---|---|
| plain | none | 25846ddf235d493569780299fe86cc68cad63fc035694df604b6c27c77423498 |
| leak | SUBLET_MUTANT_CRVC_LEAK | 564c9798e4449afdc6624d5a17679b44bd01d230b452bddeece75ce233a7d862 |
| leakng | CRVC_LEAK and SUBLET_MUTANT_NO_GUARD | 669e6aaaa76416d3c3de5e2ffdea7da77c0e4d7a6d8395b23b15c8e90a2fa997 |

The board kernel is K-fpga, built by its own copy of make-payload.sh when the images are built. Its Image and payload
sha256 go in the amendment that lists the images.

### The firmware (assigned by runtime, 2026-11-27)

- **The source:** fpga-final/make-payload.sh builds the board firmware with `make -s -C <hdd>/sbi-keep -j6
  PLATFORM=fpga/ariane ... FW_CAPSTONE_KEEP_TAGS=${KEEP:-y} ${FW_EXTRA:-}`. That tree is sbi-keep
  25b14da6f53c4af4384f78d05d362658be04e7fe, clean.
- **The copy:** a clone at that commit, with K/sbi-crvc-apply.py applied, committed locally as 09eda70de65b. The diff
  is K/sbi-crvc.diff. sbi-keep and the shared firmware are untouched.
- **What it changes:** a misaligned load or store trapped from U is redirected to S with its own cause, not emulated,
  when all of these hold:
  - the CRVC mode is on;
  - the instruction at mepc is 16-bit, in Q0 or Q2 with funct3 001 or 101. Those are c.ldc, c.stc, c.ldcsp and
    c.stcsp, which the emulator would otherwise take as C.FLD, C.FSD, C.FLDSP and C.FSDSP.
- **The condition, quoted from the patched source:**
  - `if (((regs->mstatus & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT) != PRV_U)`, with MPV clear;
  - `root = csr_read(0x5c1); if (!(root & 1) || !(root >> 12)) return false;`;
  - `return q != 1 && (f3 == 1 || f3 == 5);`.
  - With bit 0 clear, the redirect cannot fire. That is the stated claim for K0 and every OFF process.
- **Why only the instruction is read:** on b42c8c644 and N, a misaligned LDC/STC raises cause 4/6 with mtval 0 and
  mtinst 0 (rtl, from source: capstone_dyn_unit.anvil:801-804 and :871-873; cva6.sv:1625 and :1630-1632). So the
  test reads only the instruction at mepc and srevroot.
  - Without the change, the board's emulator would decode the access as C.FLD at address 0.
- **Observable:**
  - A banner line at boot: "FWCRVC: stage-2 misaligned redirect".
  - Two counters, redirects and emulations from U, print "FWCRVC redirect=N emulate-u=M" each time either reaches a
    power of two.
  - **There is no line at the end of a board boot.** The board kernel never calls SBI at power-off: its
    CONFIG_RISCV_SBI_V01 is off, and fpga/ariane registers no reset device. So a count is read as the last FWCRVC
    line of the boot, exact for N < 2 and bracketed between powers of two above that.
- **One firmware for every K boot (K0, OFF, ON):** the fpga/ariane build of this clone, with FW_FAST_RDTIME=y, as
  make-board.sh passes. The twin's build is the generic platform with FW_CAPSTONE_KEEP_TAGS=y, as fw-y is:
  - fw-crvc fw_jump.elf 3298bb764c7b6380;
  - the stock fw-y fw_jump.elf is 67115ce6494bb87d.
- **The firmware's positive control: one stage, last in every K image.** Runtime asked for the ON image only; this
  puts it in all three, which keeps K0, OFF and ON differing in nothing but the three CRVC natives.
  - **The stage, crvcmis:** a CRVC-flagged native (K/crvcmis.c, built by CRVCB's capstone-cc, e_flags 0x101; strip.dom
    41088a87663588e1). It runs one `c.ldc a1, 0(a0)`, the fixed word 0x210c, at 8 bytes past a 16-byte boundary,
    inside its object's bounds.
  - **On N (OFF and ON):** the mode is present, so the load traps misaligned to M. **Predicted:** one
    "FWCRVC redirect=1" line, the process killed, and no "no trap" line.
  - **On b42c8c644 (K0):** the probe reads 0 and the loader refuses the image. **Predicted:** "CRVC image refused" and
    no FWCRVC redirect. This is the silicon control for the probe and the refusal.
  - It runs after the four timed stages, so it cannot move them.

### K's twin checks

All run on capstone-qemu c42ff84d0799 (457477d73dde1a2d), each under the QEMU lock. None uses
x-capstone-crvc-protected: the mode comes only from the srevroot write.
- **The KTEST boots** run K/ktinit.c as /init (ktest.cpio.gz db73d12be719b215).
  - It polls each child with sched_yield rather than blocking, so the child's wakeups are switches between two mms
    and switch_mm runs for it.
  - The leak checks are VOID, not passed, if "mutant CRVC_LEAK wrote 1" does not appear.
  - It runs first the ordinary program crvcleak (K/crvcleak.c, buildroot gcc, static, sha256 4511c71c63df4d11; one
    c.fld and one c.fsd after four sleeps), then crvcmis. Each boot is bounded at 300 s.
- **The stage boots** use c4/spec8/drive4.py, which is drive3.py with the kernel and firmware taken from the
  environment. They run at shift 1 with capmix3, and are read by count8s4 and fetch8 against step 4's boots.

| check | kernel | firmware | x-capstone-crvc | runs | predicted |
|---|---|---|---|---|---|
| KA | K plain | fw-crvc | true | KTEST | probe "the CRVC mode is present". crvcleak exit 0, "standard RVC". crvcmis "binds in the CRVC mode", then one "FWCRVC redirect=1" line and a signal death, with no "no trap" line |
| KB | K plain | stock fw-y | true | KTEST | crvcmis is emulated as C.FLD: "no trap", exit 0, no FWCRVC line. This is the defect the change removes, read on the twin, where mtval is the real address |
| KC | K plain | fw-crvc | false | KTEST | probe "no CRVC mode". crvcmis "CRVC image refused", exit 127. crvcleak exit 0 |
| KD | K leak | fw-crvc | true | KTEST | "mutant CRVC_LEAK wrote 1". The guard prints "srevroot 0x1 does not match crvcleak", and crvcleak dies of signal 9 |
| KE | K leakng | fw-crvc | true | KTEST | "mutant CRVC_LEAK wrote 1" with no guard line. crvcleak exit 0, "standard RVC": with srevroot[63:12] = 0 the mode stays off |
| KT1 | K plain | fw-crvc | true | LSHOFFL natives, 4 stages | **identity:** counts equal S4-LSHOFFL within ±0.001 % per stage, outputs EQUAL, no "binds in the CRVC mode" line |
| KT2 | K plain | fw-crvc | true | CRVCBN (CRVCB's perlbench, gcc, imagick; LSHOFFL's nab), 4 stages | **the ON image's content:** perlbench, gcc and imagick within ±0.001 % of S4-CRVCB, nab within ±0.001 % of S4-LSHOFFL, outputs EQUAL to S4-LSHOFFL's, a "binds in the CRVC mode" line for each CRVC native's processes and none for nab's |
| KT4 | trapk 749fed26 (unpatched) | stock fw-y | true | CRVCB gcc and imagick, 2 stages | **the mode comes from K:** the unpatched loader binds without bit 0, so both stages die of SIGILL (rc 132) |

- **Inputs:** KT1, KT2 and KT4's initramfs are c593f1549e5bbc1a, d17db9b8d6def0d6 and 6d8a124a81762ae6. Their
  natives equal the arms' strip.dom images.
- **Any prediction that fails** stops the board images until it is explained. A refuted check is reported, not
  re-run into agreement.

### From runtime's reading of 4d0f6b69116f

- **The 275.70 s** is nested's campaign reading of CF-B5TL1-g4mk on board 2 (nested-workloads RESULT a162a8aefb60),
  on B5TL1's userland and trapk's final kernel. It is a different image from the S images, so it is still not used
  for scale.
- **The session plan gains a last step:** reflash board 2 back to b42c8c644 and read it back.
  - The session is booked only after this amendment's checks pass.
  - The slot is after camp2x2-b2's .done, or one runtime and the board lane find.
