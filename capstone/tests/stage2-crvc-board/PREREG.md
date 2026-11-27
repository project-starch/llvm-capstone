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

## Amendment 2 (2026-11-27, before any K twin boot): the firmware reads srevroot last and prints only at a redirect

Both changes answer runtime's two questions on Amendment 1. The redirect's verdict is unchanged for every trap. The
clone's head is now 514047391ee0 (09eda70de65b, then e8aa312dbace, then 514047391ee0). K/sbi-crvc-apply.py and
K/sbi-crvc.diff are updated, and the script re-applied on a fresh 25b14da6 reproduces both files byte for byte.

1. **The CSR is read last.** crvc_decoded now tests MPP = U (with MPV clear), then the 16-bit slot (Q0/Q2, funct3
   001/101), and reads srevroot only for an instruction that passed both:
   - `root = csr_read(0x5c1); return (root & 1) && (root >> 12);`
   - An ordinary misaligned trap, a C.FLD or an LD from U on K0, never executes the read.
   - The read is legal from M on both revisions anyway. The CSR_SREVROOT read case sits outside both
     CAPSTONE_SUBLET_FMT blocks: b42c8c644 csr_regfile.sv:1104 `csr_rdata = srevroot_q;`, and 796af23d9 :1109
     `csr_rdata = srevroot_q | {63'd0, crvc_q};`.
2. **One line per redirect, and no other runtime line.**
   - Each redirect prints `FWCRVC redirect=N emulate-u=M` with both exact counts. Emulations from U are only counted.
   - Amendment 1's power-of-two lines would have printed inside timed stages, where firmware output can splice into a
     program's console lines.
   - In a clean boot, the FWCRVC lines are the banner (before Linux) and the crvcmis stage's redirect line, after the
     four timed stages.
   - A redirect inside a timed stage kills that stage's process, so its line can only fall in a stage that fails, and
     that stage is UNDECIDED regardless.
   - So no reader of a timed stage can meet a FWCRVC line in a stage it reads. This is checked at grading: every
     FWCRVC line's position against the stage markers.
   - **The reading:** on N, the crvcmis line's two counts are exact. On K0 no redirect line is predicted, since
     crvcmis is refused there, so K0's emulation count is not read.

**The twin firmware** (generic platform, FW_CAPSTONE_KEEP_TAGS=y): fw_jump.elf
6804547d1f518dda69437f9ddffda6c269d079f0500bc16cbdc2e4fe0c57631b. It replaces Amendment 1's 3298bb764c7b6380 for
every check that names fw-crvc. The fpga/ariane build of the same head compiles clean.

**Prediction KA is restated for the new print:** exactly one `FWCRVC redirect=1 emulate-u=M` line, which falls inside
the crvcmis window, after "KTEST begin /crvcmis.dom". Every other row is unchanged.

## Amendment 3 (2026-11-27, after KA's boot, before any rerun): crvcmis did not run in KA, which is VOID for crvcmis; a fixed init reruns KA, KB and KC

**What KA read.**
- **Usable:** the banner "FWCRVC: stage-2 misaligned redirect"; "sublet: CRVC probe read 0x1 after writing 1: the
  CRVC mode is present"; crvcleak "standard RVC", exit 0.
- **crvcmis never ran.** The loader printed "crvcmis.dom[61]: load failed after the point of no return: -13 (line
  392)", and the child died of signal 11.
- **The cause:** line 392 is dom_resources_init's error check. That function refuses a program whose working
  directory is the whole filesystem: `if (!strcmp(d->root_name, "/")) return -EACCES;`, with the comment "A
  whole-filesystem cwd is not a useful confinement grant".
- ktinit ran its children from /. The spec init does `cd /tmp/run` first, and stage commands run inside their stage
  directory, so no stage boot is affected.
- **So KA's crvcmis prediction is VOID,** neither met nor refuted. The check never reached the instruction. The same
  holds for the crvcmis halves of KB and KC, which run the same init in the queue now. Their crvcleak halves, and KD
  and KE, which test only crvcleak, stand as pre-registered.

**The rerun.** K/ktinit2.c is ktinit with `mkdir("/w"); chdir("/w")` before the children, printing "KTEST cwd /w".
- Its initramfs, ktest2.cpio.gz (10ce44205e17603e), has the same three files plus that init.
- It is booted by kt-boot2.py, which is kt-boot.py with the initramfs named by KTCPIO.
- KA2, KB2 and KC2 repeat KA, KB and KC exactly otherwise, with the same kernels, firmware, properties and
  predictions. They run after the queued KT4, KT1 and KT2.

**For the board images:** the crvcmis stage runs through the spec init's stage runner, inside its stage directory,
never from /. That is checked when the images are built.

## Amendment 4 (2026-11-27, before any board image is built): K's twin checks, read; every prediction holds

All boots ran on capstone-qemu c42ff84d0799 under the QEMU lock. Each boot log's first line is its QEMU command, and
the kernel, firmware, initramfs and properties were checked there against Amendments 1-3.

| check | read | verdict |
|---|---|---|
| KA2 (rerun of KA) | probe "the CRVC mode is present". crvcleak "standard RVC", exit 0. crvcmis "binds in the CRVC mode", printed its first line, then exactly one `FWCRVC redirect=1 emulate-u=0`, inside its window, and died of **signal 7 (SIGBUS)**. No "no trap" line | **HOLDS** |
| KB2 (rerun of KB) | stock fw-y, no FWCRVC banner. crvcmis "binds in the CRVC mode", then "no trap, the load returned (null)", exit 0: the misaligned c.ldc was emulated as C.FLD | **HOLDS** (the defect, read) |
| KC2 (rerun of KC) | probe "read 0x0 ... no CRVC mode". crvcmis "CRVC image refused: this hart has no CRVC mode", Exec format error, exit 127. crvcleak exit 0 | **HOLDS** |
| KC (first run) | the same refusal and exit 127. The refusal precedes the loader's cwd check, so this half read even under the old init | holds (KC2 repeats it) |
| KD | "mutant CRVC_LEAK wrote 1 for crvcleak[60]", then "guard: srevroot 0x1 does not match crvcleak[60], killed", and crvcleak ended with signal 9 | **HOLDS** |
| KE | "mutant CRVC_LEAK wrote 1 for crvcleak[60]" with no guard line. crvcleak "standard RVC", exit 0 | **HOLDS** |
| KT4 | on unpatched 749fed26 with x-capstone-crvc=true, gcc and imagick run rc=132 (SIGILL). No probe line: that kernel has none | **HOLDS** |
| KT1 | against S4-LSHOFFL: nab 0, gcc 0, imagick -88, perlbench 0. Outputs EQUAL. 0 "binds in the CRVC mode" lines. Probe and banner present. The one known test.pl fault | **HOLDS** (identity) |
| KT2 | against S4-CRVCB: perlbench 0, gcc 0, imagick +267. nab against S4-LSHOFFL: +4. Outputs EQUAL against both. "binds in the CRVC mode" for cpugcc_r, imagick_r, perlbench_r and 317 .perl-native processes, none for nab_r. No SIGILL | **HOLDS** |

- **Every imagick difference** (-88, +267) lies inside the ±0.001 % band of 945 instructions. imagick's count has moved
  by up to 178 instructions between boots of one binary all night (Amendment 3 of stage2-placement-b42).
- **What this shows on the twin:**
  - K, the patched firmware and x-capstone-crvc together change nothing for unflagged images: instructions and
    outputs are unchanged (KT1).
  - The ON image's natives decode in the mode from K's bind alone, without the twin-only property (KT2, with KT4 as
    the control).
  - The refusal works (KC2).
  - The guard catches a leaked mode bit, and a leaked bit without a table does not decode in the mode (KD, KE).
  - The firmware redirects the one misaligned CRVC access that the stock firmware silently emulates (KA2 against
    KB2).
- **Not shown on the twin:** the board's misaligned trap carries mtval 0 where QEMU's carries the address. So KB2's
  silent emulation would on silicon be an emulated access at address 0. The redirect's verdict does not read mtval.

**The crvcmis predicates for the board,** stated so that the board lane can check them mechanically:
- **N (OFF and ON):** exactly one `FWCRVC redirect=1 emulate-u=M` line inside the crvcmis stage, then the program
  killed by SIGBUS, rc 135 in the stage's shell, with no "no trap" line.
- **K0 (b42c8c644):** "CRVC image refused" and exit 127, with no FWCRVC redirect line.

Board images are built next, and their sha256, the stage's directory name and the "boot:" line go in the next
amendment.

## Amendment 5 (2026-11-27, before any K board boot): the images, the control stage, the boot list and the K0 reading

**boot: K0, FLASH-N, OFF1, ON1, OFF2, ON2, OFF3, ON3, FLASH-B42**

This is the session's single boot list, on board 2, in its order:
- **K0:** S-LSHOFFL-K on b42c8c644, the bitstream resident before the session.
- **FLASH-N:** the reflash to N (Amendment 1's name and sha256), with its readback name recorded.
- **OFF1 to OFF3:** S-LSHOFFL-K on N.
- **ON1 to ON3:** S-CRVCBN-K on N.
- **FLASH-B42:** the reflash back to b42c8c644, with its readback.

**Unchanged:** a reflash or another bitstream's boot between OFF1 and ON3 voids the six.
- **The slot:** after camp8x-b2-20261127T0229's .done, and not before tests/stage2-placement-b42's RESULTS is
  pushed.

### The images

Built by c4/spec8/board/make-board-K.sh. That is make-board.sh's S image (the same four campaign stages, init, probes
and smoke.dom, 3,600 s each), plus a fifth stage, K-crvcmis, relinked by K-fpga's copy of make-payload.sh with
FW_FAST_RDTIME=y.
- **The kernel:** K-fpga. Its .config differs from fpga-final's in one line, CONFIG_INITRAMFS_SOURCE, the path of
  the copy.
- **The firmware:** the sbi-crvc clone at 514047391ee0, built for fpga/ariane, with the stock DTB caplifive-b5b2.dtb.

| image | natives | fw_payload.bin sha256 | Image sha256 | bytes |
|---|---|---|---|---|
| S-LSHOFFL-K (K0, OFF) | LSHOFFL's four | ac582c15dbb9c2087b7102573e2ed76472f15d756238eed3e713c9d5e489aa30 | 689b29f6cf765552f24c4c5c21844b743350f56f4cecd4165f47b2873eb12d89 | 25,856,520 |
| S-CRVCBN-K (ON) | CRVCB's perlbench_r, cpugcc_r, imagick_r; LSHOFFL's nab_r | a3b08c74c67c1014db77726ebd5abd09bd25a5ffa19b76dbe5e2759d92c35ae8 | 68914a64d926e4e04333bf2af8da94cac2c082bafe3a9bd9c8900f8df7f20b8d | 25,856,520 |

- **Each image's overlay check:** 574 files equal to source, 4 natives equal to the arm, 0 mismatches. keep=y,
  fast-path-refs=2. Each payload carries the FWCRVC banner string, and each Image the probe string.
- **The two overlays** (585 files each) differ in exactly the three CRVC natives: perlbench_r f3c8c80d… → fb76f981…,
  cpugcc_r 87ef79e2… → 805a16a9…, imagick_r a666e331… → f6a8d2f3…. nab_r is da57e5473a33e6fd in both.
- **Against S-LSHOFFL's overlay,** S-LSHOFFL-K differs only in stages.order and stages.bound, which gain the fifth
  stage's lines, and in the stage's two new files.
- **The size** equals the S images' 25,856,520 B, proven on b42c8c644 by the S bracket's first boot. On N it is
  unproven until N's first boot. C10's ruling for these images is runtime's.

### The control stage: K-crvcmis

- **Directory:** /stages/K-crvcmis. It is last in stages.order, with bound 120 s.
- **Files:** stage/crvcmis.dom (41088a87663588e1, the KA2 binary) and stage/run.sh (03226be0f2b018a9).
- **run.sh** runs crvcmis from /tmp/crvcmis, never from /, and prints `@@CRVCMIS rc=<rc>`. It always exits 0, so no
  grader reads the control's rc as a failed stage.
- **The predicates on the capture,** within the stage's window, from "TRAPTEST: stage K-crvcmis begin" to its
  "TRAPTEST: dmesg" lines:
  - **N (OFF1-3, ON1-3):**
    - exactly one `FWCRVC redirect=1 emulate-u=M` line;
    - `@@CRVCMIS rc=135` (SIGBUS, as KA2 read on the twin);
    - a dmesg line "crvcmis.dom[...] binds in the CRVC mode";
    - no "no trap" line.
  - **K0 (b42c8c644):**
    - the dmesg line "CRVC image refused: this hart has no CRVC mode";
    - `@@CRVCMIS rc=127`;
    - no FWCRVC redirect line;
    - the boot's probe line reads "no CRVC mode".
  - **Every boot:**
    - the banner "FWCRVC: stage-2 misaligned redirect" before Linux;
    - no FWCRVC line inside any of the four timed stages.
- A boot whose control reads otherwise is reported as that reading. It is not VOID, and it does not stop the session.
  The four timed stages are graded as before.

### K0's own reading: K and the firmware on b42c8c644

K0 is the seal control's reference, and also the only silicon reading of K and its firmware's time on unflagged
images. Its reference is every S-LSHOFFL boot on b42c8c644 with stock kernel and firmware:
- the ZCB pair's OFF1-OFF3 (d0c5af96e033);
- the placement session's LSHOFFL and LSHOFFL2.

Per stage, the rule:
- K0 holds if lo - q ≤ t(K0) ≤ hi + q, where lo and hi are the reference boots' min and max and q = 0.0005 × hi +
  0.02 s.
- If it fails on a stage, the excess is K's or the firmware's, not N's. The seal control is then UNDECIDED on that
  stage, and the difference is reported.
- **Predicted:** K0 holds on all four stages. The probe runs once at boot, and the firmware's tests run only on
  misaligned traps from U.

## Amendment 6 (2026-11-27, before any K board boot): K0's crvcmis predicate is rc 2, not 127, as both images' board-form twin boots read

**The boots.** Each K image's own initramfs (out/initramfs.cpio: base rootfs, the board init and all five stages) was
booted on the twin: K-twin plain, the fw-crvc twin build, the board's bootargs, under the QEMU lock, with no profiler.
- These are harness tests of the board form, not measurements. They are the captures the board lane's grader is
  tested on.
- **BT-ON:** S-CRVCBN-K with x-capstone-crvc=true, log sha256 19f2ddc9ea8fbebd.
- **BT-K0:** S-LSHOFFL-K without the property, so the probe reads 0, as on b42c8c644. Log sha256 d639a7f0b6bdf255.
- Both ran all five stages to "TRAPTEST: done". Each timed stage returned rc=0.

**BT-ON meets Amendment 5's N predicate.** Inside the K-crvcmis window it read:
- "crvcmis.dom[954] binds in the CRVC mode";
- "crvcmis: loading a capability at offset 8 ...";
- exactly one `FWCRVC redirect=1 emulate-u=0`;
- `@@CRVCMIS rc=135`;
- and the binds line again as "TRAPTEST: dmesg".
- The timed stages' CRVC natives bind in the mode, and nab does not. No FWCRVC line falls inside a timed stage.

**BT-K0 refutes Amendment 5's K0 rc.** It read:
- "sh[954] CRVC image refused: this hart has no CRVC mode";
- `./crvcmis.dom: line 1: syntax error: unexpected ")"`;
- `@@CRVCMIS rc=2`;
- no FWCRVC line other than the banner.
- **Why the prediction was wrong:** 127 came from KC2, where ktinit calls execl and exits 127 when it returns. run.sh
  runs the file from busybox sh instead. When execve fails with ENOEXEC, sh runs the file as a shell script, as
  POSIX specifies, and the script fails to parse.

**K0's predicate becomes, within the window:**
- the line "CRVC image refused: this hart has no CRVC mode", on the console or as "TRAPTEST: dmesg";
- the syntax-error line;
- `@@CRVCMIS rc=2`;
- no FWCRVC redirect line.
- N's predicate is unchanged.
- **The window**, for both: from "TRAPTEST: stage K-crvcmis begin" to "TRAPTEST: done". The kernel's lines reach the
  board console only through the init's dmesg dump after the stage's rc line.

## Amendment 7 (2026-11-27, before any K board boot): the seal control's spread takes the measured boot spread

The placement session (tests/stage2-placement-b42 RESULTS, c89f177d0403) read B_s, the spread between two boots of
one image on one bitstream:
- perlbench 1.58 s, nab 0.45 s, imagick 0.02 s, gcc 0.00 s.
- That is larger than the ZCB pair's range(OFF) on perlbench (0.56 s) and nab (0.08 s).

The seal control compares one K0 boot with N's three OFF boots. With range(OFF) alone as its spread, one boot's
excursion of B_s's size would read as "the seal costs time". So, as the go/no-go already does with P_s:
- "the seal costs time on s" now needs seal_s > 0.0005 × t(K0) + max(range(OFF_s), B_s) + 0.02 s.
- Everything else in the seal control and in K0's own reading (Amendment 5) is unchanged.

**The thresholds this fixes,** from the same RESULTS, for the record:
- P_s is perlbench 11.38 s, nab 2.21 s, imagick 0.05 s and gcc 0.00 s.
- A perlbench GO needs -Δ ≥ about 14.2 s (r ≥ 0.29).
- A gcc GO needs about 0.06 s with the separation rule (r ≥ 0.12).
- **Prediction 6** ("GO, carried by perlbench") stands as written. It is graded as written, and at P_perl = 11.38 s it
  now needs r near the band's top.
