# Compressed capability ISA, stage 2: the option B prototype (compiler lane, 2026-11-21)

The census (`history/21-11-2026_03-30-00_stage2-compressed-capability-census.md`) read GO: the encoding set saves
9.77 % of perl's executed fetch bytes and 15.35 % of gcc train01's. The runtime lane chose option B, a decode mode,
for the prototype. Option A stays a one-bit variant: the encodings are the same, only how the mode is selected
differs. Which of the two the ISA adopts is a question for the lead.

This document defines the mode, then the prototype and what it must read. The mode's design was approved by the
runtime lane (2026-11-21). Its two ABI allocations are recorded in `design/virtual-capstone/isa.md` (options bit 2)
and `design/capability-calling-convention.md` (`EF_CAPSTONE_CRVC`). Nothing on the RTL is built.

## The decode mode

**Where it lives.** Bit 2 of the CSRUNV admission options word, frame +56 (`CRVC`). Today bit 0 is resumable ECALL and
bit 1 resumable page faults, and any other bit is refused at admission (`options & ~3`). With the mode the
refusal becomes `options & ~7`. Like any nonzero option, it needs the 640-byte frame.

**Reset and default.** 0. There is no hart-level state: the bit exists only in a context's admission options and its
saved continuation.
- A context admitted with it clear decodes standard RVC, exactly as today.
- No other code ever decodes in the mode: the S kernel, the M monitor, Linux U processes, and physical C
  outside a virtual context.

**What it changes, and only while the context runs** (virtual C, `capstone_virtual_c`):

| 16-bit slot | mode clear (standard) | mode set |
|---|---|---|
| Q2 funct3 001 | C.FLDSP | c.ldcsp rd, uimm(sp): `ldc rd, uimm(sp)`, uimm[9:4] in C.LQSP's layout |
| Q2 funct3 101 | C.FSDSP | c.stcsp rs2, uimm(sp): C.SQSP's layout |
| Q0 funct3 001 | C.FLD | c.ldc rd', uimm(rs1'): C.LQ's layout, uimm[8:4] |
| Q0 funct3 101 | C.FSD | c.stc rs2', uimm(rs1'): C.SQ's layout |
| Q1 funct3 011, rd = x0 | C.LUI x0 (a HINT) | c.cincoffset16sp nzimm: `cincoffsetimm sp, sp, nzimm`, C.ADDI16SP's immediate layout |
| Q0 funct3 100, [12:10] = 100 | illegal | c.movc rd', rs' (rd' at [4:2], rs' at [9:7], Zcb's field convention) |
| Q0 funct3 100, [12:10] = 101 | illegal | c.scc rd', gp, rs' |
| Q0 funct3 100, [12:10] = 110 | illegal | c.cincoffset rd', rs' (rd' = rs1' at [9:7], rs' at [4:2]) |

Each expands to the existing 32-bit Capstone instruction, so the datapath is unchanged.
- In the RV128 layouts the immediate's low four bits are implied zeros, and an all-zero immediate is legal for the
  loads and stores.
- `c.cincoffset16sp` with nzimm = 0 stays illegal, as C.ADDI16SP's does.
- Q0 100 111 and the two Q1 unary slots stay reserved.

**Across traps and context switches.** The bit belongs to the context, not to the hart.
- **Escape.** A page fault, a quantum or an ECALL escapes to the trusted S caller, which decodes standard RVC from its
  first instruction. The escape is a pipeline flush, so no instruction fetched in the context is decoded in the
  caller's mode.
- **Resume.** CSRUNV action 0 or 2 restores the saved options with the rest of the continuation (QEMU
  `call->virtual_options`). An adopted context reloads them from the saved-options word at frame +944, as it does
  today. The resume is also a flush.
- **No change from inside.** The context cannot change the bit: virtual C has no instruction that writes the options.
- **Threads.** A new thread is admitted with its own options. The adapter copies the parent's CRVC bit, so every
  thread of an image decodes alike.

**Who sets it.** The trusted adapter (`capstone_vm.ko`), at first entry, from the image's ELF header:
- `e_flags` bit `EF_CAPSTONE_CRVC`, proposed as 0x0100, clear of RISC-V's assigned bits (0x0001-0x0010);
- an image with the flag is admitted with CRVC;
- an adapter that cannot set the mode must REFUSE such an image. Decoded in standard mode, `c.ldcsp` executes as
  `c.fldsp`, a silent wrong-instruction, not a fault.

**What must hold inside a CRVC image.** The whole image is decoded in the mode. So no object in it may contain the
displaced standard forms: C.FLD, C.FSD, C.FLDSP, C.FSDSP and C.LUI x0. Three guards:
- the compiler under the feature never emits them, and uses the 4-byte FLD/FSD instead;
- it marks every object it compiles with `EF_CAPSTONE_CRVC`;
- the linker refuses to combine an object with the flag and an object that has `EF_CAPSTONE_RVC` but not
  `EF_CAPSTONE_CRVC` (an error, as a float-ABI mismatch is). An object without `EF_CAPSTONE_RVC` was compiled
  without C and contains no 16-bit instruction, so it may join a CRVC image.

The SDK's prebuilt libc and runtime are compiled without C and so join under that rule. Production would rebuild them
with the feature.

## The prototype

The prototype measures the gain without pre-empting the ISA choice: compiler feature off by default, QEMU switch off
by default, byte-identical when off.

1. **MC layer.** Instruction definitions in the RV128 layouts above, behind the subtarget feature `cap-crvc` (off by
   default). Lit pins:
   - assembler and disassembler round trips, each immediate at both boundaries;
   - negative tests: a register outside x8-x15, an unaligned offset, and nzimm = 0 do not assemble to a 16-bit form;
   - with the feature, the disassembler reads the five slots as the new forms; without it, as before.
2. **QEMU** (the twin's tree, `native/c5a-twin55-check` at a0501ebc39, plus the decode change only). The switch
   `x-capstone-crvc` admits every virtual context as if CRVC were set: every twin image is built with the feature.
   - Acceptance, on the stage-1 RVC arm's pattern: every new op at a 2-mod-4 pc, under a trap, across an escape
     and resume.
   - Two-sided control: with the switch off, the D slots still decode as FP and C.LUI x0 is still a HINT.
   - Refusal (runtime lane, 2026-11-21): a flagged image given to an adapter or a kit that cannot set the mode is
     REFUSED at admission, not run.
   - An escape in the middle of a compressed sequence (a page fault on a c.ldcsp), then a resume: execution
     continues in CRVC decode at the faulting instruction.
   - The native route's contexts are entered through CSRUNV as well, so the mode reaches the paper's footing. This
     is to be confirmed in the twin before step 4.
3. **Codegen.** CompressPats on the stage-1 `+cap-base-rvc` model, plus the FP-form exclusion and the ELF flag.
   - With the feature off, the 678 SPEC objects and the 14 fixtures are byte-identical to the previous build.
   - The first positive check is static: compressed sites per candidate per image, against the census's static
     column less the libc and runtime share (pre-registered below, before step 3 runs).
4. **Twin runs** on perl and gcc (specdiff), then the rest. Reported in instructions and fetch bytes; the cycle
   question waits for the RTL step.

## Step 3 pre-registration (2026-11-21, before the feature-on arm is built)

**Arms.** CRVC0 is FRCDKMSa3's SDK, runtime, libc and flags with the current compiler, the feature off: clang
518578ef40a76b19, lld b9f671e793ab869d, built from dev f94bb28f768b with step 1 compiled in. CRVC1 is the same
plus `+cap-crvc` and a link script that writes `CV_IMAGE_MAGIC_CRVC`. The feature-off arm is the baseline, not
FRCDKMSa3, because the compiler has moved since 29027126f39b.

**The prediction, static, per benchmark.** Counted on CRVC0's compiled objects: the 4-byte instructions that fit each
form, and the 16-bit FP forms. The prebuilt libc and runtime are outside the count, and the feature does not touch
them.

| benchmark | objects | c.ldcsp | c.stcsp | c.ldc | c.stc | c.cincoffset16sp | c.movc | c.scc | c.cincoffset | set | c.fldsp | c.fsdsp | c.fld | c.fsd | 4-byte fld/fsd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500.perlbench_r | 69 | 19,105 | 13,153 | 18,094 | 6,145 | 4,349 | 13,448 | 15,814 | 1,398 | 91,506 | 196 | 150 | 235 | 18 | 93 |
| 502.gcc_r | 387 | 89,085 | 59,775 | 77,962 | 28,118 | 22,653 | 94,218 | 67,036 | 7,496 | 446,343 | 54 | 37 | 67 | 17 | 54 |
| 505.mcf_r | 12 | 213 | 194 | 82 | 94 | 46 | 118 | 35 | 3 | 785 | 2 | 2 | 0 | 0 | 1 |
| 519.lbm_r | 2 | 91 | 95 | 24 | 6 | 28 | 31 | 23 | 4 | 302 | 31 | 32 | 32 | 5 | 355 |
| 525.x264_r | 96 | 10,391 | 7,108 | 5,709 | 1,686 | 1,695 | 3,881 | 3,802 | 2,475 | 36,747 | 444 | 340 | 208 | 65 | 469 |
| 538.imagick_r | 98 | 18,484 | 10,422 | 9,402 | 5,042 | 3,595 | 11,988 | 12,989 | 857 | 72,779 | 2,888 | 1,885 | 2,627 | 806 | 5,560 |
| 544.nab_r | 20 | 1,422 | 1,276 | 1,073 | 464 | 344 | 1,116 | 1,191 | 120 | 7,006 | 246 | 198 | 360 | 195 | 886 |
| 557.xz_r | 90 | 1,508 | 1,342 | 447 | 216 | 497 | 836 | 671 | 41 | 5,558 | 3 | 3 | 4 | 0 | 7 |

**What the reading must show on CRVC1's objects:**
- each form's count equals the row above exactly (the compression runs after register allocation, which the
  feature does not change);
- no 16-bit FP form;
- 4-byte fld/fsd equal to CRVC0's 4-byte count plus its four 16-bit FP columns;
- every image's e_flags carry `EF_CAPSTONE_CRVC`, and its magic is the CRVC one.

The method's control: on the two probe programs, prediction and reading agree in all eight classes (55 = 55), and
the five 16-bit FP forms become 4-byte (1 + 5 = 6). A shortfall on SPEC is a compiler defect to find, not variance.
The dynamic prediction for step 4 is registered separately, once CRVC0's profiles exist.

## Step 3 reading (2026-11-21): six of the eight forms as predicted; c.ldc and c.stc fewer, by exactly the relocated sites

CRVC1 was built with the same compiler (clang 518578ef40a76b19, lld b9f671e793ab869d) and read on its compiled
objects.
- **As predicted:** c.ldcsp, c.stcsp, c.cincoffset16sp, c.movc, c.scc and c.cincoffset match in every benchmark;
  no 16-bit FP form remains; 4-byte fld/fsd rose by exactly CRVC0's 16-bit FP count in every benchmark.
- **Images:** all eleven carry flags 0x101 (`EF_CAPSTONE_CRVC` with RVC) and the CRVC magic. Text shrank 9.4 % on
  perl and 10.8 % on gcc (the census said 9.71 % and 10.67 %).
- **Not as predicted:** c.ldc and c.stc came in below the prediction.

| benchmark | c.ldc predicted / got | c.stc predicted / got | fitting sites with a relocation (ldc / stc) |
|---|---:|---:|---:|
| perl | 18,094 / 15,365 | 6,145 / 5,112 | 2,729 / 1,033 |
| gcc | 77,962 / 67,757 | 28,118 / 27,730 | 10,205 / 388 |
| x264 | 5,709 / 5,547 | — | 162 / 0 |
| mcf | 82 / 80 | 94 / 93 | 2 / 1 |
| imagick | 9,402 / 9,319 | 5,042 / 5,023 | 83 / 19 |
| nab | 1,073 / 815 | 464 / 462 | 258 / 2 |
| xz | 447 / 369 | — | 78 / 0 |
| lbm | as predicted | as predicted | 0 / 0 |

**CORRECTION of the pre-registered prediction.** The shortfall equals, in every benchmark and both columns, the
fitting sites whose offset carries a relocation.
- These are `-capstone-fold-global-lo` sites: a global's `%pcrel_lo` folded into the load or store. In an object
  the offset reads as 0, so the prediction counted it as fitting; the compiler cannot compress an offset that is
  unknown until link time.
- The compiler compressed every compressible site. The prediction over-counted.
- **RETRACTED the same day: "the same applies to the census".** The census reads linked images, where most folded
  offsets resolve to values that do not fit (a nab example: -0x37c), so it never counted them.
  - The folded sites that do fit on the images are 49 c.ldc and 17 c.stc on perl, and 244 and 51 on gcc (fold.py:
    auipc, then scc gp, then the access, following copies and spills; on objects 91-100 % recall, over 99.5 %
    precision).
  - Without them the census set reads 9.77 % and 15.35 %, unchanged. The retraction is on dev at b33a7a87d6f6.

Zcb byte and halfword forms with a capability base (c.lbu, c.lh, c.lhu, c.sb, c.sh) are a separate flag and a
separate arm, measured alone. P1's RTL decodes Zcb: `compressed_decoder.sv` at d83d3a633 (`OpcodeC0Zcb` under
`CVA6Cfg.RVZCB`), with `CVA6ConfigZcbExtEn = 1` in `capstone_cv64a6_imafdc_sv39_config_pkg.sv`.
