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

Zcb byte and halfword forms with a capability base (c.lbu, c.lh, c.lhu, c.sb, c.sh) are a separate flag and a
separate arm, measured alone. P1's RTL decodes Zcb: `compressed_decoder.sv` at d83d3a633 (`OpcodeC0Zcb` under
`CVA6Cfg.RVZCB`), with `CVA6ConfigZcbExtEn = 1` in `capstone_cv64a6_imafdc_sv39_config_pkg.sv`.
