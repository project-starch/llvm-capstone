# Stage 2 on the trap route: the decoder delta against B5b-2 (a spec for rtl, 2026-11-25)

*Compiler lane, at the runtime lane's request. A SPEC only: nothing here is built, and whether stage 2 enters any
bitstream is the lead's decision (Line B plan, 2026-11-25-line-b-final-campaign.md: "Stage 2 is out of the final
bitstream by default"). Two halves:*
- *(a) how the compressed-capability decode mode (CRVC) is selected on the trap route, where its prototype's
  mechanism no longer exists;*
- *(b) auipcgp's custom-1 decode.*

*The base is capstone-ariane 904cd9c6b, B5b-2's decoder. The encodings are the prototypes':*
- *CRVC: plans/2026-11-21-capability-rvc-stage2-prototype.md, compiler `+cap-crvc`;*
- *auipcgp: design/virtual-capstone/isa.md, "Prototype: auipcgp", compiler `+cap-auipcgp`.*

## (a) Selecting CRVC on the trap route

**Why the prototype's mechanism is gone.** The prototype keeps the mode in bit 2 of CSRUNV's admission options word.
B5b-2 makes CSRUNV illegal (step 8a), and a trap-route program is an ordinary Linux task (the .dom binfmt loader). The
mode therefore has to belong to the task's address space, the way the lifetime table does.

### The CSR

- **`scapmode`, CSR 0x5C3**, in the S custom read/write range next to `srevroot` (0x5C1).
  - It is free at 904cd9c6b (riscv_pkg.sv has only CSR_SREVROOT = 12'h5C1 in that range), in the reference model
    (cpu_bits.h: 0x5C0 scapctl, 0x5C1 srevroot, 0x5C2 stagtest, test-only) and in the trap kernel.
- **Bit 0 = CRVC.** Bits 63:1 are WARL, reading 0. Reset is 0.
- **Access:** read-write in S and M; U has no access (illegal instruction), as for srevroot.
- **Why not a bit of srevroot.** srevroot's low 12 bits are forced to zero (csr_regfile.sv:2076,
  `{csr_wdata[63:12], 12'b0}`), so they look free, but two things break:
  - "srevroot != 0" is what defines protected U and S rule 1 (csr_regfile.sv:3009-3017, cap_data_ctx_o);
  - the kernel's select_root/restore_root (sublet-table.c:125-140) compare and rewrite srevroot by value inside S.
  A separate CSR leaves both untouched.

### What the bit does

The decode mode is effective exactly when `scapmode.crvc == 1 && priv == U && srevroot != 0`.
- That is protected U with the bit set.
- In S and M, and in U with srevroot = 0 (an ordinary mm), decode is standard RVC whatever the bit says.
- So a stale bit cannot change the kernel's or the firmware's decode, nor an ordinary process's if switch_mm failed
  to clear it.

In the mode, the eight 16-bit slots of the prototype's table expand to their 32-bit Capstone instructions:
- c.ldcsp at Q2 001, c.stcsp at Q2 101, c.ldc at Q0 001, c.stc at Q0 101;
- c.cincoffset16sp at Q1 011 with rd = x0;
- c.movc, c.scc and c.cincoffset at Q0 100 [12:10] = 100, 101, 110.
Q0 100 111 and the two Q1 unary slots stay reserved, and `c.cincoffset16sp` with nzimm = 0 stays illegal. Outside
the mode the slots keep their RV64GC+Zcb meaning.

### Flush and ordering

- **A write of scapmode must set `flush_o`**, as srevroot and csupctl writes do since bf3db3175 ("decoder flags need
  a flushing CSR write": without it, the instruction after the write decodes under the stale mode).
- In practice the kernel writes scapmode only in S, and the mode only matters in U, so the S-to-U transition, sret,
  is the real boundary. sret is a pipeline flush on CVA6, and so is the trap into S. No instruction fetched in one
  mode is decoded in the other.
- **The check rtl should write:** a directed test with the first compressed instruction IMMEDIATELY after the sret
  that enters U with the bit set, and one immediately after a csrw scapmode in S that then srets. A gap between the
  write and the use hides the missing flush.

### Kernel (trap-route Linux series)

- **The loader** (arch/riscv/kernel/sublet/binfmt.h, which checks the ELF header at :68 and :335):
  - read `e_flags & EF_CAPSTONE_CRVC` (0x0100, design/capability-calling-convention.md) and record it in the mm
    context beside `capstone_root`;
  - REFUSE a flagged image if the hart lacks the mode. Probe at boot: write 1 to scapmode and read it back. A hart
    without it reads 0, or traps illegal at the CSR access, which a boot-time fixup catches.
  - An image decoded in standard mode would run c.ldcsp as c.fldsp, a silent wrong instruction rather than a fault.
- **switch_mm** (mm/context.c:317): write `csr_write(0x5c3, next->context.capstone_crvc)` beside the srevroot write,
  0 for an ordinary mm. Mirror SUBLET_MUTANT_LEAK_ROOT with a mutant that leaves the bit set for an ordinary mm. The
  gating by srevroot != 0 should make it harmless, and that is the check.
- **Threads** share the mm and so the bit. **execve** of an unflagged image clears it with the new mm.
- **Nothing in S decodes user instructions** on this kernel: traps_misaligned.c is built only for M-mode kernels
  (kernel/Makefile:56).

### Firmware (OpenSBI, keep-tags tree)

- `lib/sbi/sbi_misaligned_ldst.c` emulates misaligned loads and stores for S and U by decoding the trapping
  instruction, compressed forms included. In the mode, Q0 001 is c.ldc, not C.FLD; emulating it as an 8-byte FP load
  would be silent and wrong.
- The emulator must read scapmode (M can) together with mstatus.MPP and srevroot. If the trapping instruction was
  decoded in the mode, it does not emulate: it redirects the trap to S unchanged.
- A capability load or store is never legitimately misaligned, so redirecting costs nothing.

### Reference model (QEMU, the trap twin's tree, 3a9fc39167 or later)

- CSR 0x5C3 `scapmode`, with the predicates above. Its write ends the TB, as srevroot's does.
- The decode mode is a TB flag (riscv `cpu_get_tb_cpu_state`): `crvc = scapmode.crvc && priv == U && srevroot != 0`.
  A TB translated in one mode is never reused in the other.
- The decoder change is the prototype's (decode16 under the flag, the eight slots), keyed on the TB flag instead of
  `virtual_options` bit 2. The prototype's `x-capstone-crvc` property becomes the property that makes the CSR
  exist; without it, 0x5C3 is illegal, as on B5b-2.

### RTL change list (a)

1. csr_regfile.sv:
   - the scapmode register, reset 0;
   - read and write in S/M, illegal from U;
   - `flush_o = 1'b1` on its write;
   - an output `crvc_o = CAPSTONE_EXT && scapmode_q[0] && priv_lvl_q == U && srevroot_q != 0`.
2. cva6.sv / id_stage.sv: route crvc_o to every compressed_decoder instance (id_stage.sv:167-176), as vctx_eff
   reaches decoder.sv.
3. compressed_decoder.sv: an input `crvc_i`. Under it, the eight slots expand to the 32-bit forms of ldc, stc,
   cincoffsetimm, movc, scc and cincoffset; without it, as today. The expansion feeds the existing decoder, so the
   datapath and the FLU are unchanged.
4. The gates: the lint gate, and a synthesis-only run. The compressed decoder sits in the issue path once per issue
   port (CLAUDE.md: a wide combinational change on that path gets synthesis before a full build).

## (b) auipcgp: the custom-1 decode

**Definition** (isa.md): `auipcgp rd, imm20` is `scc rd, x3, (pc + (imm20 << 12))`, with pc read as auipc reads it.
- The result's authority is gp's alone; the PCC contributes an integer address and nothing else.
- **Encoding:** U-type in custom-1, opcode 0x2b: imm[31:12], rd[11:7], opcode[6:0]. It takes the whole major
  opcode.
- **At 904cd9c6b** decoder.sv has no custom-1 arm (it handles custom-0 at :506, custom-3 at :1172 and custom-2 at
  :1243), so 0x2b falls to the illegal default. The opcode is free.

### Where it is legal

The same modes as SCC, since it is SCC with a computed rs2. On B5b-2 that means:
- protected U (the program's own gp);
- S and M, where the decoder accepts SCC.
The compiler emits it only under the image-gp ABI, in application code. No kernel or firmware code uses it.

### Interaction with the untagged rule

auipcgp inherits every SCC rule unchanged (isa.md's authority rule):
- an UNINIT or sealed gp raises 26;
- a LINEAR gp is moved, leaving x3 null;
- a cursor outside gp's bounds is a window cursor, checked at the access.

The untagged-gp case follows the FLU rule of the bitstream it runs on:
- **904cd9c6b and 88db40027:** SCC's integer rule applies only under vctx (capstone_flu_unit.anvil:109), so in
  protected U an untagged gp raises cause 24, as for SCC.
- **28fcbb646** (the rule made reference-exact, the vctx gate dropped): an untagged gp yields the untagged integer
  pc + (imm20 << 12). The reference model does this today.
- Under the image-gp ABI, gp is the image's tagged data capability for the whole run. An untagged gp is a corrupted
  gp, so this choice changes the failure point, not correct programs. The untagged-op count on the B5b-2 SPEC images
  (tests/b5b2-untagged-count) counts SCC; auipcgp would add to the same op class.

### Kernel

- No decode is needed: the S kernel never emulates it.
- auipcgp is not a load or store, so the misaligned emulator never sees it.
- **The loader should refuse an image that carries it on a hart without it**, for the same reason as CRVC. Here the
  failure is an illegal-instruction fault, not silent, but it is late.
  - The prototype deliberately has no ELF flag ("unlike CRVC it needs no context mode bit, image magic or ELF flag").
  - For silicon, an e_flags bit `EF_CAPSTONE_AUIPCGP` (proposed 0x0200), set by the compiler under `+cap-auipcgp`,
    plus a boot probe, would let the loader refuse up front. That is the lead's call; without it, the fault is
    deterministic at the first execution.

### Reference model

capstone-qemu `compiler/auipcgp-qemu` 72241b6ec6 (on a0501ebc39) implements it under `x-capstone-auipcgp`, sharing
`capstone_scc_cursor` with `helper_csscc` (rs1 = 3). Port it onto the trap twin's tree. Its untagged case is the
reference rule (an integer), as for SCC.

### RTL change list (b)

1. decoder.sv: an `riscv::OpcodeCustom1` arm under a config parameter (CAPSTONE_AUIPCGP), else illegal as today.
   - It decodes the U-type immediate, forces rs1 = x3 (gp is read as a capability, so the scoreboard tracks x3),
     sets rd, and issues to the FLU as a new op AUIPCGP.
   - The decode must hand the FLU pc + (imm20 << 12) as the integer operand. Either the issue stage forms it as it
     does for AUIPC (operand_a = pc, operand_b = imm), or the FLU adds pc and imm itself.
2. capstone_flu_unit.anvil: `AUIPCGP(data)` is SCC with that integer as rs2. It is the same arms as SCC (func SCC at :108-:149),
   including the untagged arm, which follows whatever the bitstream's SCC does (gated on 904cd9c6b, reference-exact on
   28fcbb646). Simplest is to call SCC's body with the substituted operand.
3. The gates: lint, then a directed test per arm — tagged gp, an untagged gp under each FLU rule, an UNINIT/sealed gp
   (26), and the result's authority equal to gp's, not the PCC's. auipcgp adds a decode arm and an adder input, not a
   wide function, so the synthesis-only run can be batched with (a)'s.

## What this does not decide

- Whether either half enters a bitstream (the lead).
- Option A against option B for CRVC (a one-bit difference in how the mode is selected; the encodings are the
  same).
- The EF_CAPSTONE_AUIPCGP flag.

## Line B re-check and the step-4 plan (2026-11-26)

*At the runtime lane's request. The project lead has said to try stage 2 on Line B, and it has to be shown
experimentally to pay. Nothing is built, and no synthesis is requested. The choice between option A and option B
stays with the lead. This section proceeds on B.*

### (a) The mode's home on Line B: the spec above holds at b42c8c644 and on the final Full kernel

- **CSR 0x5C3 is free at b42c8c644.** riscv_pkg.sv:669 defines only CSR_SREVROOT = 12'h5C1 in that range, and nothing
  in core/ names 0x5C3, scapmode or crvc.
- **The kernel writes the spec names are where it says,** in the final Full kernel tree (the trap series, boot 13):
  - switch_mm writes srevroot from the mm context at mm/context.c:317-321;
  - the loader checks the ELF header at sublet/binfmt.h:66 and :335.
- **One addition, the return-to-U guard** (entry.S:407-425). It already compares srevroot with
  current->mm's capstone_root and kills the task on a mismatch. It should compare scapmode with the mm's capstone_crvc
  in the same way, so that a stale or missing bit never reaches U. The existing mutant switch (SUBLET_MUTANT_NO_GUARD)
  then covers it too.
- **The per-process state is per mm,** derived at switch_mm, not saved: threads share it, and execve replaces it.
- **The kernel side is in the collaborator's trap series:** context.c, entry.S and binfmt.h. Those changes go through
  the series' owner, not into it directly.
- **Homes considered and rejected:**
  - **srevroot's low 12 bits:** the reasons above. Protected U is defined by srevroot != 0, and the kernel compares
    and rewrites srevroot by value.
  - **sscratch:** on this kernel it holds the user tp with its tag across U and is swapped at every trap entry and
    exit (entry.S:70, 137-207). It has no spare bits. A decode input there would ride a register the entry code
    rewrites on every trap.
- **Option A,** the mode always on in protected U, would be the same RTL with `crvc_o = priv == U && srevroot != 0`
  and no CSR. Its cost is that every protected image is in the mode. On P1's twin, the FP-heavy rows lose fetch bytes
  in the mode: nab -3.53 % and lbm -2.98 %, with imagick +0.37 %. Under B those images stay out of it.

### (b) Step 4 on the Line B footing

**The compiler.** B5TL1's compiler (llvm-c9, 29027126f39b) has neither +cap-crvc nor +cap-base-zcb. The lane compiler
(llvm-sinkland) has both, so every arm is built by it, with B5TL1's flags.
- The off arm is LSHOFF, already counted on all eight stages (tests/spec8-lsrh). LSRH is not in these arms, so stage
  2 is measured alone.
- **The arms:**
  - **CRVC:** +cap-crvc. All eight stages are built and read, so the per-stage choice option B allows can be
    made from the readings.
  - **ZCB:** +zcb +cap-base-zcb.
  - **CRVCZ:** both together. P1's sum of the two (x264 about 3.4 %) was never measured.

**The twin.** qemu-crv 3a9fc39167 has no CRVC decode. The prototype's decode (capstone-qemu 05706b2e11,
compiler/stage2-crvc-qemu) is keyed on CSRUNV's virtual_options.
1. Port it onto the trap twin's tree as the reference-model section above specifies: CSR 0x5C3 under x-capstone-crvc,
   with the mode as a TB flag, `scapmode.crvc && priv == U && srevroot != 0`.
2. For the fetch-byte measurement, add a twin-only property that also sets the mode in every protected U, with no
   CSR write. A CRVC image then boots on the unmodified trap kernel.
   - Fetch bytes do not depend on how the mode is selected.
   - Only CRVC-built images boot with the property, and the off arm boots without it.
   - A CRVC image booted without the property must fail: it would run c.ldcsp as c.fldsp. That failure is the
     positive control that the mode is reached.
3. The kernel's loader, switch_mm and guard changes get their own twin check before any board:
   - a flagged image runs in the mode;
   - the leak mutant, which leaves the bit set for an ordinary mm, trips the guard;
   - with the guard also disabled (SUBLET_MUTANT_NO_GUARD), that ordinary process still decodes standard RVC,
     because the mode is gated on srevroot != 0. That is the hardware's half of the check.
4. The Zcb arms need zcb=true on the twin's CPU string, since qemu-crv defaults it off (target/riscv/cpu.c:2088).

**The reading.** Fetch bytes per stage at shift 1 (capmix3), against LSHOFF, on the eight-stage init of
tests/spec8-lsrh, together with instruction counts and the outputs. This gets its own pre-registration, with P1's
readings as the prior. P1 read, against CRVC0:

| | perl | gcc (train01) | x264 | mcf | xz | imagick | nab | lbm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CRVC fetch bytes saved | 8.47 % | 14.88 % | 2.16 % | 6.61 % | 1.86 % | 0.37 % | -3.53 % | -2.98 % |
| ZCB fetch bytes saved | 0.889 % | 0.593 % | 1.289 % | 0.081 % | 1.259 % | 0.982 % | 0.220 % | 0.000 % |

Line B's gcc stage is the t1.c test, not train01, so gcc is not predicted from P1.

### (c) Zcb may need no new seal

**b42c8c644's configuration enables Zcb:** capstone_cv64a6_imafdc_sv39_config_pkg.sv:77 `CVA6ConfigZcbExtEn = 1`,
and :156 `RVZCB`. If rtl confirms that the compressed decoder decodes it on that seal, and that the FPGA build uses
this configuration, then the ZCB pair (LSHOFF against ZCB images) can run on b42c8c644 itself.
- It needs no RTL and no synthesis, and it can run before any stage-2 seal exists.
- It follows ZCB9 on P1 silicon, which read perl -0.33 %, and xz and imagick flat.

### (d) The cycle band, pre-registered after the step-4 readings and before any RTL

**Calibration:** ZCB9 on P1 silicon, N=1, time saved per fetch byte saved:
- perl 0.33 / 0.889 = 0.37;
- xz 0.06 / 1.259 = 0.05;
- imagick flat (+0.26 % against 0.98 %, inside its band).

**The band:** time saved = fetch bytes saved × [0, 0.37], per stage, from the Line B step-4 fetch readings. It is
written as a band, not a point, because the ratio is N=1 and spans nearly an order of magnitude across rows.
Converted at P1's CRVC fetch figures, it would be perl 0 to -3.1 %, for scale only.

**Go/no-go for a stage-2 seal:**
- a time saving of at least 1 % on perlbench makerand (275.70 s on b42c8c644) or gcc t1.c (5.55 s, where 1 % is
  about five ticks of the 0.01 s timer);
- and no stage worse than its in-session spread + 0.05 % + the 0.02 s quantum, the LSRH board form's rule.

**The board:**
- the new seal is b42c8c644 + the decoder + the mode;
- the pair is the CRVC (or CRVCZ) image against LSHOFF on that seal;
- the control is LSHOFF on the new seal against the same image on b42c8c644, which reads what the seal itself costs.

The new seal never replaces the campaign seal unless it reads as above AND the lead agrees. The decoder change sits
in the issue path, so a synthesis-only run comes before any full build.
