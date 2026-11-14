# Compressed capability ISA, stage 2: the 16-bit decoder space (RTL lane sketch, 2026-11-14)

*A sketch for the compiler lane and the lead, not a design decision. Requested by the runtime lane, relaying the lead:
"ISA stage 2 last; decoder RTL together with the compiler lane; only sketch the decoder space for now." Nothing here is
built. The choice of encodings is the lead's, from the compiler lane's stage-2 table.*

## What it builds on

- **Stage 1** is compiler-only: `+cap-base-rvc` emits the standard RVC loads and stores with a capability base. The
  hardware expands C.LD to the same LD, which already accesses through a capability base (compiler lane,
  `capstone/docs/history/14-11-2026_00-45-00_rvc-for-virtual-images-and-capability-base-compression.md`, branch
  compiler-capbase-rvc-c8).
- **The shapes stage 2 would compress**, dynamic share of image instructions, from that record's "Stage 2" table:

| Shape | gcc | perl makerand |
|---|---|---|
| ldc / stc sp-relative (C.LQSP-like, uimm 0..1008 / 16) | 7.82 % / 7.73 % | 6.24 % / 6.30 % |
| movc rd, rs (C.MV-like) | 4.56 % | 1.30 % |
| ldc rc, off(rc) (C.LQ-like, uimm 0..496 / 16) | 3.16 % | 5.32 % |
| cincoffsetimm sp, sp (C.ADDI16SP-like) | 2.32 % | 3.55 % |
| scc rd, gp, rs | 1.79 % | 7.64 % |
| cincoffset rd, rd, rs (C.ADD-like) | 0.89 % | 0.73 % |
| cincoffsetimm rd, rd, imm6 (C.ADDI-like) | 0.72 % | 0.18 % |

- **The RTL (capstone-ariane, read at a948e4134 / daa4451ff).** `core/compressed_decoder.sv` expands each 16-bit
  instruction into a 32-bit one and carries no capability logic. Capstone's own ops sit in custom-2/custom-3
  (bits [1:0] = 11), so they have no 16-bit form today. The board configuration enables C, D and **Zcb**
  (`CVA6ConfigZcbExtEn = 1`, config package :64); Zcmp is off (:65).

## The free 16-bit space on this core (RV64GC + Zcb)

| Space | Payload | Status |
|---|---|---|
| Q0 001 C.FLD, Q0 101 C.FSD | 3 + 3 + 5 bits | taken by D; RV128 puts C.LQ / C.SQ here (CHERI-RISC-V's capability load/store choice) |
| Q2 001 C.FLDSP, Q2 101 C.FSDSP | 5 + 6 bits | taken by D; RV128 puts C.LQSP / C.SQSP here |
| Q0 100, [12:10] = 100..111 | 4 slots x 8 bits (rs1' 3, [6:5] 2, rd'/rs2' 3) | reserved (Zcb uses 000..011: c.lbu, c.lh/lhu, c.sb, c.sh; `compressed_decoder.sv` illegal default) |
| Q1 100 111 1 11, [4:2] = 110, 111 | 2 slots x 3 bits (rd') | reserved (Zcb's unary group uses 000..101) |
| C.LUI rd = x0 (Q1 011) | 6-bit imm | HINT space |
| C.LI / C.ADDI / C.SLLI rd = x0 | 6-bit imm each | HINT space |
| C.MV / C.ADD rd = x0 | 5-bit rs2 each | HINT space |
| C.LWSP / C.LDSP rd = x0, C.ADDI4SPN imm = 0, C.ADDI16SP imm = 0, C.JR x0 | 6 / 6 / 3 / 0 / 0 bits | reserved |

The binding fact is that **every two-full-register form needs 10-11 payload bits, and the only such space is the four D
slots.** Everything else is 3-bit registers (x8-x15) or a single operand.

## One allocation that fits (option A: give the four compressed FP-double slots to capabilities)

| New op | Encoding | Covers |
|---|---|---|
| c.ldcsp rd, uimm(sp) | Q2 001 (C.FLDSP / C.LQSP layout, uimm[9:4]) | shape 1 load, the largest |
| c.stcsp rs2, uimm(sp) | Q2 101 (C.FSDSP / C.SQSP layout) | shape 1 store |
| c.ldc rd', uimm(rs1') | Q0 001 (C.FLD / C.LQ layout, uimm[8:4]) | shape 3 load |
| c.stc rs2', uimm(rs1') | Q0 101 (C.FSD / C.SQ layout) | shape 3 store |
| c.cincoffset16sp nzimm | C.LUI rd = x0 (HINT), C.ADDI16SP's immediate layout | shape 4 |
| c.movc rd', rs' | Q0 100 100 | shape 2, x8-x15 only |
| c.scc rd', gp, rs' | Q0 100 101 | shape 5, x8-x15 only |
| c.cincoffset rd', rs' | Q0 100 110 | shape 6, x8-x15 only |
| (spare) | Q0 100 111, two Q1 unary slots | |

- **The cost:** compressed fld/fsd/fldsp/fsdsp disappear for every program on the bitstream (the 4-byte forms stay).
  That costs FP-heavy code size and fetch. A census of executed c.fld/c.fsd/c.fldsp/c.fsdsp on the FP SPEC rows is
  needed before choosing; it is the compiler lane's measurement.
- **Option B, a decode mode.** Keep D's meaning by default and reinterpret the four slots only when a per-image mode bit
  is set (CHERI's capability-mode pattern). A decoder input driven by a CSR needs a flushing CSR write. The twin and the
  compiler must agree on the bit, and images mix only at a domain switch. This is more RTL and more ABI, and no FP cost.
- **The 3-bit register forms are an open question.** Whether c.movc / c.scc / c.cincoffset limited to x8-x15 capture
  shapes 2, 5 and 6 depends on register allocation; it needs the census by register class (compiler lane). The
  HINT-space reuse for c.cincoffset16sp conflicts with any software that emits C.LUI x0 as a hint (none known).

## RTL constraints found while sketching (by reading, capstone-ariane a948e4134)

- **The hard-coded pc + 4 sites must stay behind 32-bit-only instructions.** The frontend resumes a commit flush at
  `pc_commit + 4` unconditionally (`core/frontend/frontend.sv:449`; its own comment says this is safe only because CSR and
  AMO have no compressed form). Capstone adds `flush_commit` for REVOKE, DROP, CREVOKE and CSRETIRE
  (`core/commit_stage.sv:893-897`), and the CSRUNV switch request uses `pc + 4` (`:689`, `:985`). **None of the stage-2
  shapes is in that set.** The R-43 replay re-fetches the SAME pc (`:902-912`), so a 16-bit ldc/stc replays correctly.
  If any of REVOKE / DROP / CREVOKE / CSRETIRE / CSRUNV / CALL ever gains a 16-bit form, those sites need the instruction
  length first.
- **The expansion is combinational and confined to `compressed_decoder.sv`:** each new op expands to the existing
  32-bit Capstone encoding. No datapath changes. The synthesis risk is small but not zero: the decoder sits on the
  issue path (gate as usual: lint, sweep, a synthesis-only size reading).
- **The twin (capstone-qemu) needs the same decode**, and the stage-1 simulation arm (perf/rvc-capmode: 2-mod-4
  targets, traps and CSRUNV resumes) is the template for the stage-2 acceptance arm: every new op at a 2-mod-4 pc, under
  a trap, under an R-43 replay, and across a CSRUNV resume.

## Next

The compiler lane's census of the FP compressed forms (option A's cost) and of register classes for shapes 2, 5 and 6;
then the lead picks A or B and the encodings. The decoder RTL and the twin follow in one change, after this bitstream.
