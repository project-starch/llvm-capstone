/* capstone_csmint.S -- vp2: the monitor's emulation of an S-mode CSMINT (board lane, 2026-10-17).
 *
 * On vp2 an S-mode CSMINT raises an illegal instruction (cause 2) and the monitor emulates it. The decode is from mtval,
 * which carries the instruction since capstone-ariane virtual-profile-2-d bf3db3175 makes an unknown custom encoding
 * illegal at decode; with mtval 0 (an older decoder) the instruction is fetched at mepc. The semantics and the order of
 * every refusal are QEMU's helper_csmint (capstone-qemu target/riscv/capstone_table.c:218-283 at d35af47b163c). The
 * emulation itself was validated end to end in RTL simulation as va-csmint-emu (capstone/tests/rtl-smoke/
 * vp2-kit-2026-10-17/csmint-emu, bf3db3175, 31 of 32 readings as pre-registered and the 32nd a wrong prediction); this
 * file is that routine shaped for the monitor, and the RTL arm va-csmint-mon assembles THIS file, not a copy.
 *
 *   csmint_dispatch(a0 = a capability over the trap frame, cursor at its base: S's x[n] at a0 + 8n)
 *     -> a0 = 0: not a CSMINT, nothing changed; the caller carries on with its own handling.
 *     -> a0 = 1: handled. Either rd was written into the frame and mepc advanced by 4, or the refusal was injected
 *        into S (scause, stval, sepc = mepc, sstatus.SPP/SPIE/SIE, mepc = stvec). The caller returns to S through its
 *        normal path, which restores the GPRs from the frame. ra, sp, gp and tp are preserved; every other GPR is
 *        clobbered (the caller restores them all from the frame).
 *
 * No MPRV and no access through S's translation from M: a page fault inside the trap handler would re-enter a trap
 * entry that is not reentrant. Every S address (the instruction at mepc, the descriptor's three words, the slot) is
 * walked in software (Sv39, satp), and every physical access goes through a NONLIN capability minted for that one page
 * with CAPCREATE/CAPTYPE/CAPBOUND/CAPPERM (legal from M; illegal only under supervision), so the routine never touches
 * the monitor's region table.
 *
 * The walk never sets A or D (vp2's walker faults on a clear A/D instead of updating them): a load needs V, A, and R or
 * (X and mstatus.MXR); a store needs V, R, W, A and D; a U page needs mstatus.SUM; a superpage must be aligned.
 *
 * Refusals, in QEMU's order, each injected into S before any write:
 *   an operand that is sp (x2) -> 2, tval the instruction (the monitor keeps S's sp in CSCRATCH, not in the frame);
 *   desc & 7 -> 4 (tval desc); a descriptor word not readable -> 13 (tval its VA); slot & 15 -> 6 (tval slot);
 *   the slot not writable -> 15 (tval slot); base >= end or perms > 15 -> 29; [base, end) not representable -> 29
 *   (a length of 4 KiB and up needs a granule-aligned base and a granule-multiple length, granule 2^(msb(len)-9):
 *   capstone_repr_ok); fewer than two free identities in the table at srevroot (capacity - next + free_count, from the
 *   header's words 0-2 as QEMU reads them, cap_rev_table.c:77-126) -> 30.
 * Then: mint the LINEAR capability; the csupctl.table window is EXACTLY csrsi csupctl, 8; csnode; csrci csupctl, 8 (the
 * RTL lane's rule); STC through a physical capability over the slot's page; rd = the ancestor id; mepc + 4.
 *
 * CSNODE's own refusals (30 below two free identities, 29 a corrupt table) are a DEFENCE behind the pre-check: for the
 * window, mtvec points at csm_window_trap, so a CSNODE fault lands there instead of in the monitor's trap entry (which
 * is not reentrant), and its cause is injected into S. The path is negative-tested by putting the table into a state
 * the pre-check passes and CSNODE refuses.
 */

#define CSM_CSNODE(rd, rs1)          .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSM_CAPCREATE(reg)           .insn r 0x7B, 0x0, 0x4, reg, x0, x0
#define CSM_CAPTYPE(rd, rs1)         .insn r 0x7B, 0x0, 0x5, rd, rs1, x0
#define CSM_CAPPERM(rd, rs1)         .insn r 0x7B, 0x0, 0x7, rd, rs1, x0
#define CSM_CAPBOUND(rd, rs1, rs2)   .insn r 0x7B, 0x0, 0x8, rd, rs1, rs2
#define CSM_CINCOFFSET(rd, rs1, rs2) .insn r 0x5B, 0x1, 0xc, rd, rs1, rs2
#define CSM_MOVC(rd, rs1)            .insn r 0x5B, 0x1, 0xa, rd, rs1, x0
#define CSM_LCC(rd, rs1, sel)        .insn r 0x5B, 0x1, 0x4, rd, rs1, sel
#define CSM_STC(data, addr, imm)     .insn s 0x5B, 0x4, data, imm(addr)
#define CSM_CCSRRW(rd, ccsr, rs1)    .insn i 0x5B, 0x7, rd, ccsr(rs1)
#define CSM_CCSR_CTVEC   0x0
#define CSM_CSUPCTL      0x7C4
#define CSM_SREVROOT     0x5C1
#define CSM_TYPE_LIN     1           /* CAPTYPE's input numbering */
#define CSM_TYPE_NONLIN  2
#define CSM_PERM_RW      6

/* CAP = a NONLIN RW capability over the 4 KiB page holding the physical address in PA, cursor at PA.
 * T1 and T2 are temporaries; all four registers must be distinct. */
#define CSM_PHYS(CAP, PA, T1, T2) \
  CSM_CAPCREATE(CAP); li T1, CSM_TYPE_NONLIN; CSM_CAPTYPE(CAP, T1); \
  srli T1, PA, 12; slli T1, T1, 12; li T2, 4096; add T2, T1, T2; CSM_CAPBOUND(CAP, T1, T2); \
  li T2, CSM_PERM_RW; CSM_CAPPERM(CAP, T2); sub T1, PA, T1; CSM_CINCOFFSET(CAP, CAP, T1)

/* Software Sv39 walk of the S virtual address in VA (unchanged) for one access; ISSTORE is 0 or 1.
 * Out: PA = the physical address, or a jump to FAIL. U is a unique label prefix.
 * Temporaries: t3 (the table's PA), t4 (the level's shift), t5, t6 (the PTE's capability, then the PTE), a7 (level),
 * a6. VA and PA must not be any of these. */
#define CSM_WALK(VA, PA, ISSTORE, FAIL, U) \
  csrr t3, satp; slli t3, t3, 20; srli t3, t3, 20; slli t3, t3, 12; li a7, 2; \
U##_lvl: \
  li t4, 9; mul t4, t4, a7; addi t4, t4, 12; srl t5, VA, t4; andi t5, t5, 511; slli t5, t5, 3; add t5, t5, t3; \
  CSM_PHYS(t6, t5, PA, a6); ld t6, 0(t6); \
  andi t5, t6, 1; beqz t5, FAIL; \
  andi t5, t6, 0xA; bnez t5, U##_leaf; \
  srli t3, t6, 10; slli t3, t3, 20; srli t3, t3, 20; slli t3, t3, 12; addi a7, a7, -1; bltz a7, FAIL; j U##_lvl; \
U##_leaf: \
  andi t5, t6, 0x40; beqz t5, FAIL; \
  .if ISSTORE; andi t5, t6, 0x86; li a6, 0x86; bne t5, a6, FAIL; \
  .else; andi t5, t6, 0x2; bnez t5, U##_rok; andi t5, t6, 0x8; beqz t5, FAIL; \
  csrr t5, mstatus; srli t5, t5, 19; andi t5, t5, 1; beqz t5, FAIL; \
U##_rok: ; .endif; \
  andi t5, t6, 0x10; beqz t5, U##_uok; csrr t5, mstatus; srli t5, t5, 18; andi t5, t5, 1; beqz t5, FAIL; \
U##_uok: \
  li a6, 1; sll a6, a6, t4; addi a6, a6, -1; srli t5, t6, 10; slli t5, t5, 20; srli t5, t5, 20; slli t5, t5, 12; \
  and t3, t5, a6; bnez t3, FAIL; and t3, VA, a6; or PA, t5, t3

/* No .section here: the includer places this file. In the monitor it must sit inside .cap_text, the PCC bounds
 * CAPENTER sets; in the RTL arm, inside the arm's text. */
.align 2
.globl csmint_dispatch
csmint_dispatch:
  csrr t0, mtval
  bnez t0, 1f
  /* mtval 0: fetch the instruction at mepc, walked as a load (R, or X with MXR) */
  csrr a1, mepc
  CSM_WALK(a1, a2, 0, csm_not_ours, csm_wi)
  CSM_PHYS(t6, a2, t3, t4)
  lwu t0, 0(t6)
1:
  li t1, 0xfe00707f
  and t2, t0, t1
  li t1, 0xa000105b
  bne t2, t1, csm_not_ours
  mv s2, t0                    /* s2: the instruction */
  srli s3, t0, 15
  andi s3, s3, 31              /* s3: rs1, the slot */
  srli s4, t0, 20
  andi s4, s4, 31              /* s4: rs2, the descriptor */
  srli s5, t0, 7
  andi s5, s5, 31              /* s5: rd */
  li t1, 2
  beq s3, t1, csm_bad_operand
  beq s4, t1, csm_bad_operand
  beq s5, t1, csm_bad_operand
  /* operands from the frame; x0 reads 0 */
  li a4, 0
  beqz s3, 2f
  slli t1, s3, 3
  CSM_CINCOFFSET(a0, a0, t1)
  ld a4, 0(a0)                 /* a4: the slot VA */
  sub t1, x0, t1
  CSM_CINCOFFSET(a0, a0, t1)
2:
  li a3, 0
  beqz s4, 3f
  slli t1, s4, 3
  CSM_CINCOFFSET(a0, a0, t1)
  ld a3, 0(a0)                 /* a3: the descriptor VA */
  sub t1, x0, t1
  CSM_CINCOFFSET(a0, a0, t1)
3:
  /* ---- desc & 7 -> 4 */
  andi t0, a3, 7
  beqz t0, 4f
  li a5, 4
  mv a6, a3
  j csm_inject
4:
  /* ---- the descriptor's three words, each walked and read physically; a word not readable -> 13 */
  mv s6, a3
  CSM_WALK(s6, a2, 0, csm_desc_fault, csm_w1)
  CSM_PHYS(t6, a2, t3, t4)
  ld s7, 0(t6)                 /* s7: base */
  addi s6, a3, 8
  CSM_WALK(s6, a2, 0, csm_desc_fault, csm_w2)
  CSM_PHYS(t6, a2, t3, t4)
  ld s8, 0(t6)                 /* s8: end */
  addi s6, a3, 16
  CSM_WALK(s6, a2, 0, csm_desc_fault, csm_w3)
  CSM_PHYS(t6, a2, t3, t4)
  ld s9, 0(t6)                 /* s9: perms */
  /* ---- slot & 15 -> 6 */
  andi t0, a4, 15
  beqz t0, 5f
  li a5, 6
  mv a6, a4
  j csm_inject
5:
  /* ---- the slot, walked as a store (15 if refused); s10: its physical address */
  CSM_WALK(a4, s10, 1, csm_slot_fault, csm_w4)
  /* ---- the descriptor: base < end, perms <= 15, representable (29) */
  bgeu s7, s8, csm_bad_desc
  li t0, 15
  bltu t0, s9, csm_bad_desc
  sub t0, s8, s7               /* len */
  li t1, 4096
  bltu t0, t1, 7f
  li t1, 63
  li t2, 1
6:
  sll t3, t2, t1
  and t3, t3, t0
  bnez t3, 61f
  addi t1, t1, -1
  j 6b
61:
  addi t1, t1, -9
  sll t1, t2, t1               /* granule = 2^(msb(len) - 9) */
  addi t1, t1, -1
  and t3, s7, t1
  bnez t3, csm_bad_desc
  and t3, t0, t1
  bnez t3, csm_bad_desc
7:
  /* ---- the table at srevroot: capacity - next + free_count >= 2, else 30 */
  csrr t0, CSM_SREVROOT
  CSM_PHYS(t6, t0, t3, t4)
  ld t1, 0(t6)                 /* capacity | flags[63:62] */
  slli t1, t1, 2
  srli t1, t1, 2
  ld t2, 8(t6)                 /* next */
  ld t3, 16(t6)                /* free_head | free_count << 32 */
  srli t3, t3, 32
  sub t1, t1, t2
  add t1, t1, t3
  li t2, 2
  blt t1, t2, csm_reserve
  /* ---- mint the LINEAR capability */
  CSM_CAPCREATE(s11)
  li t0, CSM_TYPE_LIN
  CSM_CAPTYPE(s11, t0)
  CSM_CAPBOUND(s11, s7, s8)
  CSM_CAPPERM(s11, s9)
  csrr s6, mepc                /* a CSNODE refusal traps from M and overwrites mepc and mstatus */
  csrr s7, mstatus
  /* ---- the window's trap vector. On the RTL the vector is mtvec in both modes: CCSR ctvec is {tag, metadata} over
   * mtvec's cursor, a trap does not take its PCC from it, and CCSRRW with rs1 = x0 copies without writing (the RTL
   * lane, bf3db3175: csr_regfile.sv's ctvec arm, frontend.sv's npc_metadata). So: copy the vector capability into s8,
   * point mtvec at csm_window_trap for the window (a plain csrw untags the pair, which nothing in the window reads),
   * and put cursor, metadata and tag back with one CCSRRW afterwards. */
  CSM_CCSRRW(s8, CSM_CCSR_CTVEC, x0)
  lla t0, csm_window_trap
  csrw mtvec, t0
  /* ---- the window */
  csrsi CSM_CSUPCTL, 8
  CSM_CSNODE(a6, s11)
  csrci CSM_CSUPCTL, 8
  CSM_CCSRRW(x0, CSM_CCSR_CTVEC, s8)       /* the monitor's vector back */
  /* ---- STC through a physical capability over the slot's page */
  CSM_PHYS(t6, s10, t3, t4)
  CSM_STC(s11, t6, 0)
  /* ---- rd := the ancestor; mepc + 4 */
  beqz s5, 8f
  slli t1, s5, 3
  CSM_CINCOFFSET(a0, a0, t1)
  sd a6, 0(a0)
  sub t1, x0, t1
  CSM_CINCOFFSET(a0, a0, t1)
8:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  li a0, 1
  ret

/* A CSNODE refusal inside the window lands here, in M (mepc = the CSNODE): clear the bit (its write flushes), put
 * back the trap vector, mepc and mstatus, inject. */
.align 2
csm_window_trap:
  csrci CSM_CSUPCTL, 8
  csrr a5, mcause
  li a6, 0
  CSM_CCSRRW(x0, CSM_CCSR_CTVEC, s8)
  csrw mepc, s6
  csrw mstatus, s7
  j csm_inject

csm_bad_operand:
  li a5, 2
  mv a6, s2
  j csm_inject
csm_desc_fault:
  li a5, 13
  mv a6, s6
  j csm_inject
csm_slot_fault:
  li a5, 15
  mv a6, a4
  j csm_inject
csm_bad_desc:
  li a5, 29
  li a6, 0
  j csm_inject
csm_reserve:
  li a5, 30
  li a6, 0
  /* fall through */

/* Inject cause a5 with tval a6 into S as a trap from S: scause, stval, sepc = mepc; sstatus.SPP = 1, SPIE = SIE,
 * SIE = 0; mepc = stvec (direct mode). MPP stays S. */
csm_inject:
  csrw scause, a5
  csrw stval, a6
  csrr t0, mepc
  csrw sepc, t0
  csrr t0, sstatus
  ori t0, t0, 0x100            /* SPP: from S */
  andi t1, t0, 2               /* SIE -> SPIE */
  slli t1, t1, 4
  andi t0, t0, ~0x22
  or t0, t0, t1
  csrw sstatus, t0
  csrr t0, stvec
  andi t0, t0, -4
  csrw mepc, t0
  li a0, 1
  ret

csm_not_ours:
  li a0, 0
  ret
