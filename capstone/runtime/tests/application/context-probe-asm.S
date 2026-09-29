/* Probe-only helpers for context-probe.c (docs/plans/delegation-threads.md,
 * Probe A). Not part of the runtime. */
#define ldc(rd, rs, offset) .insn i 0x5b, 0x3, rd, offset(rs)
#define stc(rs1, rs2, offset) .insn s 0x5b, 0x4, rs1, offset(rs2)
#define domreturn(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x21, rd, rs1, rs2
#define ccsrrw(rd, ccsr, rs) .insn i 0x5b, 0x7, rd, ccsr(rs)
#define cscratch 0x004

#include "../../include/capstone/context.h"

.text

/* void probe_exit_delayed(unsigned long value, unsigned long loops), never
 * returns. __capstone_context_exit with a register-only delay loop between
 * publishing done and the final switch, so that a quantum expires exactly
 * there (A6). The loop lies between probe_exit_delay_begin and
 * probe_exit_delay_end; a preemption event's pc in that range proves where
 * the context stopped. Nothing after the done store touches the thread area. */
  .global probe_exit_delayed
probe_exit_delayed:
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ld t3, 64(t0)
  ldc(t2, t0, 16)
  sd a0, 168(t0)
  fence rw, w
  li t4, 1
  sd t4, 160(t0)
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  mv t5, a1
  .global probe_exit_delay_begin
probe_exit_delay_begin:
  addi t5, t5, -1
  bnez t5, probe_exit_delay_begin
  .global probe_exit_delay_end
probe_exit_delay_end:
  bnez t3, 1f
  li t4, CAPSTONE_CONTEXT_EXITED
  sd t4, 0(t2)
1:
  li t2, 0
  li sp, 0
  li tp, 0
  li a0, 0
  li a1, 0
  lla t2, .Lprobe_reentry
  domreturn(t1, t2, x0)
.Lprobe_reentry:
  bnez a0, 2f
  li t4, CAPSTONE_CONTEXT_EXITED
  sd t4, 0(a1)
2:
  li a0, 0
  li a1, 0
  lla t2, .Lprobe_reentry
  domreturn(ra, t2, x0)

/* void *probe_descriptor(void): the invocation descriptor the current call
 * entry lent this context, or a null capability when the entry carried none. */
  .global probe_descriptor
probe_descriptor:
  ccsrrw(t0, cscratch, x0)
  ld t1, 64(t0)
  li a0, 0
  bnez t1, 3f
  ldc(a0, t0, 16)
3:
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  ret

/* void probe_store_through(unsigned long *where, unsigned long value): one
 * store at a named instruction, so that a fault it takes can be located. */
  .global probe_store_through
probe_store_through:
  .global probe_store_insn
probe_store_insn:
  sd a1, 0(a0)
  ret

/* unsigned long probe_load_through(unsigned long *where): one load at a named
 * instruction. */
  .global probe_load_through
probe_load_through:
  .global probe_load_insn
probe_load_insn:
  ld a0, 0(a0)
  ret

/* ---- A12: control state a supervised context must not be able to use to
 * escape the supervisor. Each probe instruction carries a label so that a
 * fault can be located. ---- */

/* void probe_csr_read(void): read a machine CSR. */
  .global probe_csr_read
probe_csr_read:
  .global probe_csr_insn
probe_csr_insn:
  csrr t0, mstatus
  li t0, 0
  ret

/* void probe_mret_to_user(unsigned long loops), never returns normally:
 * point cepc at a loop and execute mret. If mret is carried out, the loop
 * runs outside C-mode for `loops` iterations and then executes ecall. */
  .global probe_mret_to_user
probe_mret_to_user:
  mv t5, a0
  lla t0, probe_umode_loop
  ccsrrw(x0, 0x002, t0)
  .global probe_mret_insn
probe_mret_insn:
  mret
  .global probe_umode_loop
probe_umode_loop:
  addi t5, t5, -1
  bnez t5, probe_umode_loop
  .global probe_umode_ecall
probe_umode_ecall:
  ecall
4:
  j 4b

/* void probe_wfi(void): wait for an interrupt the context has masked. */
  .global probe_wfi
probe_wfi:
  .global probe_wfi_insn
probe_wfi_insn:
  wfi
  ret

/* ---- A2: authority at entry. probe_entry_audit is a minted seal's pc. Before
 * the runtime's entry runs it records, as plain words in probe_audit, what the
 * first entry delivered: word i (1..31) the LCC type of x<i>, 7 when untagged;
 * words 32..34 gp's base, end and permissions, 35..37 a1's, when tagged; and
 * words 38..41 cscratch's type, base, end and permissions. It clobbers only
 * t0..t3 (untagged at entry) and restores cscratch. x5 comes from cscratch,
 * which the first instruction swaps with it; x6 and x7 are stashed in the
 * start block at 176 and 192. ---- */
#define lcc(rd, rs, field) .insn r 0x5b, 0x1, 0x04, rd, rs, field
  .global probe_entry_audit
probe_entry_audit:
  ccsrrw(t0, cscratch, t0)        /* t0: start block; cscratch: x5 as entered */
  stc(t1, t0, 176)
  stc(t2, t0, 192)
.Lpcrel_audit:
  auipc t1, %pcrel_hi(probe_audit)
  addi t1, t1, %pcrel_lo(.Lpcrel_audit)
  .insn r 0x5b, 0x1, 0xc, t1, gp, t1   /* cincoffset t1, gp, t1 */
  lcc(t2, x1, x1)
  sd t2, 8(t1)
  lcc(t2, x2, x1)
  sd t2, 16(t1)
  lcc(t2, x3, x1)
  sd t2, 24(t1)
  lcc(t2, x4, x1)
  sd t2, 32(t1)
  lcc(t2, x8, x1)
  sd t2, 64(t1)
  lcc(t2, x9, x1)
  sd t2, 72(t1)
  lcc(t2, x10, x1)
  sd t2, 80(t1)
  lcc(t2, x11, x1)
  sd t2, 88(t1)
  lcc(t2, x12, x1)
  sd t2, 96(t1)
  lcc(t2, x13, x1)
  sd t2, 104(t1)
  lcc(t2, x14, x1)
  sd t2, 112(t1)
  lcc(t2, x15, x1)
  sd t2, 120(t1)
  lcc(t2, x16, x1)
  sd t2, 128(t1)
  lcc(t2, x17, x1)
  sd t2, 136(t1)
  lcc(t2, x18, x1)
  sd t2, 144(t1)
  lcc(t2, x19, x1)
  sd t2, 152(t1)
  lcc(t2, x20, x1)
  sd t2, 160(t1)
  lcc(t2, x21, x1)
  sd t2, 168(t1)
  lcc(t2, x22, x1)
  sd t2, 176(t1)
  lcc(t2, x23, x1)
  sd t2, 184(t1)
  lcc(t2, x24, x1)
  sd t2, 192(t1)
  lcc(t2, x25, x1)
  sd t2, 200(t1)
  lcc(t2, x26, x1)
  sd t2, 208(t1)
  lcc(t2, x27, x1)
  sd t2, 216(t1)
  lcc(t2, x28, x1)
  sd t2, 224(t1)
  lcc(t2, x29, x1)
  sd t2, 232(t1)
  lcc(t2, x30, x1)
  sd t2, 240(t1)
  lcc(t2, x31, x1)
  sd t2, 248(t1)
  ldc(t2, t0, 176)
  lcc(t2, t2, x1)
  sd t2, 48(t1)                   /* x6 */
  ldc(t2, t0, 192)
  lcc(t2, t2, x1)
  sd t2, 56(t1)                   /* x7 */
  stc(x0, t0, 176)
  stc(x0, t0, 192)
  li t3, 7
  lcc(t2, gp, x1)
  beq t2, t3, 1f
  lcc(t2, gp, x3)
  sd t2, 256(t1)
  lcc(t2, gp, x4)
  sd t2, 264(t1)
  lcc(t2, gp, x5)
  sd t2, 272(t1)
1:
  lcc(t2, a1, x1)
  beq t2, t3, 2f
  lcc(t2, a1, x3)
  sd t2, 280(t1)
  lcc(t2, a1, x4)
  sd t2, 288(t1)
  lcc(t2, a1, x5)
  sd t2, 296(t1)
2:
  lcc(t2, t0, x1)
  sd t2, 304(t1)
  lcc(t2, t0, x3)
  sd t2, 312(t1)
  lcc(t2, t0, x4)
  sd t2, 320(t1)
  lcc(t2, t0, x5)
  sd t2, 328(t1)
  ccsrrw(t2, cscratch, t0)        /* t2: x5 as entered; cscratch: start block */
  lcc(t2, t2, x1)
  sd t2, 40(t1)
  li t0, 0
  li t1, 0
  li t2, 0
  li t3, 0
  j __capstone_context_entry

/* The audit's positive control: an entry that leaks a copy of gp into s1
 * (x9) and then audits. */
  .global probe_entry_leak
probe_entry_leak:
  .insn r 0x5b, 0x1, 0xc, s1, gp, x0   /* cincoffset s1, gp, 0 */
  j probe_entry_audit

/* ---- A7: revocation-node instruments of capstone-qemu (LCC fields 8 and 9,
 * answered only with CAPSTONE_TEST_NODE_QUERY; 0 otherwise). ---- */

/* unsigned long probe_node_of(capstone_cap_slot *slot): the revocation node
 * of the capability in slot, which stays in place. */
  .global probe_node_of
probe_node_of:
  ldc(t0, a0, 0)
  lcc(t1, t0, x8)
  stc(t0, a0, 0)
  mv a0, t1
  li t0, 0
  ret

/* unsigned long probe_next_node(void): the node the next allocation takes. */
  .global probe_next_node
probe_next_node:
  lcc(a0, sp, x9)
  ret

/* void probe_main_gp(unsigned long out[3]): type, base and end of the gp this
 * context entered with (its recovery block, slot 32). */
  .global probe_main_gp
probe_main_gp:
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 32)
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  lcc(t2, t1, x1)
  sd t2, 0(a0)
  li t3, 7
  beq t2, t3, 3f
  lcc(t2, t1, x3)
  sd t2, 8(a0)
  lcc(t2, t1, x4)
  sd t2, 16(a0)
3:
  li t1, 0
  li t2, 0
  ret

  .section .bss
  .align 4
  .global probe_audit
probe_audit:
  .zero 336
