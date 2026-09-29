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
