/* Row 6 — CVE-2026-1979 / mruby#6701, JMPNOT-to-JMPIF peephole corruption.
 * SPATIAL, not temporal: nothing is freed. NVD assigns CWE-119 here, and only
 * CWE-119 describes what reproduces.
 *
 * asan.txt: heap-buffer-overflow WRITE of size 8, mrb_vm_exec vm.c:1788,
 *           32 bytes AFTER a 1024-byte VM stack region.
 *
 * Mechanism: codegen.c:6632 does s->iseq[fail_pos - 2] = OP_JMPIF over a
 * 3-byte OP_JMP, landing on an operand rather than an opcode. The instruction
 * becomes LOADI_5 R38 — register 38 — in a frame whose nregs is 4. The store
 * address is therefore frame_base + 38 slots, which leaves the stack once the
 * frame base is deep enough.
 *
 * The depth threshold is DERIVED here, not asserted: at top level the frame
 * has ~128 slots of slack so the stray write stays in bounds and nothing
 * faults. With 8-byte word-boxed slots and a 1024-byte (128-slot) stack:
 *   depth 28 -> slot 126 -> byte 1008  IN BOUNDS   (silent wrong answer)
 *   depth 29 -> slot 129 -> byte 1032  first crossing
 *   depth 30 -> slot 132 -> byte 1056  = +32, exactly what asan.txt reports
 * Below depth 29 this shim would report a spurious MISS and would be
 * measuring the trigger rather than CHERI, so the bound is a static assert.
 */
#include "../mock-mruby/mock_mruby.h"
#include <stdint.h>

#define STACK_BYTES 1024
#define SLOT_BYTES 8
#define STACK_SLOTS (STACK_BYTES / SLOT_BYTES)

#ifndef DEPTH
#define DEPTH 30      /* trigger.rb calls rec(30); 29-31, 80, 118 all fault */
#endif
#define TOP_SLACK 4   /* slots below the first frame */
#define FRAME_SLOTS 3 /* nregs=4 frame, 3 slots of advance per recursion */
#define BAD_REG 38    /* == OP_JMPIF, written into the operand position */

#define FRAME_BASE (TOP_SLACK + DEPTH * FRAME_SLOTS)
#define STRAY_SLOT (FRAME_BASE + BAD_REG)

_Static_assert(STRAY_SLOT >= STACK_SLOTS,
               "recursion depth is in the IN-BOUNDS regime: the stray write "
               "would stay inside the allocation, CHERI could not fault, and "
               "the row would measure the trigger instead. Depth must be >=29.");

int main(void) {
  mrb_state *mrb = mrb_open(STACK_BYTES);
  mrb_value *regs = mrb->c->stack;

  /* The corrupted LOADI_5 stores to R38 of a 4-register frame. */
  ((volatile mrb_value *)regs)[STRAY_SLOT] = 5; /* WRITE of size 8, OOB */

  mock_report("mruby_bytecode_overflow", "overflow-survived");
  mrb_close(mrb);
  return 0;
}
