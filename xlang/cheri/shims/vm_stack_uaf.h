/* The corpus's dominant defect shape — 6 of the 9 temporal rows (4, 5, 8, 10,
 * 13, 15).
 *
 * Native C caches a raw pointer into the VM register stack, calls back into
 * Ruby (#eql?, #to_s, #<=>, const_missing), and the callback recurses deep
 * enough that mrb_stack_extend reallocs the stack out of place. The cached
 * pointer is then dereferenced. The pointer is what purecap CHERI must police.
 *
 * Geometry is expressed in explicit BYTE constants on purpose. The native
 * gate validates it at 8-byte pointers and the CHERI run executes it at
 * 16-byte capabilities, so anything derived from sizeof(void*) — or from a
 * struct holding a pointer — would be validated at one layout and measured at
 * another. See check_shim_fidelity.py's module docstring.
 *
 * Per-row constants are taken from that row's asan.txt:
 *   ROW_ID       row number, for the result line
 *   STACK_BYTES  size of the region ASan names
 *   ACCESS_OFF   byte offset of the stale access inside that region
 *   ACCESS_SIZE  8 or 16, as ASan reports it
 *   ROW_IS_WRITE 1 for a WRITE row, 0 for a READ row
 */
#ifndef VM_STACK_UAF_H
#define VM_STACK_UAF_H

#include "../mock-mruby/mock_mruby.h"
#include <string.h>
/* Deliberately NOT <stdio.h>: this template never prints (mock_report is
 * declared in mock_mruby.h), and the Capstone domain build is freestanding with
 * 16-byte pointers, where glibc's struct _IO_FILE sizes an array as
 * `12*sizeof(int) - 5*sizeof(void*)` and fails to compile with a negative
 * length. Keeping the include would cost the Capstone column for nothing. */

#if !defined(ROW_ID) || !defined(STACK_BYTES) || !defined(ACCESS_OFF) ||       \
    !defined(ACCESS_SIZE) || !defined(ROW_IS_WRITE)
#error "row constants must be defined before including vm_stack_uaf.h"
#endif

_Static_assert(ACCESS_OFF < STACK_BYTES,
               "the stale access must land INSIDE the freed region: this is a "
               "temporal row, not a spatial one");
_Static_assert(ACCESS_SIZE == 8 || ACCESS_SIZE == 16, "ASan-reported width");

static volatile uint64_t sink;

/* The callback recurses and grows the stack. Crossing a size class plus the
 * adjacent wedge block forces the move; mrb_stack_extend aborts if it does
 * not move, since then there would be no stale pointer to measure. */
static void deep_callback(mrb_state *mrb) {
  mrb_stack_extend(mrb, STACK_BYTES * 4);
}

int main(void) {
  mrb_state *mrb = mrb_open(STACK_BYTES);

  /* mrb_get_args / the OP handler caches this across the call. */
  mrb_value *regs = mrb->c->stack;

  mrb_funcall_cb(mrb, deep_callback); /* frees the region `regs` points into */

  unsigned char *stale = (unsigned char *)regs + ACCESS_OFF;

#if ROW_IS_WRITE
  *(volatile uint64_t *)stale = 0xdeadbeefULL; /* WRITE of size 8 */
#elif ACCESS_SIZE == 16
  unsigned char buf[16];
  memcpy(buf, stale, 16); /* READ of size 16 */
  sink = buf[0];
#else
  sink = *(volatile uint64_t *)stale; /* READ of size 8 */
#endif

  mock_report(ROW_ID, "use-after-free-survived");
  mrb_close(mrb);
  return 0;
}

#endif /* VM_STACK_UAF_H */
