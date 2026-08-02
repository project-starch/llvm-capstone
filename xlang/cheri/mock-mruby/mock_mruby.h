/* mock-mruby — minimal mruby lifecycle harness for the CHERI baseline.
 *
 * Reproduces ONLY the memory-lifecycle events each xlang row's CHERI verdict
 * depends on: what is allocated, how it dies (realloc-move / GC sweep /
 * explicit free), and which stale pointer is dereferenced afterwards. There is
 * no VM, no parser, no GC mark phase.
 *
 * Call chains and function names mirror mruby's (stack_init,
 * stack_extend_alloc, mrb_stack_extend, mrb_realloc, obj_free, mrb_free) so an
 * ASan trace from a shim reads against the row's own asan.txt.
 *
 * Slot width: mruby's pinned builds are word-boxed, so an mrb_value is 8 bytes
 * and a 1024-byte register stack is 128 slots — the "~128 slots of slack" row 6
 * depends on. The capability under test is NOT the slot; it is the C-side
 * `mrb_value *regs` pointer held across a callback, which is exactly what
 * purecap CHERI has to police.
 */
#ifndef MOCK_MRUBY_H
#define MOCK_MRUBY_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t mrb_value; /* word-boxed slot, 8 bytes */

typedef struct mrb_context {
  mrb_value *stack;      /* the VM register stack */
  size_t     stack_size; /* bytes */
  void      *wedge;      /* stands in for the trigger's heap fragmentation */
} mrb_context;

typedef struct mrb_state {
  mrb_context *c;
  void        *gc_obj;      /* object the sweep phase will free (rows 9, 14) */
  size_t       gc_obj_size;
} mrb_state;

typedef void (*mrb_callback)(mrb_state *);

/* stack_init: calloc the register stack. */
mrb_state *mrb_open(size_t stack_bytes);
void       mrb_close(mrb_state *mrb);

/* mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc.
 * The row's defect requires the stack to MOVE; if realloc grows in place the
 * bug would not fire in real mruby either, so this aborts loudly (rc 66)
 * rather than silently measuring nothing. */
void mrb_stack_extend(mrb_state *mrb, size_t new_bytes);

/* Re-enter the VM the way a Ruby-level callback does (#eql?, #to_s, #<=>,
 * const_missing). This is the window in which the stack is reallocated. */
void mrb_funcall_cb(mrb_state *mrb, mrb_callback cb);

/* Plain allocation, as mrb_malloc/mrb_free (row 12's explicit gem free). */
void *mrb_malloc(mrb_state *mrb, size_t n);
void  mrb_free(mrb_state *mrb, void *p);

/* GC path: an object registered here is released by mrb_full_gc through
 * obj_free -> mrb_free -> free (rows 9, 14). It IS a real free() — which is
 * the audit's finding: mruby's arena does not hide these from revocation. */
void *mrb_gc_alloc(mrb_state *mrb, size_t n);
void  mrb_full_gc(mrb_state *mrb);

/* Result reporting, so run-in-guest.sh sees a stable line per shim. */
void mock_report(const char *row, const char *what);

#endif /* MOCK_MRUBY_H */
