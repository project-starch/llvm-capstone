/* mock-mruby lifecycle harness, backed by Capstone's revoke-on-free allocator.
 *
 * Same API and the same call chains as ../cheri/mock-mruby/mock_mruby.c, so a
 * row's Capstone verdict and its CHERI verdict are produced by identical
 * lifecycle events. The ONLY difference is what sits under mrb_realloc/mrb_free:
 * plain realloc/free there, rof_malloc/rof_free (SPLIT / MREV / delin / REVOKE)
 * here. That difference is the measurement.
 *
 * Contract: ../capstone/ALLOCATOR-CONTRACT.md. Read it before changing anything
 * in the realloc path -- §2 is the case the whole corpus is built on.
 */
#include "../cheri/mock-mruby/mock_mruby.h"
#include "revoke_on_free_alloc.h"

/* Freestanding: no libc. The domain provides these. */
void        mock_report(const char *row, const char *what);
extern void xlang_fail(const char *msg, int code);

/* revoke_on_free_alloc.h is entirely `static` -- it is written for a single
 * translation unit (the sqlite corpus includes it in one domain file). If the
 * domain includes it too, each TU gets its OWN rof_arena and rof_slots: the
 * domain's rof_init fills the domain's copy, the mock's rof_malloc reads the
 * mock's, which is still null and untagged, and QEMU aborts in helper_cslcc on
 * the first cap_get_base. THIS TU OWNS THE ALLOCATOR; the domain reaches it
 * only through the two functions below. */
void xlang_arena_init(void *grant) { rof_init(grant); }
void xlang_set_no_revoke(void) { rof_no_revoke = 1; }

/* Delivery probe (see the domain): allocate, revoke, hand the stale alias back.
 * No arithmetic is performed on it here or by the caller, so the next operation
 * is a plain load -- the route sqlite row3_b2 shows produces cause 25. */
void *xlang_probe_alloc_and_revoke(void) {
  void *p = rof_malloc(64);
  rof_free(p); /* revokes */
  return p;    /* now-untagged alias */
}

/* Two shims allocate directly instead of through the mock (rlua_userdata_uaf,
 * libpulse_iterator_uaf). Route their malloc/free through the SAME revoking
 * allocator, so a shim's own free() revokes exactly as mrb_free does. Anything
 * else and those rows would report MISS while measuring nothing. */
void *malloc(size_t n) { return rof_malloc((unsigned long)n); }
void  free(void *p) { rof_free(p); }

static void *mock_memset0(void *p, unsigned long n) {
  unsigned char *b = (unsigned char *)p;
  for (unsigned long i = 0; i < n; ++i)
    b[i] = 0;
  return p;
}

/* --- allocator layer, named as in mruby's gc.c / state.c ------------------ */

/* ALLOCATOR-CONTRACT §2. rof has no realloc; this is it.
 *
 * Always moves: rof_malloc carves a fresh region off the top of the arena and
 * never coalesces, so it cannot grow in place. That is exactly what the row
 * needs -- mrb_stack_extend below aborts if the stack does not move, because
 * without a move there is no stale pointer and the row would report a MISS that
 * says nothing about the mechanism.
 *
 * The copy is rof_copy_caps (ldc/stc word moves), not a byte loop: a byte loop
 * strips the tag off any capability inside the block, and the later fault on
 * that untagged pointer would look like a catch while being an artifact of the
 * copy. The shims' stacks hold uint64_t slots today, so this is latent -- it
 * stops being latent the moment a row stores a pointer in the register stack.
 *
 * rof_free REVOKES the old node, which is what makes every alias the caller
 * cached -- the whole point of this column -- stop dereferencing.
 */
static void *mrb_realloc(void *p, size_t size) {
  if (size == 0) {
    rof_free(p);
    return (void *)0;
  }
  void *np = rof_malloc((unsigned long)size);
  if (!np)
    return (void *)0;
  if (p) {
    unsigned long old = rof_size(p);
    unsigned long n = old < (unsigned long)size ? old : (unsigned long)size;
    rof_copy_caps(np, p, n);
    rof_free(p); /* the revoke */
  }
  return np;
}

void *mrb_malloc(mrb_state *mrb, size_t n) {
  (void)mrb;
  void *p = mrb_realloc((void *)0, n);
  if (!p)
    xlang_fail("mrb_malloc: arena depleted", 67);
  return p;
}

void mrb_free(mrb_state *mrb, void *p) {
  (void)mrb;
  rof_free(p);
}

/* --- VM register stack ---------------------------------------------------- */

static void stack_init(mrb_state *mrb, size_t stack_bytes) {
  mrb->c->stack = (mrb_value *)mrb_malloc(mrb, stack_bytes);
  mock_memset0(mrb->c->stack, stack_bytes);
  mrb->c->stack_size = stack_bytes;
  /* The triggers push per-frame strings/arrays to "prevent in-place
   * reallocations of the stack". Kept for parity with the CHERI mock even
   * though this allocator can never realloc in place. */
  mrb->c->wedge = mrb_malloc(mrb, 64);
}

static mrb_value *stack_extend_alloc(mrb_state *mrb, size_t new_bytes) {
  return (mrb_value *)mrb_realloc(mrb->c->stack, new_bytes);
}

void mrb_stack_extend(mrb_state *mrb, size_t new_bytes) {
  mrb_value *old = mrb->c->stack;
  mrb_value *ns = stack_extend_alloc(mrb, new_bytes);
  if (!ns)
    xlang_fail("mrb_stack_extend: arena depleted", 67);
  if (ns == old)
    xlang_fail("SHIM-INVALID: realloc grew the VM stack in place; "
               "no stale pointer exists to measure",
               66);
  mrb->c->stack = ns;
  mrb->c->stack_size = new_bytes;
}

void mrb_funcall_cb(mrb_state *mrb, mrb_callback cb) { cb(mrb); }

/* --- GC path -------------------------------------------------------------- */

void *mrb_gc_alloc(mrb_state *mrb, size_t n) {
  mrb->gc_obj = mrb_malloc(mrb, n);
  mrb->gc_obj_size = n;
  mock_memset0(mrb->gc_obj, n);
  return mrb->gc_obj;
}

static void obj_free(mrb_state *mrb, void *p) { mrb_free(mrb, p); }

static void incremental_sweep_phase(mrb_state *mrb) {
  if (mrb->gc_obj) {
    obj_free(mrb, mrb->gc_obj);
    mrb->gc_obj = (void *)0;
  }
}

void mrb_full_gc(mrb_state *mrb) { incremental_sweep_phase(mrb); }

/* --- lifecycle ------------------------------------------------------------ */

mrb_state *mrb_open(size_t stack_bytes) {
  mrb_state *mrb = (mrb_state *)mrb_malloc((mrb_state *)0, sizeof *mrb);
  mock_memset0(mrb, sizeof *mrb);
  mrb->c = (mrb_context *)mrb_malloc(mrb, sizeof *mrb->c);
  mock_memset0(mrb->c, sizeof *mrb->c);
  stack_init(mrb, stack_bytes);
  return mrb;
}

void mrb_close(mrb_state *mrb) {
  if (!mrb)
    return;
  /* NOTE: the stale access already happened by the time a shim calls this. On
   * the CHERI side these frees are plain free(); here each one revokes. Order
   * matters only in that mrb->c must outlive its own fields' frees. */
  mrb_free(mrb, mrb->c->stack);
  mrb_free(mrb, mrb->c->wedge);
  mrb_free(mrb, mrb->c);
  mrb_free(mrb, mrb);
}
