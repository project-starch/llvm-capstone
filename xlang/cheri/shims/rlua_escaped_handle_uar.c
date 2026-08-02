/* Row 2 — rlua #97, unconstrained callback lifetime.
 *
 * The ONLY row in this corpus that is not a heap defect. ASan calls it
 * `stack-use-after-return`, READ of size 8, at offset 576 in the escaped
 * frame. The rlua API let a `Table` handle escape the closure it was scoped
 * to; `<rlua::table::Table>::len` then read it after that frame returned.
 *
 * Why this one is measurable despite being a Rust row: no allocator is
 * involved at all. Nothing is malloc'd and nothing is freed, so there is no
 * allocator route for a shim author to pick — the machine-level event is
 * "take a pointer into a frame, return, reuse the frame, read the pointer",
 * which is language-independent. Contrast row 1, whose free goes through
 * Rust's drop glue: shimmed in C it would be indistinguishable from rows
 * 4/5/8/... and would teach nothing.
 *
 * Why it matters for the CHERI column: revocation is a HEAP mechanism. It
 * quarantines allocations returned to malloc and sweeps dangling capabilities
 * to them. A returned stack frame is never handed to the allocator, so no
 * configuration — not even revoke-on-every-free — has anything to act on.
 * Predicted MISS under spatial, temporal AND eager. That makes this the
 * corpus's clean "CHERI cannot" case, and unlike the allocator-hidden worst
 * case it needs no simulation to demonstrate.
 *
 * Purecap nuance worth stating: the compiler DOES bound an address-taken
 * local to that object, so this is not a bounds violation — the capability
 * stays exactly in bounds of the storage it was derived from. The storage is
 * simply reused by a later frame. Bounds cannot see that, and revocation is
 * not watching the stack.
 */
#include "../mock-mruby/mock_mruby.h"   /* mock_report only */
#include <stdint.h>

#define FRAME_SLOTS 96          /* offset 576 = slot 72 of an 8-byte-slot frame */
#define ESCAPED_SLOT 72

static volatile uint64_t sink;

/* The closure body: builds a handle into ITS OWN frame and lets it escape,
 * exactly as rlua's callback signature permitted. */
static uint64_t *scope_that_returns(void) {
  uint64_t frame[FRAME_SLOTS];
  for (int i = 0; i < FRAME_SLOTS; i++) frame[i] = 0xA11CE000UL + i;
  /* The Table handle the caller keeps hold of. */
  return &frame[ESCAPED_SLOT];
}

/* A later, unrelated call that reuses the same stack storage. */
static uint64_t reuse_the_frame(void) {
  uint64_t other[FRAME_SLOTS];
  for (int i = 0; i < FRAME_SLOTS; i++) other[i] = 0xDEAD0000UL + i;
  return other[ESCAPED_SLOT];
}

int main(void) {
  uint64_t *escaped = scope_that_returns();  /* frame is dead on return */
  (void)reuse_the_frame();                   /* storage recycled */

  /* <rlua::table::Table>::len reads the escaped handle. READ of size 8. */
  sink = *(volatile uint64_t *)escaped;

  mock_report("rlua_escaped_handle_uar", "stack-use-after-return-survived");
  return 0;
}
