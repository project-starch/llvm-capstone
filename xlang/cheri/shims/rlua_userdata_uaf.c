/* Row 1 — rlua #19, heap use-after-free of Rust-owned memory via Lua `__gc`.
 *
 * asan.txt: heap-use-after-free, READ of size 43, 0 bytes inside a 43-byte
 * region. Free site `rlua::util::destructor` (util.rs:279), driven by Lua's
 * garbage collector; stale-use site is the userdata `access` method.
 *
 * Mechanism: a Rust value is stored as Lua userdata. Lua's `__gc` runs the
 * registered destructor, which drops the Rust value and releases its heap
 * buffer. The userdata handle is then resurrected and its method reads
 * through the pointer the destructor already freed.
 *
 * WHY THIS IS SHIMMABLE despite the free running inside Rust's drop glue:
 * the CHERI-relevant event is malloc -> free -> read. Rust's double-drop is
 * the CAUSE of the early free, not the behaviour CHERI observes, and Rust's
 * allocator bottoms out in malloc/free like everything else here. No Rust
 * ownership semantics are re-implemented; only the memory events are.
 *
 * EXPECT NO SURPRISE. This row's verdict duplicates the other
 * allocator-visible UAF rows by construction — it is here for completeness of
 * the corpus table, not because it adds a new shape. Its value is that a
 * reader asking "what happened to row 1?" gets a measured answer rather than
 * an exclusion note.
 */
#include "../mock-mruby/mock_mruby.h"   /* mock_report only */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define USERDATA_BYTES 43   /* the freed region ASan names */

static volatile uint64_t sink;

/* The Rust-owned buffer behind the Lua userdata. */
static unsigned char *rust_value_new(void) {
  unsigned char *p = (unsigned char *)malloc(USERDATA_BYTES);
  if (!p) abort();
  memset(p, 0xA5, USERDATA_BYTES);
  return p;
}

/* rlua::util::destructor, driven by Lua's __gc: drops the Rust value. */
static void lua_gc_destructor(unsigned char *p) { free(p); }

int main(void) {
  unsigned char *ud = rust_value_new();

  /* The userdata handle Lua hands back to the resurrected reference. */
  volatile unsigned char *handle = ud;

  lua_gc_destructor(ud);          /* __gc -> destructor -> drop -> free */

  /* The `access` method reads the whole userdata through the stale handle. */
  unsigned char buf[USERDATA_BYTES];
  memcpy(buf, (const void *)handle, USERDATA_BYTES);  /* READ of size 43 */
  sink = buf[0];

  mock_report("rlua_userdata_uaf", "use-after-free-survived");
  return 0;
}
