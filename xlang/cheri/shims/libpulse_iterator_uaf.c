/* Row 3 — GHSA-f56g-chqp-22m9, libpulse-binding `Proplist::Iterator` UAF.
 *
 * The only Rust row that CAN be measured under purecap, because the memory
 * lifecycle is entirely C-side: the Rust binding merely triggers it. Rows 1
 * and 2 free through Rust's own drop glue (`core::ptr::drop_glue::<RawVec>`)
 * and have no C seam, so they stay out until a purecap Rust toolchain exists.
 *
 * valgrind (native): "Invalid read of size 8, 32 bytes inside a freed
 * 1,072-byte block". Native ASan is blind to the original because the stale
 * read executes inside prebuilt, uninstrumented libpulse.so — this shim is
 * our own code, so ASan does see it.
 *
 * Mechanism, from target.md:
 *   struct Iterator { ptr: *const ProplistInternal, state: *mut c_void }
 * carries a raw copy of the C `pa_proplist*` with NO lifetime tie to the Rust
 * owner. `<Proplist as IntoIterator>::into_iter` consumes the Proplist, whose
 * Drop calls pa_proplist_free (proplist.rs:453); the iterator then calls
 * pa_proplist_iterate -> pa_hashmap_iterate (proplist.rs:171) through the
 * pointer it copied. Free and use are adjacent statements — no GC, no
 * threading, no layout dependence, which is why it reproduces 10/10.
 *
 * Geometry is the valgrind report's: a 1072-byte object, read of size 8 at
 * offset 32 (where pa_hashmap_iterate walks the hashmap's entry list).
 */
#include "../mock-mruby/mock_mruby.h"   /* mock_report only */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPLIST_BYTES 1072
#define ITERATE_OFF 32

/* The C object the binding owns. Allocated and freed by libpulse itself, so
 * plain malloc/free is the faithful model — the same standard as every other
 * row here, and an upper bound on what revocation can see. */
typedef struct ProplistInternal {
  unsigned char opaque[PROPLIST_BYTES];
} ProplistInternal;

static ProplistInternal *pa_proplist_new(void) {
  ProplistInternal *p = (ProplistInternal *)malloc(sizeof *p);
  if (!p) abort();
  memset(p, 0, sizeof *p);
  return p;
}

static void pa_proplist_free(ProplistInternal *p) { free(p); }

/* pa_proplist_iterate -> pa_hashmap_iterate: walks the entry list living
 * inside the proplist object. This is the stale dereference. */
static void *pa_proplist_iterate(const ProplistInternal *p, void **state) {
  const unsigned char *base = (const unsigned char *)p;
  uint64_t entry;
  memcpy(&entry, base + ITERATE_OFF, sizeof entry);  /* READ of size 8 */
  *state = (void *)(uintptr_t)entry;
  return *state;
}

/* The iterator: a raw pointer copy with no lifetime binding it to the owner. */
struct Iterator {
  const ProplistInternal *ptr;
  void *state;
};

int main(void) {
  ProplistInternal *pl = pa_proplist_new();

  /* Proplist::iter() borrows &self but returns an unbound Iterator. */
  struct Iterator it = { pl, NULL };

  /* into_iter() consumes the Proplist; its Drop runs pa_proplist_free. */
  pa_proplist_free(pl);

  /* Iterator::next() -> pa_proplist_iterate through the copied pointer. */
  (void)pa_proplist_iterate(it.ptr, &it.state);

  mock_report("libpulse_iterator_uaf", "use-after-free-survived");
  return 0;
}
