/* Is a freed object in CheriBSD's revocation quarantine, and is it reused before a
 * sweep clears it? Both questions answered without running a sweep: the kernel
 * exposes a shadow bitmap, one bit per 16-byte granule, set while that granule is
 * quarantined. Reading a bit is O(1), so this can run on every malloc/free.
 *
 * LD_PRELOAD this and the wrapped allocator reports at exit:
 *   frees             how many free() calls happened
 *   quarantined       of those, how many had their shadow bit SET right after free
 *   reused_unswept    how many malloc() results came back with the bit STILL set
 *                     -- memory handed out again while quarantined, which is exactly
 *                     the window an async sweep leaves open
 * A stale pointer into memory that never reaches free() shows up as neither.      */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <cheri/cheric.h>
#include <cheri/revoke.h>

static void *(*real_malloc)(size_t);
static void (*real_free)(void *);
static unsigned char *shadow;
static int shadow_err;
static unsigned long n_malloc, n_free, n_quarantined, n_reused_unswept;
static int ready;

static void init(void) {
  if (ready) return;
  ready = 1;
  real_malloc = dlsym(RTLD_NEXT, "malloc");
  real_free = dlsym(RTLD_NEXT, "free");
  void *s = NULL;
  if (cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_NOVMEM_ENTIRE, NULL, &s) != 0) shadow_err = errno;
  else shadow = s;
}

/* The fine-grained map is one bit per capability granule, which revoke.h gives as
 * VM_CHERI_REVOKE_GSZ_MEM_NOMAP (16 bytes here), indexed from the returned pointer. */
static int bit_set(const void *p) {
  if (!shadow) return -1;
  unsigned long g = (unsigned long)cheri_getaddress(p) / VM_CHERI_REVOKE_GSZ_MEM_NOMAP;
  return (shadow[g / 8] >> (g % 8)) & 1;
}
int quarantine_bit(const void *p) { init(); return bit_set(p); }

void *malloc(size_t n) {
  init();
  void *p = real_malloc(n);
  if (p) { n_malloc++; if (bit_set(p) == 1) n_reused_unswept++; }
  return p;
}

void free(void *p) {
  init();
  if (p) {
    real_free(p);
    n_free++;
    if (bit_set(p) == 1) n_quarantined++;
    return;
  }
  real_free(p);
}

__attribute__((destructor)) static void report(void) {
  char line[256];
  int k = snprintf(line, sizeof line,
    "QUARANTINE shadow=%s mallocs=%lu frees=%lu quarantined_after_free=%lu "
    "reused_while_quarantined=%lu\n",
    shadow ? "mapped" : (shadow_err ? "ERRNO" : "null"),
    n_malloc, n_free, n_quarantined, n_reused_unswept);
  if (k > 0) write(2, line, (size_t)k);
}
