/* Refuse an unadapted libc backing path rather than silently supplying memory
 * with no allocator-lifetime protection. AllocSet, Generation, Slab and Bump
 * use the explicit context-pool interface; none should reach these functions.
 * Keep the guard for future upstream paths and unsupported configurations. */
#include <stddef.h>

void pg_domain_text(const char *s);
__attribute__((noreturn)) void pg_subpool_refuse(const char *what);

static void refuse(const char *what) {
  pg_domain_text("pg-sublet: ");
  pg_domain_text(what);
  pg_domain_text(" reached an unadapted backing path; the four Sublet managers "
                 "must allocate through context pools.\n");
  pg_subpool_refuse(what);
}

void *malloc(size_t n) {
  (void)n;
  refuse("malloc");
  return NULL;
}
void *calloc(size_t k, size_t n) {
  (void)k;
  (void)n;
  refuse("calloc");
  return NULL;
}
void *realloc(void *p, size_t n) {
  (void)p;
  (void)n;
  refuse("realloc");
  return NULL;
}
void free(void *p) {
  (void)p;
  refuse("free");
}
