/* What apr_pools.c references and this fixture does not use. Each one aborts:
 * a case that silently depended on a stub would otherwise pass quietly, and
 * the point of the seam is that it is visible. The census enumerates exactly
 * this list with nm --undefined-only. */
#include <stdlib.h>
#include <stdio.h>
static void off_path(const char *name) {
  fprintf(stderr, "CONTROL-FAILED fixture reached the stub for %s\n", name);
  exit(75);
}
int apr_atomic_init(void *p) { (void)p; return 0; } /* called by apr_pool_initialize */
void *apr_hash_make(void *p) { (void)p; off_path("apr_hash_make"); return 0; }
void *apr_hash_get(void *h, const void *k, long n) { (void)h; (void)k; (void)n; off_path("apr_hash_get"); return 0; }
void apr_hash_set(void *h, const void *k, long n, const void *v) { (void)h; (void)k; (void)n; (void)v; off_path("apr_hash_set"); }
int apr_proc_kill(void *p, int s) { (void)p; (void)s; off_path("apr_proc_kill"); return 0; }
int apr_proc_wait(void *p, int *e, int *w, int h) { (void)p; (void)e; (void)w; (void)h; off_path("apr_proc_wait"); return 0; }
char *apr_pstrdup(void *p, const char *s) { (void)p; (void)s; off_path("apr_pstrdup"); return 0; }
void apr_sleep(long t) { (void)t; off_path("apr_sleep"); }
int apr_vformatter(int (*f)(void *, const char *, int), void *d, const char *fmt, void *ap) {
  (void)f; (void)d; (void)fmt; (void)ap; off_path("apr_vformatter"); return 0;
}
