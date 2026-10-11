/* What apr_pools.c references and no pool workload reaches: the userdata
 * hash, the subprocess chain and apr_psprintf's formatter share its
 * translation unit. Each one ends the run with its own code rather than
 * returning quietly, so a case that came to depend on one cannot pass
 * unnoticed. apr_atomic_init is the exception: apr_pool_initialize calls it
 * on every run, and it has nothing to initialize here. */
#include "apr_shim.h"
#include "port.h"

apr_status_t apr_atomic_init(struct apr_pool_t *p) {
  (void)p;
  return APR_SUCCESS;
}
apr_hash_t *apr_hash_make(struct apr_pool_t *p) {
  (void)p;
  aprp_fail(601);
}
void *apr_hash_get(apr_hash_t *h, const void *k, apr_ssize_t n) {
  (void)h;
  (void)k;
  (void)n;
  aprp_fail(602);
}
void apr_hash_set(apr_hash_t *h, const void *k, apr_ssize_t n, const void *v) {
  (void)h;
  (void)k;
  (void)n;
  (void)v;
  aprp_fail(603);
}
apr_status_t apr_proc_kill(apr_proc_t *p, int s) {
  (void)p;
  (void)s;
  aprp_fail(604);
}
apr_status_t apr_proc_wait(apr_proc_t *p, int *e, int *w, int h) {
  (void)p;
  (void)e;
  (void)w;
  (void)h;
  aprp_fail(605);
}
char *apr_pstrdup(struct apr_pool_t *p, const char *s) {
  (void)p;
  (void)s;
  aprp_fail(606);
}
void apr_sleep(apr_time_t t) {
  (void)t;
  aprp_fail(607);
}
int apr_vformatter(int (*flush)(apr_vformatter_buff_t *), apr_vformatter_buff_t *b,
                   const char *fmt, va_list ap) {
  (void)flush;
  (void)b;
  (void)fmt;
  (void)ap;
  aprp_fail(608);
}
#ifdef APRP_DOMAIN
/* apr_buckets_alloc.c aborts when a create cannot get its first block. */
_Noreturn void abort(void) { aprp_fail(606); }
#endif
