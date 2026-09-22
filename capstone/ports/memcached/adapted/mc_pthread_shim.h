/* A single-threaded port: memcached's mutexes stay where upstream takes them
 * and do nothing. slabs.c takes slabs_lock around every operation and cache.c
 * its per-cache mutex; both compile against these. The census counted 31
 * pthread references in slabs.c alone (../README.md), and this is where they
 * land. The port states that it is single-threaded; it does not pretend the
 * mover or the LRU maintainer ran. */
#ifndef MC_PTHREAD_SHIM_H
#define MC_PTHREAD_SHIM_H
#if __STDC_HOSTED__
/* A hosted program that links the allocator also includes its libc's headers,
 * whose pthread types these would collide with; the host's mutexes are
 * uncontended here and cost nothing. */
#include <pthread.h>
#else
typedef int pthread_mutex_t;
typedef int pthread_mutexattr_t;
#define PTHREAD_MUTEX_INITIALIZER 0
static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
  (void)m;
  (void)a;
  return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m) {
  (void)m;
  return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *m) {
  (void)m;
  return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *m) {
  (void)m;
  return 0;
}
#endif /* __STDC_HOSTED__ */
#endif
