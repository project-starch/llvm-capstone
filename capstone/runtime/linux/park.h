/* The launcher's parking queue (docs/plans/delegation-threads.md, "Parking").
 *
 * Linux cannot see domain memory, so a domain lock word cannot be handed to
 * futex. The domain keeps the authoritative lock state; this queue only puts
 * launcher threads to sleep and wakes them. A table of generation words, one
 * per bucket, is shared with the domain: only this code writes it, under the
 * park mutex, with release semantics. A key is a domain address as an integer
 * and is never dereferenced here.
 *
 * Each launcher thread owns one wait record. Its `notified` word is that
 * thread's private futex word: a waker sets it before waking, so a wake
 * between enqueue and sleep is never lost. Queue membership and every final
 * state change happen under the park mutex, which is never held while
 * sleeping. A generation stops at UINT64_MAX instead of wrapping: from then on
 * the bucket answers every WAIT with RECHECK.
 */
#ifndef CAPSTONE_LINUX_PARK_H
#define CAPSTONE_LINUX_PARK_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

enum capstone_park_outcome {
  CAPSTONE_PARK_WOKEN,   /* selected by WAKE, or by the wake part of REQUEUE */
  CAPSTONE_PARK_RECHECK, /* the generation moved, or the bucket is saturated */
  CAPSTONE_PARK_TIMEOUT,
  CAPSTONE_PARK_EINTR,
};

enum capstone_park_state {
  CAPSTONE_PARK_IDLE,
  CAPSTONE_PARK_QUEUED,
  CAPSTONE_PARK_NOTIFIED,
  CAPSTONE_PARK_ABORTED,
};

struct capstone_park_record {
  _Atomic uint32_t notified;
  int state, outcome;
  unsigned bucket;
  uint64_t key;
  struct capstone_park_record *next, *prev;
};

struct capstone_park_queue {
  struct capstone_park_record *head, *tail;
};

struct capstone_park {
  pthread_mutex_t lock;
  _Atomic uint64_t *gen;
  unsigned buckets; /* a power of two */
  struct capstone_park_queue *queues;
};

/* Points where a test may stop the calling thread. NULL outside tests. */
enum capstone_park_point {
  CAPSTONE_PARK_ENQUEUED,   /* WAIT: queued and unlocked, before its first sleep */
  CAPSTONE_PARK_SLEEPING,   /* WAIT: immediately before each futex wait */
  CAPSTONE_PARK_ABORTING,   /* WAIT: the sleep ended by timeout or signal, before the mutex */
  CAPSTONE_PARK_COMPLETING, /* waker, mutex held: the record is notified, before the futex wake */
};
extern void (*capstone_park_test_hook)(enum capstone_park_point, struct capstone_park_record *);

/* gen: `buckets` generation words, shared with the domain. 0, or -1 with errno. */
int capstone_park_init(struct capstone_park *park, _Atomic uint64_t *gen, unsigned buckets);
void capstone_park_destroy(struct capstone_park *park);
unsigned capstone_park_bucket(const struct capstone_park *park, uint64_t key);

/* Sleep on key if the bucket's generation still is gen. deadline: absolute
 * CLOCK_MONOTONIC, or NULL. It is not modified, so a caller continuing after
 * EINTR keeps its original budget. The record must be IDLE and is IDLE again
 * on return. */
enum capstone_park_outcome capstone_park_wait(struct capstone_park *park,
                                              struct capstone_park_record *record,
                                              uint64_t key, uint64_t gen,
                                              const struct timespec *deadline);

/* Advance key's generation and complete up to n records waiting on key as
 * WOKEN. Returns how many were selected. */
unsigned capstone_park_wake(struct capstone_park *park, uint64_t key, unsigned n);

/* Advance src's generation, complete up to nwake records waiting on src as
 * WOKEN, and move up to nmove further ones to dst (completing them as RECHECK
 * instead when dst's bucket is saturated). Returns woken + moved. */
unsigned capstone_park_requeue(struct capstone_park *park, uint64_t src, uint64_t dst,
                               unsigned nwake, unsigned nmove);

/* Records in bucket's queue with key, whatever their state (diagnostics and
 * tests: a record left in a queue is found). */
unsigned capstone_park_queued(struct capstone_park *park, unsigned bucket, uint64_t key);

#endif
