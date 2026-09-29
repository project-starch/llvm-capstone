/* The launcher's parking queue; see park.h. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "park.h"
#include "capstone/delegate.h"

#include <errno.h>
#include <linux/futex.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

void (*capstone_park_test_hook)(enum capstone_park_point, struct capstone_park_record *);

static void hook(enum capstone_park_point point, struct capstone_park_record *record) {
  if (capstone_park_test_hook)
    capstone_park_test_hook(point, record);
}

int capstone_park_init(struct capstone_park *park, _Atomic uint64_t *gen, unsigned buckets) {
  if (!buckets || (buckets & (buckets - 1))) {
    errno = EINVAL;
    return -1;
  }
  park->queues = calloc(buckets, sizeof *park->queues);
  if (!park->queues)
    return -1;
  park->gen = gen;
  park->buckets = buckets;
  pthread_mutex_init(&park->lock, NULL);
  return 0;
}

void capstone_park_destroy(struct capstone_park *park) {
  pthread_mutex_destroy(&park->lock);
  free(park->queues);
  park->queues = NULL;
}

unsigned capstone_park_bucket(const struct capstone_park *park, uint64_t key) {
  return capstone_park_bucket_of(key, park->buckets);
}

static void enqueue(struct capstone_park *park, unsigned bucket, struct capstone_park_record *r) {
  struct capstone_park_queue *q = &park->queues[bucket];
  r->bucket = bucket;
  r->next = NULL;
  r->prev = q->tail;
  if (q->tail)
    q->tail->next = r;
  else
    q->head = r;
  q->tail = r;
}

static void dequeue(struct capstone_park *park, struct capstone_park_record *r) {
  struct capstone_park_queue *q = &park->queues[r->bucket];
  if (r->prev)
    r->prev->next = r->next;
  else
    q->head = r->next;
  if (r->next)
    r->next->prev = r->prev;
  else
    q->tail = r->prev;
  r->next = r->prev = NULL;
}

/* With the mutex held. The waker's last access to the record is the futex
 * wake; the waiter marks the record IDLE only under the mutex, after it. */
static void complete(struct capstone_park *park, struct capstone_park_record *r, int outcome) {
  dequeue(park, r);
  r->outcome = outcome;
  r->state = CAPSTONE_PARK_NOTIFIED;
  atomic_store_explicit(&r->notified, 1, memory_order_release);
  hook(CAPSTONE_PARK_COMPLETING, r);
  syscall(SYS_futex, &r->notified, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1, NULL, NULL, 0);
}

static int saturated(struct capstone_park *park, unsigned bucket) {
  return atomic_load_explicit(&park->gen[bucket], memory_order_relaxed) == UINT64_MAX;
}

/* With the mutex held. Returns whether the bucket is saturated now. */
static int advance(struct capstone_park *park, unsigned bucket) {
  uint64_t g = atomic_load_explicit(&park->gen[bucket], memory_order_relaxed);
  if (g != UINT64_MAX)
    atomic_store_explicit(&park->gen[bucket], ++g, memory_order_release);
  return g == UINT64_MAX;
}

/* With the mutex held: every record still queued in bucket completes as RECHECK. */
static void flush(struct capstone_park *park, unsigned bucket) {
  while (park->queues[bucket].head)
    complete(park, park->queues[bucket].head, CAPSTONE_PARK_RECHECK);
}

static long futex_sleep(void *context, _Atomic uint32_t *word, const struct timespec *deadline) {
  (void)context;
  /* FUTEX_WAIT_BITSET takes an absolute CLOCK_MONOTONIC deadline, so
     spurious wakeups and retries keep the original budget. */
  if (syscall(SYS_futex, word, FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG, 0, deadline, NULL,
              FUTEX_BITSET_MATCH_ANY) == -1)
    return -errno;
  return 0;
}

enum capstone_park_outcome capstone_park_wait(struct capstone_park *park,
                                              struct capstone_park_record *record,
                                              uint64_t key, uint64_t gen,
                                              const struct timespec *deadline) {
  return capstone_park_wait_with(park, record, key, gen, deadline, futex_sleep, NULL);
}

enum capstone_park_outcome capstone_park_wait_with(struct capstone_park *park,
                                                   struct capstone_park_record *record,
                                                   uint64_t key, uint64_t gen,
                                                   const struct timespec *deadline,
                                                   capstone_park_sleep_fn sleep, void *context) {
  unsigned bucket = capstone_park_bucket(park, key);
  int aborted = 0, outcome;
  pthread_mutex_lock(&park->lock);
  if (saturated(park, bucket) ||
      atomic_load_explicit(&park->gen[bucket], memory_order_relaxed) != gen) {
    pthread_mutex_unlock(&park->lock);
    return CAPSTONE_PARK_RECHECK;
  }
  record->key = key;
  record->outcome = -1;
  record->state = CAPSTONE_PARK_QUEUED;
  atomic_store_explicit(&record->notified, 0, memory_order_relaxed);
  enqueue(park, bucket, record);
  pthread_mutex_unlock(&park->lock);
  hook(CAPSTONE_PARK_ENQUEUED, record);
  while (!atomic_load_explicit(&record->notified, memory_order_acquire)) {
    hook(CAPSTONE_PARK_SLEEPING, record);
    long r = sleep(context, &record->notified, deadline);
    if (r == -ETIMEDOUT || r == -EINTR || r == CAPSTONE_PARK_SLEEP_RETRY) {
      aborted = r == CAPSTONE_PARK_SLEEP_RETRY ? -1 : (int)-r;
      break;
    }
  }
  if (aborted)
    hook(CAPSTONE_PARK_ABORTING, record);
  pthread_mutex_lock(&park->lock);
  /* A notification wins over the timeout or signal that ended the sleep: the
     waker has already dequeued the record and counted it. */
  if (atomic_load_explicit(&record->notified, memory_order_acquire)) {
    outcome = record->outcome;
  } else {
    dequeue(park, record);
    record->state = CAPSTONE_PARK_ABORTED;
    outcome = aborted == ETIMEDOUT ? CAPSTONE_PARK_TIMEOUT
            : aborted == EINTR ? CAPSTONE_PARK_EINTR : CAPSTONE_PARK_RETRY;
  }
  record->state = CAPSTONE_PARK_IDLE;
  pthread_mutex_unlock(&park->lock);
  return outcome;
}

unsigned capstone_park_wake(struct capstone_park *park, uint64_t key, unsigned n) {
  unsigned bucket = capstone_park_bucket(park, key), count = 0;
  pthread_mutex_lock(&park->lock);
  int full = advance(park, bucket);
  struct capstone_park_record *r = park->queues[bucket].head, *next;
  for (; r && count < n; r = next) {
    next = r->next;
    if (r->key == key) {
      complete(park, r, CAPSTONE_PARK_WOKEN);
      ++count;
    }
  }
  if (full)
    flush(park, bucket);
  pthread_mutex_unlock(&park->lock);
  return count;
}

unsigned capstone_park_requeue(struct capstone_park *park, uint64_t src, uint64_t dst,
                               unsigned nwake, unsigned nmove) {
  unsigned from = capstone_park_bucket(park, src), to = capstone_park_bucket(park, dst);
  unsigned woken = 0, moved = 0;
  pthread_mutex_lock(&park->lock);
  int full = advance(park, from);
  /* Visit the records queued before this call only: a record moved to the
     tail of the same bucket is not looked at again. */
  struct capstone_park_record *r = park->queues[from].head, *last = park->queues[from].tail, *next;
  for (; r && (woken < nwake || moved < nmove); r = next) {
    int end = r == last;
    next = r->next;
    if (r->key == src) {
      if (woken < nwake) {
        complete(park, r, CAPSTONE_PARK_WOKEN);
        ++woken;
      } else {
        if (saturated(park, to)) {
          complete(park, r, CAPSTONE_PARK_RECHECK);
        } else {
          dequeue(park, r);
          r->key = dst;
          enqueue(park, to, r);
        }
        ++moved;
      }
    }
    if (end)
      break;
  }
  if (full)
    flush(park, from);
  pthread_mutex_unlock(&park->lock);
  return woken + moved;
}

unsigned capstone_park_queued(struct capstone_park *park, unsigned bucket, uint64_t key) {
  unsigned n = 0;
  pthread_mutex_lock(&park->lock);
  for (struct capstone_park_record *r = park->queues[bucket].head; r; r = r->next)
    n += r->key == key;
  pthread_mutex_unlock(&park->lock);
  return n;
}
