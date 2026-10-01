/* Minted execution contexts (docs/plans/delegation-threads.md, Probe A).
 *
 * The application's context arena is one LINEAR capability that _start splits
 * off the data region (start-musl.S) when the image declares CONTEXT_BYTES.
 * A context takes one thread area from it:
 *
 *   [ seal region | start block | TLS block | stack ]
 *
 * The parent handle, made before any split, stays in the caller's
 * struct capstone_context, outside the area. Revoking it kills every
 * capability derived from the area: the start block, TLS and stack aliases,
 * and the seal, wherever it is held. The area then sits in the handle slot
 * again, ready to be minted anew.
 *
 * The seal region stays linear until it is sealed; the other three children
 * are delinearized into aliases, which the handle still covers. Linear values
 * never sit in C variables here, only in capstone_cap_slot records. */
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include "pthread_impl.h"
#include "stdio_impl.h"
#include <capstone/capability.h>
#include <capstone/lock.h>
#include <capstone/context.h>
#include <capstone/delegate.h>

/* Every image links this; an application with CONTEXT_BYTES links domreq.S's
   strong definition instead. start-musl.S reads it at the first entry. */
__attribute__((__weak__)) const unsigned long __capstone_context_arena_bytes = 0;

extern capstone_cap_slot __capstone_context_arena;
extern char __capstone_context_entry[];
size_t __capstone_tls_block_bytes(void);
char *__capstone_tls_block_init(char *mem, size_t bytes);
void __capstone_context_seal(capstone_cap_slot *region, void *entry,
                             void *start_block, capstone_cap_slot *out,
                             unsigned long mstatus, unsigned long mie);
long __capstone_context_offer(capstone_cap_slot *seal, unsigned long ticket);
long __capstone_delegate_context(uint64_t nr, uint64_t a, uint64_t b, void *event);
long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c);
int __capstone_delegate_transport(unsigned long index);
unsigned long __capstone_context_run(void *start_block);

#define MIN_STACK_BYTES 8192

static size_t tls_bytes(void)
{
  return (__capstone_tls_block_bytes() + 15) & ~(size_t)15;
}

/* Several contexts may mint at once (capstone/lock.h, Q6). */
static volatile int arena_lock;

/* Move the front `bytes` of the arena into *out. */
static int arena_take_held(size_t bytes, capstone_cap_slot *out)
{
  if (capstone_cap_type(&__capstone_context_arena) != CAPSTONE_CAP_LINEAR)
    return -1;
  unsigned long base = capstone_cap_base(&__capstone_context_arena);
  unsigned long end = capstone_cap_end(&__capstone_context_arena);
  if (end - base < bytes)
    return -1;
  if (end - base == bytes) {
    capstone_cap_move(&__capstone_context_arena, out);
    return 0;
  }
  capstone_cap_slot rest = {0};
  capstone_cap_split(&__capstone_context_arena, base + bytes, &rest);
  capstone_cap_move(&__capstone_context_arena, out);
  capstone_cap_move(&rest, &__capstone_context_arena);
  return 0;
}

static int arena_take(size_t bytes, capstone_cap_slot *out)
{
  capstone_lock(&arena_lock);
  int r = arena_take_held(bytes, out);
  capstone_unlock(&arena_lock);
  return r;
}

/* Thread identities of minted contexts: from 2^22 up, above Linux's pid range
   (pids stay below PID_MAX_LIMIT, 2^22), so none is a pid; below 0x3fffffff,
   since musl keeps a tid in 30 bits of a lock word (bit 30 is MAYBE_WAITERS,
   0x3fffffff putc's marker and a mutex's "not recoverable"); and never reused
   in the process's life, so a recursive lock never takes a new context for an
   old owner (Q2). When they run out, minting fails. */
#define TID_FIRST 0x400000u
#define TID_LAST 0x3ffffffeu
static unsigned next_tid = TID_FIRST;

/* Mint into the linear area in *area; the handle is made here, senior to
 * every split below, and ends up in c->handle. */
static int mint_area(struct capstone_context *c, capstone_cap_slot *area, void *entry,
                     unsigned long (*start)(void *), void *arg, int split,
                     unsigned long mstatus, unsigned long mie)
{
  size_t tls = tls_bytes();
  unsigned long base = capstone_cap_base(area);
  unsigned long end = capstone_cap_end(area);
  unsigned long start_at = base + CAPSTONE_CONTEXT_SEAL_BYTES;
  unsigned long tls_at = start_at + CAPSTONE_CONTEXT_START_BYTES;
  unsigned long stack_at = tls_at + tls;
  if (end < stack_at + MIN_STACK_BYTES)
    return -1;

  capstone_cap_slot start_s = {0}, tls_s = {0}, stack_s = {0};
  capstone_cap_make_handle(area, &c->handle);
  capstone_cap_split(area, start_at, &start_s);
  capstone_cap_split(&start_s, tls_at, &tls_s);
  capstone_cap_split(&tls_s, stack_at, &stack_s);
  if (split) {
    capstone_cap_make_handle(&start_s, &c->child[0]);
    capstone_cap_make_handle(&tls_s, &c->child[1]);
    capstone_cap_make_handle(&stack_s, &c->child[2]);
  }
  unsigned long *sb = capstone_cap_delinearize(&start_s);
  char *tls_block = capstone_cap_delinearize(&tls_s);
  char *stack = capstone_cap_delinearize(&stack_s);

  memset(sb, 0, CAPSTONE_CONTEXT_START_BYTES);
  char *tp = __capstone_tls_block_init(tls_block, tls);
  unsigned tid = __atomic_fetch_add(&next_tid, 1, __ATOMIC_RELAXED);
  if (!tp || tid < TID_FIRST || tid > TID_LAST)
    return -1;
  ((struct pthread *)(tp - sizeof(struct pthread)))->tid = (int)tid;
  char *top = stack + (end - stack_at);
  void **slot = (void **)sb;
  slot[CAPSTONE_CONTEXT_SLOT_SP / 16] = top;
  slot[CAPSTONE_CONTEXT_SLOT_TP / 16] = tp;
  slot[CAPSTONE_CONTEXT_SLOT_START / 16] = (void *)__capstone_context_run;
  slot[CAPSTONE_CONTEXT_SLOT_ARG / 16] = sb;
  slot[CAPSTONE_CONTEXT_SLOT_USER_START / 16] = (void *)start;
  slot[CAPSTONE_CONTEXT_SLOT_USER_ARG / 16] = arg;

  c->done = (volatile unsigned long *)(sb + CAPSTONE_CONTEXT_WORD_DONE / 8);
  c->value = (volatile unsigned long *)(sb + CAPSTONE_CONTEXT_WORD_VALUE / 8);
  c->start = sb;
  c->tp = tp;
  c->area_base = base;
  c->area_bytes = end - base;
  c->stack_base = stack_at;
  c->stack_top = end;
  __capstone_context_seal(area, entry, sb, &c->seal, mstatus, mie);
  return 0;
}

int capstone_context_mint(struct capstone_context *c, size_t area_bytes,
                          unsigned long (*start)(void *), void *arg)
{
  if (area_bytes & 15)
    return -1;
  memset(c, 0, sizeof *c);
  capstone_cap_slot area = {0};
  if (arena_take(area_bytes, &area))
    return -1;
  return mint_area(c, &area, __capstone_context_entry, start, arg, 0, CAPSTONE_CONTEXT_MSTATUS, 0);
}

int capstone_context_mint_words(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg,
                                unsigned long mstatus, unsigned long mie)
{
  if (area_bytes & 15)
    return -1;
  memset(c, 0, sizeof *c);
  capstone_cap_slot area = {0};
  if (arena_take(area_bytes, &area))
    return -1;
  return mint_area(c, &area, __capstone_context_entry, start, arg, 0, mstatus, mie);
}

int capstone_context_mint_entry(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg, void *entry)
{
  if (area_bytes & 15)
    return -1;
  memset(c, 0, sizeof *c);
  capstone_cap_slot area = {0};
  if (arena_take(area_bytes, &area))
    return -1;
  return mint_area(c, &area, entry, start, arg, 0, CAPSTONE_CONTEXT_MSTATUS, 0);
}

int capstone_context_mint_split(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg)
{
  if (area_bytes & 15)
    return -1;
  memset(c, 0, sizeof *c);
  capstone_cap_slot area = {0};
  if (arena_take(area_bytes, &area))
    return -1;
  return mint_area(c, &area, __capstone_context_entry, start, arg, 1, CAPSTONE_CONTEXT_MSTATUS, 0);
}

void capstone_context_revoke_children(struct capstone_context *c)
{
  for (int i = 0; i < 3; ++i)
    capstone_cap_revoke(&c->child[i]);
}

void capstone_context_revoke(struct capstone_context *c)
{
  capstone_cap_revoke(&c->handle);
  if (capstone_cap_type(&c->handle) == CAPSTONE_CAP_UNINITIALIZED)
    capstone_cap_initialize_zero(&c->handle);
  c->done = c->value = 0;
  c->start = 0;
  c->tp = 0;
}

int capstone_context_remint(struct capstone_context *c,
                            unsigned long (*start)(void *), void *arg)
{
  if (capstone_cap_type(&c->handle) != CAPSTONE_CAP_LINEAR)
    return -1;
  capstone_cap_slot area = {0};
  capstone_cap_move(&c->handle, &area);
  capstone_cap_clear(&c->seal);
  return mint_area(c, &area, __capstone_context_entry, start, arg, 0, CAPSTONE_CONTEXT_MSTATUS, 0);
}

/* The first function a minted context runs (the entry glue calls it with its
   start block): the transport its creator reserved, then the application's
   function, whose value the entry glue passes on to __capstone_context_exit. */
unsigned long __capstone_context_run(void *start_block)
{
  void **slot = (void **)start_block;
  unsigned long *word = (unsigned long *)start_block;
  unsigned long transport = word[CAPSTONE_CONTEXT_WORD_TRANSPORT / 8];
  if (transport)
    __capstone_delegate_transport(transport);   /* a bad index: every call gets -EIO */
  unsigned long (*start)(void *) =
      (unsigned long (*)(void *))slot[CAPSTONE_CONTEXT_SLOT_USER_START / 16];
  return start(slot[CAPSTONE_CONTEXT_SLOT_USER_ARG / 16]);
}

/* One ticket per offer of this context: a late or repeated request can never
   consume a later offer. The monitor keeps offers per offering context, so the
   count is the context's own and needs no atomic. */
static __thread unsigned long next_ticket = 1;

/* musl's own switch at its first pthread_create, before a second context can
   run: stdio locks every FILE from now on (its lock word leaves -1), and
   libc.need_locks turns musl's internal locks and the runtime's on. The
   count only rises: a context's end is not seen here (Q4), so the locks stay
   on, which is correct and only slower. */
#pragma weak __ofl_lock
#pragma weak __ofl_unlock
#pragma weak __stdin_used
#pragma weak __stdout_used
#pragma weak __stderr_used
static void lock_file(FILE *volatile *used)
{
  if (used && *used && (*used)->lock < 0)
    (*used)->lock = 0;
}

static void threads_begin(void)
{
  if (!libc.threaded) {
    if (__ofl_lock) {
      for (FILE *f = *__ofl_lock(); f; f = f->next)
        if (f->lock < 0)
          f->lock = 0;
      __ofl_unlock();
    }
    lock_file(&__stdin_used);
    lock_file(&__stdout_used);
    lock_file(&__stderr_used);
    libc.threaded = 1;
  }
  if (!__atomic_fetch_add(&libc.threads_minus_1, 1, __ATOMIC_RELAXED))
    libc.need_locks = 1;
}

long capstone_context_create(struct capstone_context *c, unsigned mode)
{
  unsigned long ticket = next_ticket++;
  long transport = 0;
  if (mode == CAPSTONE_CONTEXT_THREAD) {
    threads_begin();
    /* Before the request, not after it: the launcher may start the context
       before this context has the request's answer. */
    transport = __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0);
    if (transport < 0)
      return transport;
    c->start[CAPSTONE_CONTEXT_WORD_TRANSPORT / 8] = (unsigned long)transport;
  }
  /* The request consumes the reservation whatever its outcome, so it is made
     even when there is nothing to offer; it then fails to adopt. */
  int offered = __capstone_context_offer(&c->seal, ticket) == 0;
  long id = __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_CREATE, ticket, mode,
                                     (unsigned long)transport);
  if (!offered)
    return id < 0 ? id : -EINVAL;
  if (id >= 0)
    c->id = (unsigned long)id;
  return id;
}

long capstone_context_step(unsigned long id, struct capstone_context_event *event)
{
  return __capstone_delegate_context(CAPSTONE_NR_CONTEXT_STEP, id, 0, event);
}

long capstone_context_forget(unsigned long id)
{
  return __capstone_delegate_context(CAPSTONE_NR_CONTEXT_FORGET, id, 0, 0);
}
