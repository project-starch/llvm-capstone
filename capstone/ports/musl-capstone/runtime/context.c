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
#define _GNU_SOURCE /* the CLONE_ flags */
#include <errno.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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
long __capstone_delegate_ints4(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d);
int __capstone_delegate_transport(unsigned long index);
unsigned long __capstone_context_run(void *start_block);
uint64_t __capstone_park_key(volatile int *word);
void *__capstone_signals_detach(void);
int __capstone_context_tid(void);
void __capstone_hc_note_unserved(long n);
_Noreturn void __capstone_context_exit_clear(unsigned long value, volatile int *clear);

#if defined(CAPSTONE_GP_CAPTABLE_ABI) && CAPSTONE_GP_CAPTABLE_ABI
/* gp-captable (B1.3, docs/plans/b0-silicon-delegated-runtime.md): a seal's PC is the code capability the monitor
   parked for this application, which the glue stores in __capstone_silicon_code_cap at the first entry, with its
   cursor moved to the entry. C cannot name __capstone_context_entry itself: it is an assembly label, and under
   gp-captable such a reference is derived from gp and faults on silicon (C-13). NULL when nothing was parked:
   minting then fails rather than sealing an integer PC, which silicon would run with no PC-capability check. */
void *__capstone_silicon_entry_cap(void *code);
extern void *__capstone_silicon_code_cap;
static void *context_entry(void)
{
  void *code = __capstone_silicon_code_cap;
  unsigned long type;
  /* LCC's type query is total on silicon and capstone-qemu and answers cap_type - 1: 7 is NOT_CAP. */
  __asm__ volatile ("lcc %0, %1, 1" : "=r"(type) : "r"(code));
  if (type == 7)
    return 0;
  return __capstone_silicon_entry_cap(code);
}
#define CONTEXT_ENTRY context_entry()
#else
#define CONTEXT_ENTRY ((void *)__capstone_context_entry)
#endif

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
  if (!entry)
    return -1;
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
  return mint_area(c, &area, CONTEXT_ENTRY, start, arg, 0, CAPSTONE_CONTEXT_MSTATUS, 0);
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
  return mint_area(c, &area, CONTEXT_ENTRY, start, arg, 0, mstatus, mie);
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
  return mint_area(c, &area, CONTEXT_ENTRY, start, arg, 1, CAPSTONE_CONTEXT_MSTATUS, 0);
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
  return mint_area(c, &area, CONTEXT_ENTRY, start, arg, 0, CAPSTONE_CONTEXT_MSTATUS, 0);
}

/* The first function a minted context runs (the entry glue calls it with its
   start block): the transport its creator reserved, then the application's
   function, whose value the entry glue passes on to __capstone_context_exit. */
/* Set in every context the runtime minted. The first context's stack is the
   monitor's and its TLS block the runtime's first allocation; neither is ever
   freed. */
static __thread int minted;

unsigned long __capstone_context_run(void *start_block)
{
  minted = 1;
  void **slot = (void **)start_block;
  unsigned long *word = (unsigned long *)start_block;
  unsigned long transport = word[CAPSTONE_CONTEXT_WORD_TRANSPORT / 8];
  if (transport)
    __capstone_delegate_transport(transport);   /* a bad index: every call gets -EIO */
  unsigned long (*start)(void *) =
      (unsigned long (*)(void *))slot[CAPSTONE_CONTEXT_SLOT_USER_START / 16];
  unsigned long value = start(slot[CAPSTONE_CONTEXT_SLOT_USER_ARG / 16]);
  free(__capstone_signals_detach());   /* returning ends the context */
  return value;
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

/* Offer and create; a THREAD context gets its transport first. musl's own
   threads (__clone below) come here directly: pthread_create has already
   switched musl's locks on. */
static long create(struct capstone_context *c, unsigned mode)
{
  unsigned long ticket = next_ticket++;
  long transport = 0;
  if (mode == CAPSTONE_CONTEXT_THREAD) {
    /* Before the request, not after it: the launcher may start the context
       before this context has the request's answer. */
    transport = __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_RESERVE, 0, 0, 0);
    if (transport < 0)
      return transport;
    c->start[CAPSTONE_CONTEXT_WORD_TRANSPORT / 8] = (unsigned long)transport;
  }
  /* The request consumes the reservation whatever its outcome, so it is made
     even when there is nothing to offer; it then fails to adopt. */
  /* the thread identity the context's struct pthread carries: tkill names it */
  long tid = c->tp ? ((struct pthread *)((char *)c->tp - sizeof(struct pthread)))->tid : 0;
  int offered = __capstone_context_offer(&c->seal, ticket) == 0;
  long id = __capstone_delegate_ints4(CAPSTONE_NR_CONTEXT_CREATE, ticket, mode,
                                      (unsigned long)transport, (unsigned long)tid);
  if (!offered)
    return id < 0 ? id : -EINVAL;
  if (id >= 0)
    c->id = (unsigned long)id;
  return id;
}

long capstone_context_create(struct capstone_context *c, unsigned mode)
{
  if (mode == CAPSTONE_CONTEXT_THREAD)
    threads_begin();
  return create(c, mode);
}

long capstone_context_step(unsigned long id, struct capstone_context_event *event)
{
  return __capstone_delegate_context(CAPSTONE_NR_CONTEXT_STEP, id, 0, event);
}

long capstone_context_forget(unsigned long id)
{
  return __capstone_delegate_context(CAPSTONE_NR_CONTEXT_FORGET, id, 0, 0);
}

/* musl's threads (docs/plans/delegation-threads.md, T4).
 *
 * musl's pthread_create builds the new thread's stack, TLS block and struct
 * pthread in a mapping of its own and calls __clone for the rest, as on
 * Linux; its join, detach and exit are musl's own too. __clone mints a context
 * whose area is only a seal region and a start block, entered on musl's stack
 * with musl's thread pointer, and runs it on a launcher thread (THREAD mode).
 *
 * A thread's end, in order (SYS_exit below, then __capstone_context_exit_clear):
 *   1. CONTEXT_EXITING tells the launcher the context is ending and which
 *      word to wake once it has returned for good;
 *   2. the CLONE_CHILD_CLEARTID word (musl's thread-list lock) becomes 0:
 *      from here a joiner may free musl's mapping, the stack and TLS included;
 *   3. the completion word becomes 1: from here the area may be revoked;
 *   4. the context returns EXITED, and the launcher frees its transport and
 *      wakes the word from 1 (as Linux does after clearing it).
 * Nothing runs on the stack or TLS after 2; after 3 the area sees only the
 * final switch's own write of the seal region.
 *
 * Areas are revoked and kept, never returned to the arena: a record per area,
 * outside every area, holds its handle. The reaper revokes each area whose
 * context has published completion, at the next __clone; a detached thread's
 * mapping (__unmapself) goes back to the heap then too. So a finished,
 * unreaped context costs one area and at most one mapping until the next
 * thread is made. */
#define CLONE_AREA_BYTES (CAPSTONE_CONTEXT_SEAL_BYTES + CAPSTONE_CONTEXT_START_BYTES)
#define CLONE_REQUIRED (CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SETTLS)
#define CLONE_ALLOWED (CLONE_REQUIRED | CLONE_SYSVSEM | CLONE_PARENT_SETTID | CLONE_CHILD_SETTID | \
                       CLONE_CHILD_CLEARTID | CLONE_DETACHED)

struct clone_record {
  struct capstone_context c;   /* the area's handle stays here between threads */
  int (*func)(void *);
  void *arg;
  volatile int *clear;         /* CLONE_CHILD_CLEARTID */
  void *signal_events;         /* the ended thread's signal list, for the reaper to free */
  void *unmap_base;            /* __unmapself: the mapping the reaper frees */
  size_t unmap_size;
  int live;                    /* created, and not yet reaped */
  struct clone_record *next;
};

static struct clone_record *clones;   /* every record made; none is freed */
static volatile int clones_lock;      /* before the arena, the maps and the heap */
/* Threads __clone made that have not reached their end, and whether the first
   context has ended: atomics, not clones_lock, because an ending thread may
   take no musl lock after its pthread_exit announced it (need_locks, below). */
static int clones_running, first_ended, first_status;
static __thread struct clone_record *self_clone;
static __thread volatile int *clear_tid;

static void reap_held(void)
{
  for (struct clone_record *r = clones; r; r = r->next) {
    if (!r->live || !__atomic_load_n(r->c.done, __ATOMIC_ACQUIRE))
      continue;
    capstone_context_revoke(&r->c);
    if (r->unmap_base)
      __munmap(r->unmap_base, r->unmap_size);
    free(r->signal_events);
    r->signal_events = 0;
    r->unmap_base = 0;
    r->live = 0;
  }
}

/* Mint into *area: seal region and start block only. */
static void mint_clone(struct capstone_context *c, capstone_cap_slot *area, void *stack,
                       void *tp, unsigned long (*start)(void *), void *arg)
{
  unsigned long base = capstone_cap_base(area);
  unsigned long end = capstone_cap_end(area);
  unsigned long start_at = base + CAPSTONE_CONTEXT_SEAL_BYTES;
  capstone_cap_slot start_s = {0};
  capstone_cap_make_handle(area, &c->handle);
  capstone_cap_split(area, start_at, &start_s);
  unsigned long *sb = capstone_cap_delinearize(&start_s);
  memset(sb, 0, CAPSTONE_CONTEXT_START_BYTES);
  void **slot = (void **)sb;
  slot[CAPSTONE_CONTEXT_SLOT_SP / 16] = stack;
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
  c->stack_base = c->stack_top = 0;
  __capstone_context_seal(area, CONTEXT_ENTRY, sb, &c->seal, CAPSTONE_CONTEXT_MSTATUS, 0);
}

long __capstone_thread_exit(int status);
int __capstone_delegate_ready(void);

static unsigned long clone_start(void *p)
{
  struct clone_record *r = p;
  self_clone = r;
  clear_tid = r->clear;
  __capstone_thread_exit(r->func(r->arg));
  for (;;);   /* a minted context's end never returns */
}

int __clone(int (*func)(void *), void *stack, int flags, void *arg, ...)
{
  va_list ap;
  va_start(ap, arg);
  int *ptid = va_arg(ap, int *);
  void *tls = va_arg(ap, void *);
  int *ctid = va_arg(ap, int *);
  va_end(ap);
  /* A thread of this process, nothing else: fork and vfork are not contexts. */
  if ((flags & CLONE_REQUIRED) != CLONE_REQUIRED || (flags & ~CLONE_ALLOWED)) {
    __capstone_hc_note_unserved(SYS_clone);
    return -ENOSYS;
  }
  /* No code capability for the seal (gp-captable, see context_entry): no thread, rather than one without PCC. */
  if (!CONTEXT_ENTRY)
    return -ENOSYS;

  long r = -EAGAIN;
  struct clone_record *rec;
  capstone_lock(&clones_lock);
  reap_held();
  for (rec = clones; rec && rec->live; rec = rec->next)
    ;
  if (!rec && (rec = calloc(1, sizeof *rec))) {
    rec->next = clones;
    clones = rec;
  }
  capstone_cap_slot area = {0};
  if (!rec)
    goto out;
  if (capstone_cap_type(&rec->c.handle) == CAPSTONE_CAP_LINEAR)
    capstone_cap_move(&rec->c.handle, &area);
  else if (arena_take(CLONE_AREA_BYTES, &area))
    goto out;
  unsigned tid = __atomic_fetch_add(&next_tid, 1, __ATOMIC_RELAXED);
  if (tid < TID_FIRST || tid > TID_LAST) {
    capstone_cap_move(&area, &rec->c.handle);
    goto out;
  }
  rec->func = func;
  rec->arg = arg;
  rec->clear = flags & CLONE_CHILD_CLEARTID ? ctid : 0;
  rec->unmap_base = 0;
  if (flags & CLONE_PARENT_SETTID)
    *ptid = (int)tid;
  if (flags & CLONE_CHILD_SETTID)
    *ctid = (int)tid;
  mint_clone(&rec->c, &area, stack, tls, clone_start, rec);
  /* counted before it can run: it may end before CREATE answers */
  __atomic_fetch_add(&clones_running, 1, __ATOMIC_SEQ_CST);
  r = create(&rec->c, CAPSTONE_CONTEXT_THREAD);
  if (r < 0) {
    __atomic_fetch_sub(&clones_running, 1, __ATOMIC_SEQ_CST);
    capstone_context_revoke(&rec->c);   /* no context runs after a failure */
  } else {
    rec->live = 1;
  }
out:
  capstone_unlock(&clones_lock);
  return r < 0 ? (int)r : (int)tid;
}

/* A detached thread's last call (musl's pthread_exit): the mapping it runs on
   goes back to the heap once its context can never run again. */
_Noreturn void __unmapself(void *base, size_t size)
{
  if (self_clone) {
    self_clone->unmap_base = base;
    self_clone->unmap_size = size;
  }
  __capstone_thread_exit(0);
  for (;;);
}

/* set_tid_address: the word SYS_exit clears (musl passes its thread-list lock). */
long __capstone_set_tid_address(volatile int *word)
{
  clear_tid = word;
  return __capstone_context_tid();
}

static volatile int never_cleared, forever;

/* Test only (pthread-probe): runs in a further context's end after
   CONTEXT_EXITING and before the clear word is released, to hold the context
   there across quanta (B10, and a reservation that waits for it). */
void (*__capstone_thread_exit_test_gap)(void);

/* Records made, and those whose context is created and not yet reaped. */
void __capstone_clone_stats(unsigned *made, unsigned *live)
{
  unsigned m = 0, l = 0;
  capstone_lock(&clones_lock);
  for (struct clone_record *r = clones; r; r = r->next) {
    ++m;
    l += r->live;
  }
  capstone_unlock(&clones_lock);
  *made = m;
  *live = l;
}

/* SYS_exit: the calling context ends, the others go on.
 *
 * musl's pthread_exit sets need_locks to -1 when the thread count reaches 0,
 * and the next musl lock then switches locking off for good: an ending thread
 * must take no musl lock (capstone_lock included) after that, while another
 * thread may already be making a third. So this path uses atomics only.
 *
 * The first context's stack and TLS are never freed: it clears and wakes its
 * word itself and then waits for good while a thread lives. The last musl
 * thread's pthread_exit finds itself alone and calls exit(0), as on Linux;
 * when the last thread ends with SYS_exit instead, the application ends with
 * the first context's status, as Linux reports the group leader's. Alone, the
 * first context ends the application with its own. Returns -EIO only in a
 * first context whose transport is not there yet (musl calls again). */
long __capstone_thread_exit(int status)
{
  volatile int *clear = clear_tid;
  if (minted) {
    if (self_clone && __atomic_sub_fetch(&clones_running, 1, __ATOMIC_SEQ_CST) == 0 &&
        __atomic_load_n(&first_ended, __ATOMIC_SEQ_CST))
      _Exit(__atomic_load_n(&first_status, __ATOMIC_RELAXED));
    /* No more signals here; the list is freed where a lock may be taken: by
       the reaper for a musl thread (which may take no musl lock now, above),
       at once for a context of the runtime's own interface. */
    void *signal_events = __capstone_signals_detach();
    if (self_clone)
      self_clone->signal_events = signal_events;
    else
      free(signal_events);
    __capstone_delegate_ints(CAPSTONE_NR_CONTEXT_EXITING, clear ? __capstone_park_key(clear) : 0, 0, 0);
    if (__capstone_thread_exit_test_gap)
      __capstone_thread_exit_test_gap();
    __capstone_context_exit_clear((unsigned long)status, clear ? clear : &never_cleared);
  }
  if (!__capstone_delegate_ready())
    return -EIO;
  if (clear) {
    __atomic_store_n(clear, 0, __ATOMIC_RELEASE);
    __syscall(SYS_futex, clear, FUTEX_WAKE | FUTEX_PRIVATE, 1);
  }
  __atomic_store_n(&first_status, status, __ATOMIC_RELAXED);
  __atomic_store_n(&first_ended, 1, __ATOMIC_SEQ_CST);
  if (!__atomic_load_n(&clones_running, __ATOMIC_SEQ_CST))
    _Exit(status);
  for (;;)
    __syscall(SYS_futex, &forever, FUTEX_WAIT | FUTEX_PRIVATE, 0, 0);
}

/* musl's __synccall runs a function in every thread with the others held,
 * by signalling each one; its callers are setuid, setgid, setgroups and
 * their relatives (setrlimit only when prlimit64 fails, and it is served).
 * A context takes a signal at its next delegated call, so one computing
 * without calls would hold every other for good. None of those calls is
 * served (they answer ENOSYS, reported); were one served, the launcher would
 * have to apply it to each of its own threads, since Linux keeps credentials
 * per thread, not the domain to each context. So the function runs once, in
 * the caller. The thread-list lock's weak stand-ins are synccall.o's, for a
 * program that makes no thread. */
void __synccall(void (*func)(void *), void *ctx)
{
  func(ctx);
}

static void no_list_lock(void) {}
__attribute__((__weak__, __alias__("no_list_lock"))) void __tl_lock(void);
__attribute__((__weak__, __alias__("no_list_lock"))) void __tl_unlock(void);
