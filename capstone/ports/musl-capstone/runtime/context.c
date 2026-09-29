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
#include <capstone/capability.h>
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
                             void *start_block, capstone_cap_slot *out);
long __capstone_context_offer(capstone_cap_slot *seal, unsigned long ticket);
long __capstone_delegate_context(uint64_t nr, uint64_t a, uint64_t b, void *event);

#define MIN_STACK_BYTES 8192

static size_t tls_bytes(void)
{
  return (__capstone_tls_block_bytes() + 15) & ~(size_t)15;
}

/* Move the front `bytes` of the arena into *out. */
static int arena_take(size_t bytes, capstone_cap_slot *out)
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

/* Mint into the linear area in *area; the handle is made here, senior to
 * every split below, and ends up in c->handle. */
static int mint_area(struct capstone_context *c, capstone_cap_slot *area,
                     unsigned long (*start)(void *), void *arg, int split)
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
  if (!tp)
    return -1;
  char *top = stack + (end - stack_at);
  void **slot = (void **)sb;
  slot[CAPSTONE_CONTEXT_SLOT_SP / 16] = top;
  slot[CAPSTONE_CONTEXT_SLOT_TP / 16] = tp;
  slot[CAPSTONE_CONTEXT_SLOT_START / 16] = (void *)start;
  slot[CAPSTONE_CONTEXT_SLOT_ARG / 16] = arg;

  c->done = (volatile unsigned long *)(sb + CAPSTONE_CONTEXT_WORD_DONE / 8);
  c->value = (volatile unsigned long *)(sb + CAPSTONE_CONTEXT_WORD_VALUE / 8);
  c->start = sb;
  c->tp = tp;
  c->area_base = base;
  c->area_bytes = end - base;
  c->stack_base = stack_at;
  c->stack_top = end;
  __capstone_context_seal(area, __capstone_context_entry, sb, &c->seal);
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
  return mint_area(c, &area, start, arg, 0);
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
  return mint_area(c, &area, start, arg, 1);
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
  return mint_area(c, &area, start, arg, 0);
}

/* One ticket per offer: a late or repeated request can never consume a later
   offer of this application. */
static unsigned long next_ticket = 1;

long capstone_context_create(struct capstone_context *c, unsigned mode)
{
  unsigned long ticket = next_ticket++;
  if (__capstone_context_offer(&c->seal, ticket))
    return -EINVAL;
  long id = __capstone_delegate_context(CAPSTONE_NR_CONTEXT_CREATE, ticket, mode, 0);
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
