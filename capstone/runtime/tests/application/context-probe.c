/* Probe A, domain half (docs/plans/delegation-threads.md).
 *
 * The domain mints a context over a thread area from its context arena and
 * enters it with a NESTED, unsupervised CALL from the main context
 * (__capstone_context_call); no monitor, driver or launcher change is
 * involved. That checks the seal layout, the entry, the exit and re-entry
 * paths and the revoke/remint cycle on the unchanged platform, before the
 * monitor adopts minted seals.
 *
 * A child entered this way must not make a delegated call: its yield would
 * return to this caller, not to the launcher. The start functions below
 * compute and store only.
 *
 * Every mode prints "context-probe <mode>: PASS" and exits 0, or fails the
 * CHECK naming the broken property. revoked-call is a negative mode: it must
 * end in a domain fault, and prints REACHED if it does not. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <capstone/capability.h>
#include <capstone/context.h>
#include <capstone/delegate.h>
#include <errno.h>
#include <sched.h>

/* The driver's step events (package/modcapstone include/capstone.h). */
#define STEP_RETURNED 0
#define STEP_PREEMPTED 1
#define STEP_FAULT 2
#define STEP_DEAD 3
#define STEP_STALE 4

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "context-probe:%d: %s\n", __LINE__, #test); return 1; \
} } while (0)

#define AREA_BYTES 65536

static const char *mode;
static struct capstone_context ctx;

static volatile unsigned long counter;
static volatile uintptr_t child_local, child_tls, parent_local;
static __thread volatile unsigned long tls_word = 3;
static volatile int ctor_runs;
static int target_global = 5;
static int *const cap_global = &target_global;   /* a capability-global initializer */
static volatile int child_cap_global;

__attribute__((constructor)) static void count_ctor(void) { ctor_runs++; }

static unsigned long child_enter(void *arg)
{
  volatile int local = 0;
  counter += (unsigned long)(uintptr_t)arg;
  child_local = (uintptr_t)&local;
  child_tls = (uintptr_t)&tls_word;
  tls_word = 7;
  child_cap_global = *cap_global;
  return 42;
}

static unsigned long lcg_rounds(unsigned long seed, unsigned long rounds)
{
  unsigned long x = seed;
  for (unsigned long i = 0; i < rounds; ++i)
    x = x * 6364136223846793005ul + 1442695040888963407ul;
  return x;
}

#define PREEMPT_ROUNDS 40000000ul
static unsigned long child_preempt(void *arg)
{
  return lcg_rounds((unsigned long)(uintptr_t)arg, PREEMPT_ROUNDS);
}

static unsigned long child_second(void *arg)
{
  volatile int local = 0;
  child_local = (uintptr_t)&local;
  return 1000 + (unsigned long)(uintptr_t)arg;
}

static long monotonic_ms(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static int enter_once(unsigned long *result)
{
  *result = 0;
  __capstone_context_call(&ctx.seal, 0, result);
  return 0;
}

static int nested_enter(void)
{
  volatile int local = 0;
  unsigned long result;
  parent_local = (uintptr_t)&local;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  CHECK(*ctx.done == 0);
  enter_once(&result);
  CHECK(result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.done == 1);
  CHECK(*ctx.value == 42);
  CHECK(counter == 1);
  CHECK(child_local >= ctx.stack_base && child_local < ctx.stack_top);
  CHECK(parent_local < ctx.area_base || parent_local >= ctx.area_base + ctx.area_bytes);
  CHECK(child_tls != (uintptr_t)&tls_word);
  CHECK(tls_word == 3);
  CHECK(child_cap_global == 5);
  CHECK(ctor_runs == 1);
  return 0;
}

static int nested_preempt(void)
{
  unsigned long result;
  unsigned long seed = 0x1234;
  unsigned long want = lcg_rounds(seed, PREEMPT_ROUNDS);
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_preempt, (void *)(uintptr_t)seed));
  long t0 = monotonic_ms();
  enter_once(&result);
  long elapsed = monotonic_ms() - t0;
  CHECK(result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == want);
  printf("context-probe nested-preempt: child ran %ld ms\n", elapsed);
  CHECK(elapsed > 20);
  return 0;
}

static int nested_reenter(void)
{
  unsigned long result;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  enter_once(&result);
  CHECK(result == CAPSTONE_CONTEXT_EXITED);
  for (int i = 0; i < 3; ++i) {
    enter_once(&result);
    CHECK(result == CAPSTONE_CONTEXT_EXITED);
  }
  CHECK(counter == 1);
  CHECK(*ctx.value == 42);
  return 0;
}

static int remint(void)
{
  unsigned long result;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  unsigned long area = ctx.area_base;
  enter_once(&result);
  CHECK(result == CAPSTONE_CONTEXT_EXITED && *ctx.value == 42);
  capstone_context_revoke(&ctx);
  CHECK(!capstone_context_remint(&ctx, child_second, (void *)(uintptr_t)7));
  CHECK(ctx.area_base == area);
  CHECK(*ctx.done == 0);
  enter_once(&result);
  CHECK(result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == 1007);
  CHECK(child_local >= ctx.stack_base && child_local < ctx.stack_top);
  return 0;
}

/* Negative: the seal lives in ctx.seal, memory; after the revoke it must not
   be enterable. Expected: a domain fault at the CALL. */
static int revoked_call(void)
{
  unsigned long result;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  enter_once(&result);
  CHECK(result == CAPSTONE_CONTEXT_EXITED);
  capstone_context_revoke(&ctx);
  fflush(stdout);
  enter_once(&result);
  printf("context-probe revoked-call: REACHED result=%lx\n", result);
  return 1;
}

/* ---- Through the monitor: the launcher registers the offered seal (ADOPT)
 * and steps it; no nested CALL. ---- */

/* Step a registered context until it stops being preempted. */
static long step_to_end(unsigned long id, struct capstone_context_event *ev,
                        unsigned long *preemptions)
{
  long r;
  unsigned long n = 0;
  do {
    memset(ev, 0, sizeof *ev);
    r = capstone_context_step(id, ev);
    if (r)
      return r;
    if (ev->kind == STEP_PREEMPTED)
      ++n;
  } while (ev->kind == STEP_PREEMPTED);
  if (preemptions)
    *preemptions = n;
  return 0;
}

static int adopt_enter(void)
{
  struct capstone_context_event ev;
  volatile int local = 0;
  parent_local = (uintptr_t)&local;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(capstone_cap_type(&ctx.seal) == CAPSTONE_CAP_EMPTY);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED);
  CHECK(ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.done == 1 && *ctx.value == 42);
  CHECK(counter == 1);
  CHECK(child_local >= ctx.stack_base && child_local < ctx.stack_top);
  CHECK(child_tls != (uintptr_t)&tls_word && tls_word == 3);
  CHECK(child_cap_global == 5 && ctor_runs == 1);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

static int adopt_thread(void)
{
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_THREAD);
  CHECK(id > 0);
  long t0 = monotonic_ms();
  while (!*ctx.done && monotonic_ms() - t0 < 20000)
    sched_yield();
  CHECK(*ctx.done == 1 && *ctx.value == 42);
  CHECK(counter == 1);
  CHECK(child_local >= ctx.stack_base && child_local < ctx.stack_top);
  return 0;
}

static int adopt_preempt(void)
{
  struct capstone_context_event ev;
  unsigned long seed = 0x1234, preemptions = 0;
  unsigned long want = lcg_rounds(seed, PREEMPT_ROUNDS);
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_preempt, (void *)(uintptr_t)seed));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, &preemptions));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == want);
  printf("context-probe adopt-preempt: %lu preemptions\n", preemptions);
  CHECK(preemptions >= 2);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

static int adopt_reenter(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  for (int i = 0; i < 3; ++i) {
    CHECK(!step_to_end((unsigned long)id, &ev, 0));
    CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  }
  CHECK(counter == 1 && *ctx.value == 42);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* After the domain revokes an exited context's area, the monitor's copy of
   the seal is dead: STEP says DEAD, FORGET removes it once, and an old id is
   refused afterwards. The area is then reused for a new context. */
static int adopt_dead(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  unsigned long area = ctx.area_base;
  capstone_context_revoke(&ctx);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_DEAD);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_DEAD);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  CHECK(capstone_context_forget((unsigned long)id) == -ESTALE);
  CHECK(capstone_context_step((unsigned long)id, &ev) < 0);
  /* The same area, a new context. */
  CHECK(!capstone_context_remint(&ctx, child_second, (void *)(uintptr_t)7));
  CHECK(ctx.area_base == area);
  long id2 = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id2 > 0 && id2 != id);
  CHECK(!step_to_end((unsigned long)id2, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == 1007);
  CHECK(capstone_context_forget((unsigned long)id2) == 0);
  printf("context-probe adopt-dead: ids %lx then %lx\n", (unsigned long)id, (unsigned long)id2);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: context-probe MODE\n");
    return 2;
  }
  mode = argv[1];
  int rc;
  if (!strcmp(mode, "nested-enter")) rc = nested_enter();
  else if (!strcmp(mode, "nested-preempt")) rc = nested_preempt();
  else if (!strcmp(mode, "nested-reenter")) rc = nested_reenter();
  else if (!strcmp(mode, "remint")) rc = remint();
  else if (!strcmp(mode, "revoked-call")) rc = revoked_call();
  else if (!strcmp(mode, "adopt-enter")) rc = adopt_enter();
  else if (!strcmp(mode, "adopt-thread")) rc = adopt_thread();
  else if (!strcmp(mode, "adopt-preempt")) rc = adopt_preempt();
  else if (!strcmp(mode, "adopt-reenter")) rc = adopt_reenter();
  else if (!strcmp(mode, "adopt-dead")) rc = adopt_dead();
  else {
    fprintf(stderr, "context-probe: unknown mode %s\n", mode);
    return 2;
  }
  if (!rc)
    printf("context-probe %s: PASS\n", mode);
  return rc;
}
