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

/* ---- Probe-only helpers (context-probe-asm.S) ---- */
void probe_exit_delayed(unsigned long value, unsigned long loops) __attribute__((noreturn));
void *probe_descriptor(void);
void probe_store_through(unsigned long *where, unsigned long value);
extern char probe_exit_delay_begin[], probe_exit_delay_end[];
long __capstone_context_offer(capstone_cap_slot *seal, unsigned long ticket);
long __capstone_delegate_context(uint64_t nr, uint64_t a, uint64_t b, void *event);

#define DELAY_LOOPS 30000000ul
static unsigned long child_delayed(void *arg)
{
  probe_exit_delayed(4242, (unsigned long)(uintptr_t)arg);
}

/* A4: the start block, TLS block and stack are revoked while the seal stays
   valid; every later entry still reports EXITED from registers alone. */
static int reenter_revoked(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint_split(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == 42);
  capstone_context_revoke_children(&ctx);
  for (int i = 0; i < 3; ++i) {
    CHECK(!step_to_end((unsigned long)id, &ev, 0));
    CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  }
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* A6: the context is preempted between publishing done and its final switch.
   The joiner sees done, copies the value, revokes and reuses the area; the old
   context then steps DEAD and never runs again, and the new one is intact. */
static int done_preempted(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_delayed, (void *)(uintptr_t)DELAY_LOOPS));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  do {
    memset(&ev, 0, sizeof ev);
    CHECK(!capstone_context_step((unsigned long)id, &ev));
  } while (ev.kind == STEP_PREEMPTED && *ctx.done == 0);
  CHECK(ev.kind == STEP_PREEMPTED && *ctx.done == 1);
  CHECK(ev.pc >= (uintptr_t)probe_exit_delay_begin && ev.pc < (uintptr_t)probe_exit_delay_end);
  unsigned long value = *ctx.value;
  capstone_context_revoke(&ctx);
  CHECK(!capstone_context_remint(&ctx, child_second, (void *)(uintptr_t)9));
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_DEAD);
  long id2 = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id2 > 0);
  CHECK(!step_to_end((unsigned long)id2, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == 1009 && value == 4242);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  CHECK(capstone_context_forget((unsigned long)id2) == 0);
  return 0;
}

/* A11: the launcher fails to start the Linux thread (run with
   CAPSTONE_CONTEXT_TEST_THREAD_FAILS=1): the registration is taken back and
   no child runs. An old ticket cannot consume a newer offer. */
static int rollback_thread(void)
{
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  CHECK(capstone_context_create(&ctx, CAPSTONE_CONTEXT_THREAD) == -EAGAIN);
  for (int i = 0; i < 50; ++i)
    sched_yield();
  CHECK(counter == 0 && *ctx.done == 0);
  capstone_context_revoke(&ctx);
  CHECK(!capstone_context_remint(&ctx, child_enter, (void *)(uintptr_t)1));
  /* A new offer under ticket 1001; a request carrying the old ticket must not
     consume it, the right ticket then does. */
  CHECK(!__capstone_context_offer(&ctx.seal, 1001));
  CHECK(__capstone_delegate_context(CAPSTONE_NR_CONTEXT_CREATE, 1000, CAPSTONE_CONTEXT_REGISTER, 0) == -ESTALE);
  long id = __capstone_delegate_context(CAPSTONE_NR_CONTEXT_CREATE, 1001, CAPSTONE_CONTEXT_REGISTER, 0);
  CHECK(id > 0);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* A13: an offer is registered once; two launcher threads stepping one
   context serialise into one continuous execution. */
static int duplicate_adopt(void)
{
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  CHECK(!__capstone_context_offer(&ctx.seal, 77));
  long id = __capstone_delegate_context(CAPSTONE_NR_CONTEXT_CREATE, 77, CAPSTONE_CONTEXT_REGISTER, 0);
  CHECK(id > 0);
  CHECK(__capstone_delegate_context(CAPSTONE_NR_CONTEXT_CREATE, 77, CAPSTONE_CONTEXT_REGISTER, 0) == -ENOENT);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

static int two_steppers(void)
{
  struct capstone_context_event ev;
  unsigned long seed = 0x77, mine = 0;
  unsigned long want = lcg_rounds(seed, PREEMPT_ROUNDS);
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_preempt, (void *)(uintptr_t)seed));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_THREAD);
  CHECK(id > 0);
  while (!*ctx.done) {
    memset(&ev, 0, sizeof ev);
    if (capstone_context_step((unsigned long)id, &ev))
      break;
    if (ev.kind != STEP_PREEMPTED && ev.kind != STEP_RETURNED)
      break;
    ++mine;
  }
  long t0 = monotonic_ms();
  while (!*ctx.done && monotonic_ms() - t0 < 30000)
    sched_yield();
  CHECK(*ctx.done == 1 && *ctx.value == want);
  printf("context-probe two-steppers: %lu steps from the main thread\n", mine);
  CHECK(mine > 0);
  return 0;
}

/* A14: the loan survives preemption: a copy of the descriptor taken before a
   long computation still writes after many resumes. */
static unsigned long child_loan(void *arg)
{
  unsigned long *d = probe_descriptor();
  unsigned long x = lcg_rounds((unsigned long)(uintptr_t)arg, PREEMPT_ROUNDS);
  probe_store_through(d, 0x5a5a);
  return x;
}

static int loan_preempt(void)
{
  struct capstone_context_event ev;
  unsigned long preemptions = 0;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_loan, (void *)(uintptr_t)3));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, &preemptions));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == lcg_rounds(3, PREEMPT_ROUNDS));
  CHECK(preemptions >= 2);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* A15 (negative): a copy of this call's descriptor, kept past a cooperative
   return, has no authority in the next call. Expected: a fault at
   probe_store_insn. On a monitor that does not revoke the loan it prints
   REACHED instead. */
static void *volatile saved_descriptor;
static int loan_after_return(void)
{
  saved_descriptor = probe_descriptor();
  CHECK(saved_descriptor != 0);
  sched_yield();
  fflush(stdout);
  probe_store_through((unsigned long *)saved_descriptor, 0x1234);
  printf("context-probe loan-after-return: REACHED\n");
  return 1;
}

/* ---- A12: control state (Q3). Each mode passes on a clean refusal, or on
 * confined execution that is still preempted; it fails when a long run
 * outside the supervisor's reach is never preempted. ---- */
void probe_csr_read(void);
void probe_mret_to_user(unsigned long loops) __attribute__((noreturn));
void probe_wfi(void);
extern char probe_csr_insn[], probe_mret_insn[], probe_umode_ecall[], probe_wfi_insn[];
#define STEP_REFUSED 5
#define UMODE_LOOPS 300000000ul
#define PRIV_U_MSTATUS 0x800000000ul   /* (0 << 38) | (2 << 34): user privilege */
#define MIE_ALL 0x888ul                 /* MSIE, MTIE, MEIE */

static unsigned long child_csr(void *arg) { probe_csr_read(); return 1; }
static unsigned long child_mret(void *arg) { probe_mret_to_user(UMODE_LOOPS); }
static unsigned long child_wfi(void *arg) { probe_wfi(); return 3; }

static int step_report(const char *what, unsigned long id, struct capstone_context_event *ev,
                       unsigned long *preemptions)
{
  long t0 = monotonic_ms();
  int r = step_to_end(id, ev, preemptions);
  long ms = monotonic_ms() - t0;
  printf("context-probe %s: kind %lu cause %lu pc %#lx result %#lx, %lu preemptions in %ld ms\n",
         what, (unsigned long)ev->kind, (unsigned long)ev->cause, (unsigned long)ev->pc,
         (unsigned long)ev->result, *preemptions, ms);
  return r;
}

static int ctl_csr(void)
{
  struct capstone_context_event ev;
  unsigned long pre = 0;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_csr, 0));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_report("ctl-csr", (unsigned long)id, &ev, &pre));
  CHECK(ev.kind == STEP_FAULT && ev.cause == 2 && ev.pc == (uintptr_t)probe_csr_insn);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

static int ctl_mret(void)
{
  struct capstone_context_event ev;
  unsigned long pre = 0;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_mret, 0));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_report("ctl-mret", (unsigned long)id, &ev, &pre));
  capstone_context_forget((unsigned long)id);
  if (ev.kind == STEP_FAULT && ev.pc == (uintptr_t)probe_mret_insn)
    return 0;                                         /* refused at the mret */
  CHECK(ev.kind == STEP_FAULT && ev.pc == (uintptr_t)probe_umode_ecall);
  CHECK(pre > 0);                                     /* ran outside C-mode: still preempted? */
  return 0;
}

static int ctl_words(const char *what, unsigned long mstatus, unsigned long mie)
{
  struct capstone_context_event ev;
  unsigned long pre = 0, seed = 0x55;
  CHECK(!capstone_context_mint_words(&ctx, AREA_BYTES, child_preempt, (void *)(uintptr_t)seed,
                                     mstatus, mie));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_report(what, (unsigned long)id, &ev, &pre));
  capstone_context_forget((unsigned long)id);
  if (ev.kind == STEP_REFUSED)
    return 0;                                         /* refused at the first entry */
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*ctx.value == lcg_rounds(seed, PREEMPT_ROUNDS));
  CHECK(pre >= 2);
  return 0;
}

/* A12, nested (negative): the main context enters a seal minted with user
   privilege by a direct CALL. Expected: a fault at the CALL (cause 2). */
static int ctl_priv_nested(void)
{
  unsigned long result = 0;
  CHECK(!capstone_context_mint_words(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1,
                                     PRIV_U_MSTATUS, 0));
  fflush(stdout);
  __capstone_context_call(&ctx.seal, 0, &result);
  printf("context-probe ctl-priv-nested: REACHED result=%lx counter=%lu\n", result, counter);
  return 1;
}

static int ctl_wfi(void)
{
  struct capstone_context_event ev;
  unsigned long pre = 0;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_wfi, 0));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_report("ctl-wfi", (unsigned long)id, &ev, &pre));
  capstone_context_forget((unsigned long)id);
  if (ev.kind == STEP_FAULT && ev.pc == (uintptr_t)probe_wfi_insn)
    return 0;
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  return 0;
}

/* ---- A2: authority at entry. The seal's first pc is probe_entry_audit
 * (context-probe-asm.S), which records what the first entry delivered in
 * every register, and in cscratch, before the runtime's entry runs. ---- */
extern char probe_entry_audit[], probe_entry_leak[], probe_load_insn[], probe_store_insn[];
extern unsigned long probe_audit[42];
void probe_main_gp(unsigned long out[3]);
unsigned long probe_load_through(unsigned long *where);
#define CAP_TYPE_SEALEDRET 5
#define CAP_PERMS_WO 2
/* capstone-qemu reports a data access outside a capability's bounds as an
   access fault, store 7 and load 5 (op_helper.c _helper_access_with_cap); the
   RTL's load/store unit raises 28. */
#define CAUSE_STORE_OUTSIDE_BOUNDS 7

/* The first register other than ra, gp and a1 that entered tagged, or 0. */
static int audit_extra_tagged(void)
{
  for (int i = 1; i < 32; ++i)
    if (i != 1 && i != 3 && i != 11 && probe_audit[i] != CAPSTONE_CAP_EMPTY)
      return i;
  return 0;
}

/* Positive control: the audit must name the register probe_entry_leak fills. */
static int entry_audit_control(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint_entry(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1,
                                     probe_entry_leak));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  printf("context-probe entry-audit-control: extra tagged register x%d\n", audit_extra_tagged());
  CHECK(audit_extra_tagged() == 9);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

static int entry_audit(void)
{
  struct capstone_context_event ev;
  unsigned long *a = probe_audit, main_gp[3] = {0};
  CHECK(!capstone_context_mint_entry(&ctx, AREA_BYTES, child_enter, (void *)(uintptr_t)1,
                                     probe_entry_audit));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  /* The audit handed the entry on intact. */
  CHECK(*ctx.value == 42 && counter == 1);
  probe_main_gp(main_gp);
  printf("context-probe entry-audit: types");
  for (int i = 1; i < 32; ++i)
    printf(" %lu", a[i]);
  printf("\ncontext-probe entry-audit: gp [%#lx, %#lx) perms %lu, main gp type %lu [%#lx, %#lx)\n",
         a[32], a[33], a[34], main_gp[0], main_gp[1], main_gp[2]);
  printf("context-probe entry-audit: a1 [%#lx, %#lx) perms %lu; cscratch type %lu [%#lx, %#lx) "
         "perms %lu; area [%#lx, %#lx)\n", a[35], a[36], a[37], a[38], a[39], a[40], a[41],
         ctx.area_base, ctx.area_base + ctx.area_bytes);
  /* Tagged at entry: the return capability, gp and the descriptor loan. */
  CHECK(audit_extra_tagged() == 0);
  CHECK(a[1] == CAP_TYPE_SEALEDRET);
  /* gp: the shared image, no more than the main context entered with. */
  CHECK(a[3] == CAPSTONE_CAP_NONLINEAR && main_gp[0] == CAPSTONE_CAP_NONLINEAR);
  CHECK(a[32] >= main_gp[1] && a[33] <= main_gp[2] && a[32] < a[33]);
  /* a1: the loan, write-only over the 64-byte descriptor, outside the image. */
  CHECK(a[11] == CAPSTONE_CAP_NONLINEAR);
  CHECK(a[36] - a[35] == 64);
  CHECK(a[37] == CAP_PERMS_WO);
  CHECK(a[36] <= main_gp[1] || a[35] >= main_gp[2]);
  /* cscratch: the start block, and nothing more of the area. */
  CHECK(a[38] == CAPSTONE_CAP_NONLINEAR);
  CHECK(a[39] == ctx.area_base + CAPSTONE_CONTEXT_SEAL_BYTES);
  CHECK(a[40] == ctx.area_base + CAPSTONE_CONTEXT_SEAL_BYTES + CAPSTONE_CONTEXT_START_BYTES);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* A2, negative: a store one word past the 64-byte loan faults in the context.
   Then a load through the write-only loan, reported and not checked:
   capstone-qemu has no permission clause on data access, so on this platform
   the loan's authority is its bounds. */
static unsigned long child_past_loan(void *arg)
{
  unsigned long *d = probe_descriptor();
  probe_store_through(d + 8, 1);
  return 1;
}

static unsigned long child_read_loan(void *arg)
{
  return probe_load_through(probe_descriptor());
}

static int entry_negative(void)
{
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_past_loan, 0));
  long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  printf("context-probe entry-negative: past the loan: kind %lu cause %lu pc %#lx\n",
         (unsigned long)ev.kind, (unsigned long)ev.cause, (unsigned long)ev.pc);
  CHECK(ev.kind == STEP_FAULT && ev.cause == CAUSE_STORE_OUTSIDE_BOUNDS &&
        ev.pc == (uintptr_t)probe_store_insn);
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  capstone_context_revoke(&ctx);
  CHECK(!capstone_context_remint(&ctx, child_read_loan, 0));
  id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
  CHECK(id > 0);
  CHECK(!step_to_end((unsigned long)id, &ev, 0));
  printf("context-probe entry-negative: load through the loan (not checked): kind %lu cause %lu "
         "pc %#lx%s\n", (unsigned long)ev.kind, (unsigned long)ev.cause, (unsigned long)ev.pc,
         ev.kind == STEP_FAULT && ev.pc == (uintptr_t)probe_load_insn ? " at probe_load_insn" : "");
  CHECK(capstone_context_forget((unsigned long)id) == 0);
  return 0;
}

/* ---- A10: Linux never forgets. ---- */

/* The monitor's slot table (process-abi.h CAPSTONE_PROCESS_SLOTS); each
   application may hold 8 descriptors, one for its first context. */
#define A10_SLOTS 32
#define A10_CYCLES (4 * A10_SLOTS)

/* 128 create/exit/revoke cycles in one application, never forgetting, while a
   live context is stepped once per cycle. Without retirement the eighth
   registration of this application finds no descriptor. */
static int exhaust(void)
{
  static struct capstone_context live;
  struct capstone_context_event ev;
  unsigned long seed = 0x99, slots = 0, max_gen = 0;
  CHECK(!capstone_context_mint(&live, AREA_BYTES, child_preempt, (void *)(uintptr_t)seed));
  long live_id = capstone_context_create(&live, CAPSTONE_CONTEXT_REGISTER);
  CHECK(live_id > 0);
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_second, 0));
  for (unsigned long i = 0; i < A10_CYCLES; ++i) {
    if (i)
      CHECK(!capstone_context_remint(&ctx, child_second, (void *)(uintptr_t)i));
    long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
    if (id <= 0)
      printf("context-probe exhaust: cycle %lu: create %ld\n", i, id);
    CHECK(id > 0);
    CHECK(!step_to_end((unsigned long)id, &ev, 0));
    CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
    CHECK(*ctx.value == 1000 + i);
    capstone_context_revoke(&ctx);
    slots |= 1ul << ((unsigned long)id & 63);
    if ((unsigned long)id >> 32 > max_gen)
      max_gen = (unsigned long)id >> 32;
    memset(&ev, 0, sizeof ev);
    CHECK(!capstone_context_step((unsigned long)live_id, &ev));
    CHECK(ev.kind == STEP_PREEMPTED || ev.kind == STEP_RETURNED);
  }
  CHECK(!step_to_end((unsigned long)live_id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(*live.value == lcg_rounds(seed, PREEMPT_ROUNDS));
  printf("context-probe exhaust: %d cycles over %d slots (mask %#lx), highest generation %lu\n",
         A10_CYCLES, __builtin_popcountl(slots), slots, max_gen);
  CHECK(capstone_context_forget((unsigned long)live_id) == 0);
  return 0;
}

/* One application of several run at once (run-context.py HOLDERS): a live
   context and six dead registrations it never forgets, eight slots with its
   first context. Five of them claim 40 slots of 32 while all are alive, so
   they can only succeed when an adoption or a launch retires another
   application's dead registrations. The live context must be untouched. */
#define HOLD_DEAD 6
#define HOLD_MS 20000
static int hold(void)
{
  static struct capstone_context live;
  struct capstone_context_event ev;
  CHECK(!capstone_context_mint(&live, AREA_BYTES, child_enter, (void *)(uintptr_t)1));
  long live_id = capstone_context_create(&live, CAPSTONE_CONTEXT_REGISTER);
  CHECK(live_id > 0);
  CHECK(!step_to_end((unsigned long)live_id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(!capstone_context_mint(&ctx, AREA_BYTES, child_second, 0));
  for (unsigned long i = 0; i < HOLD_DEAD; ++i) {
    if (i)
      CHECK(!capstone_context_remint(&ctx, child_second, (void *)(uintptr_t)i));
    long id = capstone_context_create(&ctx, CAPSTONE_CONTEXT_REGISTER);
    if (id <= 0)
      printf("context-probe hold: registration %lu: create %ld\n", i, id);
    CHECK(id > 0);
    CHECK(!step_to_end((unsigned long)id, &ev, 0));
    CHECK(ev.kind == STEP_RETURNED && *ctx.value == 1000 + i);
    capstone_context_revoke(&ctx);
  }
  printf("context-probe hold: holding at %ld\n", monotonic_ms());
  fflush(stdout);
  struct timespec pause = {HOLD_MS / 1000, 0};
  while (nanosleep(&pause, &pause) && errno == EINTR)
    ;
  printf("context-probe hold: released at %ld\n", monotonic_ms());
  CHECK(!step_to_end((unsigned long)live_id, &ev, 0));
  CHECK(ev.kind == STEP_RETURNED && ev.result == CAPSTONE_CONTEXT_EXITED);
  CHECK(counter == 1 && *live.value == 42);
  CHECK(capstone_context_forget((unsigned long)live_id) == 0);
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
  else if (!strcmp(mode, "reenter-revoked")) rc = reenter_revoked();
  else if (!strcmp(mode, "done-preempted")) rc = done_preempted();
  else if (!strcmp(mode, "rollback-thread")) rc = rollback_thread();
  else if (!strcmp(mode, "duplicate-adopt")) rc = duplicate_adopt();
  else if (!strcmp(mode, "two-steppers")) rc = two_steppers();
  else if (!strcmp(mode, "loan-preempt")) rc = loan_preempt();
  else if (!strcmp(mode, "loan-after-return")) rc = loan_after_return();
  else if (!strcmp(mode, "ctl-csr")) rc = ctl_csr();
  else if (!strcmp(mode, "ctl-mret")) rc = ctl_mret();
  else if (!strcmp(mode, "ctl-priv")) rc = ctl_words("ctl-priv", PRIV_U_MSTATUS, 0);
  else if (!strcmp(mode, "ctl-mie")) rc = ctl_words("ctl-mie", CAPSTONE_CONTEXT_MSTATUS, MIE_ALL);
  else if (!strcmp(mode, "ctl-wfi")) rc = ctl_wfi();
  else if (!strcmp(mode, "ctl-priv-nested")) rc = ctl_priv_nested();
  else if (!strcmp(mode, "entry-audit")) rc = entry_audit();
  else if (!strcmp(mode, "entry-negative")) rc = entry_negative();
  else if (!strcmp(mode, "entry-audit-control")) rc = entry_audit_control();
  else if (!strcmp(mode, "exhaust")) rc = exhaust();
  else if (!strcmp(mode, "hold")) rc = hold();
  else {
    fprintf(stderr, "context-probe: unknown mode %s\n", mode);
    return 2;
  }
  if (!rc)
    printf("context-probe %s: PASS\n", mode);
  return rc;
}
