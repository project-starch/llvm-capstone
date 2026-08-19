/* Reference mruby, running precompiled Ruby bytecode in a pure-capability domain.
 *
 * WHY BYTECODE AND NOT RUBY SOURCE. mruby's parser does not compile for this
 * target: parse.y stores integers inside `node *` fields (`#define nint(x)
 * ((node*)(intptr_t)(x))`, parse.y:68-72), and forging a capability from an
 * integer is what this machine refuses -- it presents as
 * `Cannot select: i128 = xor ..., Constant:i128<2>`. Precompiling with the host
 * `mrbc` and loading the result with mrb_load_irep is the ordinary embedded
 * mruby configuration, and it puts the VM in the domain without the compiler.
 * Runtime `eval` of Ruby source is what this gives up; see the history note.
 *
 * WHAT REACHING 400 REQUIRES: the VM's dispatch (switch, not computed goto),
 * the object system, symbol and class tables, the GC, and an Array that grows
 * -- i.e. a realloc that MOVES a capability-bearing buffer. A probe that only
 * created a state would prove far less.
 *
 * REPORTING GOES THROUGH SAY, the raw hostcall, not through Ruby. If the VM is
 * broken, a probe that reports through it says nothing at all.
 */
#ifndef MRUBY_PROBE_STAGE
#define MRUBY_PROBE_STAGE 0
#endif

#include <mruby.h>
#include <mruby/irep.h>
#include <mruby/value.h>
#ifdef MRUBY_PROBE_CDP
#include <mruby/class.h>
#include <mruby/compile.h>
#include <mruby/proc.h>
#include <mruby/variable.h>
#include <mruby/string.h>
#endif

#include "probe_irep.c"

#ifdef MRUBY_PROBE_PARSER
#include <mruby/compile.h>
/* Byte for byte what probe.rb contains, minus its comment. Kept as one string
   rather than read from a file: a domain has no working directory, and the
   point is the parser, not file I/O. */
#define PROBE_SOURCE \
  "t = []\n"                                                                  \
  "i = 1\n"                                                                   \
  "while i <= 20\n"                                                           \
  "  t[i - 1] = i * i\n"                                                      \
  "  i += 1\n"                                                                \
  "end\n"                                                                     \
  "t[19]\n"
#endif

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);
#define SAY(s) __capstone_hc_write(1, (s), sizeof(s) - 1)

/* THE ALLOCATOR IS INSTRUMENTED SO THAT RUNNING OUT OF MEMORY RETURNS A RESULT.
 *
 * The first run of this probe faulted between S1 and S2, i.e. inside mrb_open,
 * with `cause = 4`. Heap exhaustion is one of the candidates, and the shape it
 * takes is a null from malloc that mruby then walks into -- a capability fault
 * some distance from the cause, which says nothing about how much memory was
 * wanted. Wrapping mruby's own allocator hook converts that into a number.
 *
 * mrb_allocf contract: size 0 means free and returns NULL, otherwise it is a
 * realloc. Accounting is deliberately crude -- the sizes of freed blocks are not
 * tracked, so `live` only ever grows and is an upper bound. It does not need to
 * be exact to answer "how much heap does mrb_open want". */
static unsigned long mem_live, mem_peak, mem_calls, mem_fails;

/* MRUBY_PROBE_BUMP: the same probe with an O(1) allocator that never frees.
 *
 * The matched arm for a WEDGE inside mrb_open. libc-ext/malloc.c is first fit
 * over an address-ordered list, so every malloc walks live blocks -- O(n) per
 * call, quadratic over a run, and its own comment says so. mruby loading mrblib
 * makes tens of thousands of calls, and under TCG quadratic is indistinguishable
 * from hung. A bump allocator differs from it in exactly that one property:
 * same probe, same mruby, same heap size, no search. If mruby then completes,
 * the allocator is too slow rather than mruby being broken; if it still wedges,
 * the allocator is exonerated and the wedge is inside mruby. */
#ifdef MRUBY_PROBE_BUMP
#ifndef MRUBY_PROBE_BUMP_BYTES
#define MRUBY_PROBE_BUMP_BYTES (1024 * 1024)
#endif
static char bump_arena[MRUBY_PROBE_BUMP_BYTES] __attribute__((aligned(16)));
static size_t bump_used;

/* A 16-byte header carrying the size, so realloc copies min(old, new) instead of
   reading past the old block. Without it this arm faults on its own bounds and
   the run would read as "mruby faults", which is the answer it exists to test. */
static void *bump_alloc(size_t size) {
  size_t want = (size + 15u) & ~(size_t)15u;
  if (want + 16u > sizeof(bump_arena) - bump_used)
    return 0;
  size_t *h = (size_t *)(void *)(bump_arena + bump_used);
  *h = size;
  bump_used += want + 16u;
  return (char *)h + 16;
}

static size_t bump_size_of(void *p) { return *(size_t *)(void *)((char *)p - 16); }
#endif

static void *probe_allocf(mrb_state *mrb, void *ptr, size_t size, void *ud) {
  (void)mrb;
  (void)ud;
  mem_calls++;
  /* HEARTBEAT. mrb_open neither returns nor faults, and "hung" and "slow under
     TCG at -O0" look identical from outside. One marker every 4096 allocations
     separates them: ticks that keep coming mean progress and the timeout is the
     wrong instrument; ticks that stop mean a real loop, and the count says how
     far mruby got.

     FIRST PASS used 1/4096 and printed NOTHING, which already rules out "slow
     because of millions of allocations" -- it wedged inside the first 4096. Now
     every call for the first 16, then every 64, so the last number brackets the
     wedge to within 64 allocations. */
  if (mem_calls <= 16 || (mem_calls & 0x3f) == 0) {
    char tb[64];
    int tn = snprintf(tb, sizeof tb, "MRUBY TICK %lu\n", mem_calls);
    if (tn > 0)
      __capstone_hc_write(1, tb, (unsigned long)tn);
  }
  if (size == 0) {
#ifndef MRUBY_PROBE_BUMP
    free(ptr);
#endif
    return 0;
  }
#ifdef MRUBY_PROBE_BUMP
  void *p = bump_alloc(size);
  if (p && ptr) {
    size_t old = bump_size_of(ptr);
    size_t n = old < size ? old : size;
    const char *from = ptr;
    char *to = p;
    for (size_t i = 0; i < n; i++)
      to[i] = from[i];
  }
#else
  void *p = realloc(ptr, size);
#endif
  if (!p) {
    mem_fails++;
    SAY("MRUBY FAIL: allocator returned null -- heap exhausted\n");
    return 0;
  }
  mem_live += size;
  if (mem_live > mem_peak)
    mem_peak = mem_live;
  return p;
}

/* Printed through snprintf, which this libc now has. Not through mruby: the
   point is to report even when mruby is the thing that is broken. */
#ifdef MRUBY_PROBE_REVOKE
/* The revoking allocator owns its own counters; ours only see what mruby asked
   for. carved is every byte ever handed out (the arena never reclaims), and
   peak_slots is the high-water of CONCURRENTLY live allocations -- the number
   that actually sizes ROF_MAX_SLOTS, which our call count cannot give. */
extern unsigned long xlang_mem_carved(void);
extern unsigned long xlang_mem_live_bytes(void);
extern unsigned xlang_mem_live_count(void);
extern unsigned xlang_mem_peak_slots(void);
/* What the allocator actually FREED. mruby frees GC pages, not objects, so a
   use-after-free inside a page never reaches revocation -- see row 14. */
extern unsigned long xlang_mem_free_calls(void);
extern unsigned long xlang_mem_free_big(void);
extern unsigned long xlang_mem_free_bytes(void);

static void say_arena(const char *stage) {
  char b[224];
  int n = snprintf(b, sizeof b,
                   /* TWO PRINTS, FEW ARGUMENTS EACH. One call with eight
                      conversions produced a seventh value that could not be
                      true (824 frees of which "404288" were large) and an
                      eighth that was 0xFFFFFFFF. Whether the limit is the
                      argument COUNT or the va_arg handling this target already
                      has form for (C-26), the fix is the same and costs
                      nothing: keep each formatted line short. */
                   "MRUBY ARENA %s: carved=%lu live_bytes=%lu live=%lu peak_slots=%lu\n",
                   stage, xlang_mem_carved(), xlang_mem_live_bytes(),
                   (unsigned long)xlang_mem_live_count(),
                   (unsigned long)xlang_mem_peak_slots());
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
  n = snprintf(b, sizeof b,
                   "MRUBY FREES %s: calls=%lu big=%lu bytes=%lu\n",
                   stage, xlang_mem_free_calls(), xlang_mem_free_big(),
                   xlang_mem_free_bytes());
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
}
#else
#define say_arena(stage) ((void)0)
#endif

/* The LIBC heap, which is a different thing from the counters above and the one
   that actually sizes the image. The arena numbers only see what the OUTER mruby
   state asked for; the test suite's sub-interpreters are opened with
   mrb_default_allocf and go straight to libc malloc, so they are invisible there
   and dominate the requirement. hwm is the far end of the furthest block ever
   handed out (libc-ext/malloc.c). */
extern unsigned long __capstone_libc_heap_hwm(void);
extern unsigned long __capstone_libc_heap_size(void);

/* TWO snprintf CALLS, NOT ONE, and that is not style.
 *
 * Folding the heap numbers into the line above made it seven varargs, and the
 * SEVENTH came back as garbage that differed on every call -- 4329522383,
 * 4329507597, ... for a value that is a compile-time constant. The hwm beside it
 * was correct and monotone, so the values are fine and the argument POSITION is
 * not. A capability is two register slots wide here, so a call that looks like
 * six arguments is already past what the ABI passes in registers.
 *
 * Not chased further because splitting the line costs nothing and the numbers
 * are then trustworthy. Left recorded rather than silently worked around: a
 * vararg that returns a different wrong answer each call is worth knowing about
 * before something depends on the sixth argument of a printf. */
static void say_mem(const char *stage) {
  char b[160];
  int n = snprintf(b, sizeof b,
                   "MRUBY MEM %s: requested=%lu peak=%lu calls=%lu fails=%lu\n",
                   stage, mem_live, mem_peak, mem_calls, mem_fails);
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
  n = snprintf(b, sizeof b, "MRUBY LIBCHEAP %s: used=%lu of=%lu\n", stage,
               __capstone_libc_heap_hwm(), __capstone_libc_heap_size());
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
}

/* Generated by mruby's build system from the gembox; a core-only domain has no
   gems, so the list is empty rather than absent. Defined here so the absence is
   deliberate and visible instead of a link error someone silences later. */
#ifndef MRUBY_PROBE_GEMS
/* No gems: the stubs stand in for what mruby's build system would generate.
   With MRUBY_PROBE_GEMS the build generates the real thing instead. */
void mrb_init_mrbgems(mrb_state *mrb) { (void)mrb; }
void mrb_final_mrbgems(mrb_state *mrb) { (void)mrb; }
#endif

/* STAGED ARMS, one per hypothesis, each RETURNING a marker instead of running
 * to the failure. This is the shape CLAUDE.md prescribes for a wedge, and the
 * reason it is here is that the previous nine boots each tested one hypothesis
 * serially -- exactly what that section was written to prevent.
 *
 * mrb_open_allocf decomposes as: allocate the state, then
 * mrb_core_init_protect(init_gc_and_core), then the gems (our stub).
 * init_gc_and_core is mrb_gc_init, then a context allocation, then
 * mrb_init_core. The stages below replicate that split, so a wedge lands
 * between two markers rather than inside one opaque call.
 *
 * Ordering for the batch: 1, 2, 3, then the full run LAST. A wedge ends the
 * boot, so everything expected to return goes first and at most one arm is
 * expected not to. */
#if MRUBY_PROBE_STAGE > 0
void mrb_gc_init(mrb_state *, mrb_gc *);
void mrb_init_core(mrb_state *);
#if MRUBY_PROBE_STAGE >= 7
#include "ladder_ireps.c"
#endif
#if MRUBY_PROBE_STAGE >= 6
/* The real headers, not hand-written prototypes. The first attempt declared
   mrb_top_run's last parameter as `unsigned int` where mruby has `mrb_int`, and
   the build failed with "conflicting types" -- after the run script had already
   been launched, so the boot executed the previous image and the log looked
   like a stage-5 result. Declaring by hand what a header already declares is
   how an arm silently becomes a different arm. */
#include <mruby/dump.h>
#include <mruby/proc.h>
#endif
#if MRUBY_PROBE_STAGE >= 4
void mrb_init_symtbl(mrb_state *);     void mrb_init_class(mrb_state *);
void mrb_init_object(mrb_state *);     void mrb_init_kernel(mrb_state *);
void mrb_init_comparable(mrb_state *); void mrb_init_enumerable(mrb_state *);
void mrb_init_symbol(mrb_state *);     void mrb_init_string(mrb_state *);
void mrb_init_exception(mrb_state *);  void mrb_init_proc(mrb_state *);
void mrb_init_array(mrb_state *);      void mrb_init_hash(mrb_state *);
void mrb_init_numeric(mrb_state *);    void mrb_init_range(mrb_state *);
void mrb_init_gc(mrb_state *);         void mrb_init_version(mrb_state *);
void mrb_init_mrblib(mrb_state *);
#endif

/* STAGE 9: a variadic 32-byte struct WITHOUT mruby.
 *
 * mrb_funcall_id(argc=0) returns and mrb_funcall_id(argc=1) faults, so the
 * vararg loop is the variable and va_arg of an mrb_value -- 32 bytes by value
 * under MRB_NO_BOXING -- is the suspect. The shape is reproduced here with four
 * named parameters, two of them structs by value, because that is what
 * mrb_funcall_id has and it decides which registers hold varargs.
 *
 * If this faults, the reproducer is ten lines and mruby is out of the picture.
 * If it returns, the shape alone is not enough and mruby's frame is. */
#if MRUBY_PROBE_STAGE >= 9
struct probe_val { union { long i; double f; void *p; } u; int tt; };

static long va_sum(struct probe_val self, int mid, int argc, ...) {
  struct probe_val argv[16];
  va_list ap;
  long sum = self.u.i + mid;
  va_start(ap, argc);
  for (int i = 0; i < argc; i++)
    argv[i] = va_arg(ap, struct probe_val);
  va_end(ap);
  for (int i = 0; i < argc; i++)
    sum += argv[i].u.i;
  return sum;
}
#endif

int capstone_main(void);   /* defined below; the base print needs its address */

static int run_stage(void) {
  char b[80];
  int n;

  /* THE LOAD ADDRESS, for every staged arm and not just the full run.
   * Without it a fault pc can only be mapped by ASSUMING which instruction
   * faulted and deriving the base from that -- which is circular, and I did it
   * once before catching it. With it the mapping is arithmetic. */
  n = snprintf(b, sizeof b, "MRUBY BASE: capstone_main=%lx\n",
               (unsigned long)(uintptr_t)(void *)&capstone_main);
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);

#if MRUBY_PROBE_STAGE >= 9
  {
    struct probe_val s = {{7}, 3};
    struct probe_val a = {{11}, 3};
    SAY("MRUBY STAGE 9: variadic struct, argc=0\n");
    long r0 = va_sum(s, 5, 0);
    char b9[80];
    int n9 = snprintf(b9, sizeof b9, "MRUBY STAGE 9: argc=0 gave %ld (want 12)\n", r0);
    if (n9 > 0)
      __capstone_hc_write(1, b9, (unsigned long)n9);

    SAY("MRUBY STAGE 9: variadic struct, argc=1\n");
    long r1 = va_sum(s, 5, 1, a);
    n9 = snprintf(b9, sizeof b9, "MRUBY STAGE 9: argc=1 gave %ld (want 23)\n", r1);
    if (n9 > 0)
      __capstone_hc_write(1, b9, (unsigned long)n9);
    SAY("__CAPSTONE_MRUBY_STAGE_PASSED__\n");
    return 0;
  }
#endif

  mrb_state *mrb = (mrb_state *)probe_allocf(0, 0, sizeof(mrb_state), 0);
  if (!mrb) {
    SAY("MRUBY STAGE FAIL: no memory for mrb_state\n");
    return 1;
  }
  for (size_t i = 0; i < sizeof(mrb_state); i++)
    ((char *)mrb)[i] = 0;
  mrb->allocf_ud = 0;
  mrb->allocf = probe_allocf;
  mrb->atexit_stack_len = 0;
  SAY("MRUBY STAGE 1: mrb_state allocated and zeroed\n");
  if (MRUBY_PROBE_STAGE == 1)
    goto done;

  mrb_gc_init(mrb, &mrb->gc);
  SAY("MRUBY STAGE 2: mrb_gc_init returned\n");
  if (MRUBY_PROBE_STAGE == 2)
    goto done;

  mrb->c = (struct mrb_context *)probe_allocf(mrb, 0, sizeof(struct mrb_context), 0);
  if (!mrb->c) {
    SAY("MRUBY STAGE FAIL: no memory for mrb_context\n");
    return 1;
  }
  for (size_t i = 0; i < sizeof(struct mrb_context); i++)
    ((char *)mrb->c)[i] = 0;
  mrb->root_c = mrb->c;
  SAY("MRUBY STAGE 2b: context allocated\n");

#if MRUBY_PROBE_STAGE >= 4
  /* STAGE 4: mrb_init_core's own sequence, inlined with a marker after each
   * call. Not seventeen arms -- one arm with seventeen markers. The fault kills
   * QEMU, but SAY is a synchronous hostcall, so everything printed before it
   * survives and the LAST marker names the init function. Copied verbatim from
   * src/init.c, including the mrb_gc_arena_restore after each, because the
   * point is to reproduce that function and not an approximation of it. */
#define INIT_STEP(fn)                                                          \
  do {                                                                         \
    fn(mrb);                                                                   \
    mrb_gc_arena_restore(mrb, 0);                                              \
    SAY("MRUBY INIT: " #fn " returned\n");                                     \
  } while (0)

  INIT_STEP(mrb_init_symtbl);
  INIT_STEP(mrb_init_class);
  INIT_STEP(mrb_init_object);
  INIT_STEP(mrb_init_kernel);
  INIT_STEP(mrb_init_comparable);
  INIT_STEP(mrb_init_enumerable);
  INIT_STEP(mrb_init_symbol);
  INIT_STEP(mrb_init_string);
  INIT_STEP(mrb_init_exception);
  INIT_STEP(mrb_init_proc);
  INIT_STEP(mrb_init_array);
  INIT_STEP(mrb_init_hash);
  INIT_STEP(mrb_init_numeric);
  INIT_STEP(mrb_init_range);
  INIT_STEP(mrb_init_gc);
  INIT_STEP(mrb_init_version);
#if MRUBY_PROBE_STAGE >= 5
  /* STAGE 5 SKIPS mrb_init_mrblib and runs OUR irep instead.
   *
   * Stage 4 measured that sixteen of seventeen init functions return and the
   * seventeenth faults, so mruby's whole C object system works here and what
   * fails is loading the Ruby-level standard library -- mrb_load_irep over
   * mrblib's bytecode. This arm is the matched pair: same VM, same loader, a
   * DIFFERENT irep, namely the four-line chunk from probe.rb.
   *
   * Returning 400 means the VM executes Ruby bytecode in a domain and the
   * defect is specific to mrblib's irep (its size, or a construct in it).
   * Faulting the same way means the loader or the VM is broken for any irep and
   * mrblib is innocent. Either answer is worth a boot; neither is guessable. */
  SAY("MRUBY STAGE 5: skipping mrb_init_mrblib on purpose\n");
#if MRUBY_PROBE_STAGE >= 6
  /* STAGE 6 SPLITS mrb_load_irep into its two halves.
   *
   * Stage 5 established that the fault is not about mrblib: the same
   * helper_cslcc assertion fires on our own four-line irep. mrb_load_irep is
   * mrb_read_irep, which turns the binary into an mrb_irep and an RProc, then
   * mrb_top_run, which executes it. Those are very different suspects -- a
   * binary-format reader walking a const array in .rodata, versus the VM's
   * dispatch loop -- and one marker between them separates them. */
#if MRUBY_PROBE_STAGE >= 7
  /* THE BYTECODE LADDER. mrb_read_irep and mrb_proc_new both return, so the
   * fault is inside mrb_top_run. Six chunks, each one construct larger than the
   * last, run in ascending order with a marker after each. The last marker names
   * the Ruby construct whose bytecode the VM cannot execute here -- which is a
   * far smaller search than "somewhere in vm.c". */
  {
    static const struct {
      const char *name;
      const unsigned char *irep;
      long want;
    } rungs[] = {
      {"1 nil",          ladder_1_nil,         -1},
      {"2 integer",      ladder_2_int,          1},
      {"3 add",          ladder_3_add,          3},
      {"4 empty array",  ladder_4_emptyarray,   0},
      {"5 array store",  ladder_5_arrayset,     7},
      {"6 while loop",   ladder_6_whileloop,  210},
    };
    for (unsigned r = 0; r < sizeof(rungs) / sizeof(rungs[0]); r++) {
      char rb[96];
      int rn = snprintf(rb, sizeof rb, "MRUBY RUNG %s: about to run\n", rungs[r].name);
      if (rn > 0)
        __capstone_hc_write(1, rb, (unsigned long)rn);

      mrb_value rv = mrb_load_irep(mrb, rungs[r].irep);
      long got = mrb_fixnum_p(rv) ? (long)mrb_fixnum(rv) : -1;
      rn = snprintf(rb, sizeof rb, "MRUBY RUNG %s: returned %ld (want %ld)%s\n",
                    rungs[r].name, got, rungs[r].want,
                    mrb->exc ? " EXCEPTION" : "");
      if (rn > 0)
        __capstone_hc_write(1, rb, (unsigned long)rn);
      mrb->exc = 0;
    }
    SAY("MRUBY STAGE 7: every rung returned\n");
#if MRUBY_PROBE_STAGE >= 8
    /* STAGE 8: mrb_funcall_id DIRECTLY, without mrblib.
     *
     * The full run still faults after the C-25 fix, now in mrb_funcall_id+0x160
     * with cause 24 -- an untagged operand where a capability was required, at
     * `cincoffsetimm a0, s0, -0x258`, i.e. addressing its own `mrb_value
     * argv[MRB_FUNCALL_ARGC_MAX]`. That function is VARIADIC and reads
     * va_arg(ap, mrb_value), a 32-byte struct by value under MRB_NO_BOXING --
     * a case the earlier va_arg check (int, long, void*, double) never covered.
     *
     * Calling it here with argc 0 and then 1 separates "the function is broken
     * at all" from "the vararg loop is what breaks it", and does so without
     * mrblib in the picture. A five-line reproducer beats a 226 KB one. */
    SAY("MRUBY STAGE 8: mrb_funcall_id with argc=0\n");
    {
      mrb_value r0 = mrb_funcall_id(mrb, mrb_top_self(mrb),
                                    mrb_intern_lit(mrb, "class"), 0);
      (void)r0;
      SAY("MRUBY STAGE 8: argc=0 returned\n");
      mrb->exc = 0;
    }
    SAY("MRUBY STAGE 8: mrb_funcall_id with argc=1\n");
    {
      mrb_value r1 = mrb_funcall_id(mrb, mrb_top_self(mrb),
                                    mrb_intern_lit(mrb, "=="), 1,
                                    mrb_fixnum_value(1));
      (void)r1;
      SAY("MRUBY STAGE 8: argc=1 returned\n");
      mrb->exc = 0;
    }
#endif
    goto done;
  }
#endif
  {
    struct RProc *proc;
    mrb_irep *irep = mrb_read_irep(mrb, probe_irep);
    SAY("MRUBY STAGE 6: mrb_read_irep returned\n");
    if (!irep) {
      SAY("MRUBY STAGE 6: ... but it returned NULL\n");
      goto done;
    }
    proc = mrb_proc_new(mrb, irep);
    SAY("MRUBY STAGE 6: mrb_proc_new returned\n");
    if (!proc) {
      SAY("MRUBY STAGE 6: ... but it returned NULL\n");
      goto done;
    }
    proc->c = NULL;
    mrb_value v6 = mrb_top_run(mrb, proc, mrb_top_self(mrb), 0);
    SAY("MRUBY STAGE 6: mrb_top_run returned\n");
    if (mrb_fixnum_p(v6) && mrb_fixnum(v6) == 400)
      SAY("MRUBY STAGE 6: t[19] == 400\n");
    else
      SAY("MRUBY STAGE 6: ran but did not produce 400\n");
  }
#else
  {
    mrb_value v5 = mrb_load_irep(mrb, probe_irep);
    if (mrb->exc) {
      SAY("MRUBY STAGE 5: exception while running our own irep\n");
    } else if (mrb_fixnum_p(v5) && mrb_fixnum(v5) == 400) {
      SAY("MRUBY STAGE 5: our irep ran, t[19] == 400\n");
    } else {
      SAY("MRUBY STAGE 5: our irep ran but did not produce 400\n");
    }
  }
#endif
#else
  INIT_STEP(mrb_init_mrblib);
#endif
  SAY("MRUBY STAGE 4: every mrb_init_* returned\n");
#else
  mrb_init_core(mrb);
  SAY("MRUBY STAGE 3: mrb_init_core returned\n");
#endif

done:
  n = snprintf(b, sizeof b, "MRUBY STAGE %d DONE: allocs=%lu peak=%lu\n",
               MRUBY_PROBE_STAGE, mem_calls, mem_peak);
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
  SAY("__CAPSTONE_MRUBY_STAGE_PASSED__\n");
  return 0;
}
#endif

#ifdef MRUBY_PROBE_ROW
#include <mruby/compile.h>
#include <mruby/string.h>
#include <mruby/variable.h>
/* AN ACTUAL CORPUS ROW, on the real interpreter, with the corpus's own trigger.
 *
 * xlang/repro/<n>/trigger.rb is embedded VERBATIM by embed-ruby.py -- the whole
 * value of this is that the input is the corpus's file and not a paraphrase.
 *
 * Row 10 (CVE-2022-1106, OP_RANGE_INC) is the one that can run without porting
 * anything new: its pinned mruby commit is the tree this port was validated
 * against. Its trigger is pure Ruby, so it needs no C harness at all -- the
 * parser and the VM are the whole vehicle.
 *
 * Matched arms are the same two the corpus uses, selected by
 * MRUBY_PROBE_CDP_CONTROL: revocation off (expect the stale write to land) and
 * revocation on (expect a fault). The control is load-bearing: without it, a
 * fault could mean the trigger never armed. */
#include "row_trigger.c"
extern void xlang_set_no_revoke(void);

static int run_row(mrb_state *mrb) {
  char b[96];
  int n = snprintf(b, sizeof b, "ROW %d: running the corpus trigger verbatim\n",
                   MRUBY_PROBE_ROW);
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);

  /* THREE arms, not two, and the label must say which one -- a mislabelled arm
     is how a run gets attributed to the wrong mechanism.
       revoke   rof, SPLIT per allocation: exact bounds AND revocation
       norevoke rof with the REVOKE suppressed: exact bounds, no revocation
       libc     libc-ext/malloc.c: neither. Ordinary malloc semantics, the same
                thing the corpus shims ran against and the same thing x86 gives.
     The third exists because the second is NOT a clean control: it still bounds
     every allocation exactly, so a benign over-read that ordinary malloc absorbs
     faults there. Whether that is what makes the control arm die inside
     GC.start is exactly what this arm decides. */
#if !defined(MRUBY_PROBE_REVOKE)
  SAY("ROW ARM: libc allocator (no bounds per allocation, no revocation)\n");
#elif defined(MRUBY_PROBE_CDP_CONTROL)
  xlang_set_no_revoke();
  SAY("ROW ARM: rof with revocation DISABLED (exact bounds remain)\n");
#else
  SAY("ROW ARM: rof, revoke-on-free (the shipped configuration)\n");
#endif

  /* ALLOCATION COUNT ACROSS THE TRIGGER, and it is the only progress readout
     that works for EVERY row. `$arr.size` below is specific to the six rows
     built on the recurse(150) template; row 14 builds a local hash, whose name
     is out of scope the moment mrb_load_string returns, so it reports -1 there
     and that -1 means nothing. A trigger that ran does thousands of
     allocations; one that raised on its first line does almost none. Print
     both sides so the difference is visible without another boot. */
  say_mem("before-row");
  say_arena("before-row");
  mrb_load_string(mrb, row_trigger);
  say_mem("after-row");
  if (mrb->exc) {
    /* NAME the exception. "raised a Ruby exception" is where two arms report
       identically and neither means anything -- it is the shape of a trigger
       that never armed. The message says which method is missing, which is the
       difference between "the mechanism did not fire" and "this gem is not
       linked". */
    SAY("ROW: the trigger raised a Ruby exception: ");
    mrb_value mesg = mrb_iv_get(mrb, mrb_obj_value(mrb->exc),
                                mrb_intern_lit(mrb, "mesg"));
    if (mrb_string_p(mesg))
      __capstone_hc_write(1, RSTRING_PTR(mesg), (unsigned long)RSTRING_LEN(mesg));
    else
      SAY("<no message>");
    SAY("\n");
    mrb->exc = 0;
  }
  /* DID THE TRIGGER ACTUALLY RUN TO THE END? "Completed without a fault" does
   * not answer that, and on 2026-08-15 three arms reported it identically while
   * none of them had armed the bug. Row 10's trigger pushes TWO objects into
   * $arr per recursion level, so $arr.size == 2 * depth is a direct readout of
   * how deep it got. `recurse(150)` runs at depths 150 down to 0, so a complete
   * run is 151 levels and $arr.size == 302; less means it stopped early.
   * Asked in Ruby, after the fact, so the corpus's trigger file stays verbatim.
   *
   * A NEGATIVE result here is a real answer and must not read as an error: -1
   * means $arr never existed, which is a different failure from a short run. */
  mrb_value depth = mrb_load_string(mrb, "$arr ? $arr.size : -1");
  if (mrb->exc) {
    SAY("ROW: could not read $arr.size back\n");
    mrb->exc = 0;
  } else if (mrb_fixnum_p(depth)) {
    long d = (long)mrb_fixnum(depth);
    /* -1 is NOT a failure: it means the trigger has no $arr, which is true of
       every row outside the recurse(150) family. Saying so keeps the number
       from being read as "the recursion did not run". */
    n = d < 0
            ? snprintf(b, sizeof b,
                       "ROW: no $arr in this trigger; see MRUBY MEM before/after-row\n")
            : snprintf(b, sizeof b,
                       "ROW: $arr.size = %ld (302 = all 151 levels ran)\n", d);
    if (n > 0)
      __capstone_hc_write(1, b, (unsigned long)n);
  }

  SAY("ROW: trigger COMPLETED without a capability fault\n");
  say_arena("after-row");
  SAY("__CAPSTONE_MRUBY_ROW_DONE__\n");
  return 0;
}
#endif

#ifdef MRUBY_PROBE_CDP
/* THE CROSS-DOMAIN POINTER BUG, ON THE REAL INTERPRETER.
 *
 * Six of the twelve Ruby rows in xlang/ are the SAME mechanism, and its own
 * RESULTS.md says so: a raw interior pointer into the VM stack, cached across a
 * re-entrant Ruby callback that reallocates that stack. The corpus measures it
 * through distilled C shims against a mock allocator, because until today there
 * was no interpreter to measure it on.
 *
 * This is that mechanism written against mruby itself, on the revoking arena:
 *
 *   1. a C method caches mrb->c->ci->stack, an interior pointer;
 *   2. it calls back into RUBY, which recurses until the VM stack is extended
 *      -- mruby reallocates it and frees the old buffer, which under
 *      revoke-on-free REVOKES every capability into it;
 *   3. it dereferences the cached pointer.
 *
 * MATCHED PAIR, mirroring the corpus's two configs exactly:
 *   control  xlang_set_no_revoke() -- the arena still allocates and frees, but
 *            free does not revoke. The stale read should SUCCEED (MISS).
 *   revoke   the shipped configuration. The stale read should FAULT (BLOCKED).
 * Anything else is a result about the instrument, not the mechanism: if the
 * control also faults, the stack never moved and the probe proves nothing. */
extern void xlang_set_no_revoke(void);

static mrb_value cdp_stale(mrb_state *mrb, mrb_value self) {
  mrb_value *cached = mrb->c->ci->stack;
  SAY("CDP 1: cached an interior pointer into the VM stack\n");

  mrb_funcall(mrb, self, "deepen", 0);
  if (mrb->exc) {
    SAY("CDP FAIL: the Ruby callback raised\n");
    mrb->exc = 0;
    return mrb_nil_value();
  }
  SAY("CDP 2: callback returned; the VM stack has been extended\n");

  /* THE OFFENDING ACCESS. Reached only if the capability survived. */
  mrb_value stale = cached[0];
  SAY("CDP 3: stale read COMPLETED -- not blocked\n");
  return stale;
}

/* Recurse deep enough that mruby must extend the VM stack. 200 frames is well
   past the initial allocation and cheap under TCG. */
#define CDP_RUBY                                                               \
  "def deepen(n = 0)\n"                                                        \
  "  return n if n > 200\n"                                                    \
  "  deepen(n + 1)\n"                                                          \
  "end\n"                                                                      \
  "stale\n"

static int run_cdp(mrb_state *mrb) {
#ifdef MRUBY_PROBE_CDP_CONTROL
  /* The control arm: same allocations, same frees, revocation disabled. */
  xlang_set_no_revoke();
  SAY("CDP ARM: control (revocation DISABLED)\n");
#else
  SAY("CDP ARM: revoke-on-free (the shipped configuration)\n");
#endif

  mrb_define_method(mrb, mrb->object_class, "stale", cdp_stale, MRB_ARGS_NONE());
  mrb_load_string(mrb, CDP_RUBY);
  if (mrb->exc) {
    SAY("CDP: an exception reached the top level\n");
    mrb->exc = 0;
  }
  say_arena("after-cdp");
  SAY("__CAPSTONE_MRUBY_CDP_DONE__\n");
  return 0;
}
#endif

#ifdef MRUBY_PROBE_MRBTEST
/* MRUBY'S OWN TEST SUITE, RUN INSIDE THE DOMAIN.
 *
 * The sequence is driver.c's main(), minus the command line: define the test
 * driver's helpers on Kernel, load test/assert.rb, run the tests, ask for the
 * report. It is spelled out here rather than calling that main because a domain
 * has no argv and because mrb_open_allocf is what gives this probe its memory
 * accounting -- the outer state has to be ours.
 *
 * The 43 core test files each get their OWN interpreter, opened by the generated
 * gem_test.c with mrb_open_core, so mrbgemtest_init is 43 full open/run/close
 * cycles. That is a considerably harder exercise of the port than the probe
 * above: every one of them builds a symbol table, a class hierarchy and a GC
 * arena from scratch and tears them down again.
 *
 * REPORTING IS DOUBLE. The suite prints its own progress through t_print, which
 * is C in driver.c and goes out through stdio -- fine while stdio works. The
 * stage markers below go through SAY, the raw hostcall, so that a failure INSIDE
 * stdio or inside the VM still says which stage it reached. The probe above
 * exists because of exactly that distinction. */
extern const uint8_t mrbtest_assert_irep[];
void mrb_init_test_driver(mrb_state *mrb, mrb_bool verbose);
void mrbgemtest_init(mrb_state *mrb);

static int run_mrbtest(mrb_state *mrb) {
  mrb_init_test_driver(mrb, 0);
  if (mrb->exc) {
    SAY("MRBTEST FAIL: mrb_init_test_driver raised\n");
    return 20;
  }
  SAY("MRBTEST T1: test driver installed\n");
  say_mem("after-driver");

  mrb_load_irep(mrb, mrbtest_assert_irep);
  if (mrb->exc) {
    SAY("MRBTEST FAIL: assert.rb raised\n");
    return 21;
  }
  SAY("MRBTEST T2: assert.rb loaded\n");
  say_mem("after-assert");

  /* The long one: 43 interpreters. Anything printed between here and T3 is the
     suite's own per-assertion output. */
  SAY("MRBTEST T3: running core tests\n");
  mrbgemtest_init(mrb);
  if (mrb->exc) {
    SAY("MRBTEST FAIL: a test file raised out of its own interpreter\n");
    return 22;
  }
  SAY("MRBTEST T4: all test files ran\n");
  say_mem("after-tests");

  /* `report` prints the Total/OK/KO/Crash block and returns whether everything
     passed. A false here is a REAL RESULT, not an error: the suite ran and some
     assertions failed, which is exactly the number this whole exercise is for.
     So it is reported as its own marker rather than folded into a failure. */
  mrb_value ok = mrb_funcall(mrb, mrb_top_self(mrb), "report", 0);
  if (mrb->exc) {
    SAY("MRBTEST FAIL: report raised\n");
    return 23;
  }
  SAY(mrb_test(ok) ? "MRBTEST T5: report says every assertion passed\n"
                   : "MRBTEST T5: report says some assertions failed\n");
  say_mem("at-report");
  return 0;
}
#endif

int capstone_main(void) {
  SAY("MRUBY S1: entered\n");

#if MRUBY_PROBE_STAGE > 0
  return run_stage();
#endif

  /* REPORT THE RUNTIME BASE, so a fault pc can be turned into a symbol.
   * The domain is linked at 0x10000 but loaded wherever the guest's mmap put
   * it, and the fault line gives an absolute pc. Two runs with different heap
   * sizes faulted at the SAME pc (0x101600250), which already refutes heap
   * exhaustion; what it does not give is WHICH function. base = this value
   * minus capstone_main's link address, and then llvm-nm on the .dom answers
   * it. Cheaper than another round of guessing from address arithmetic. */
  {
    char b[96];
    /* The STACK too. The fault is on setjmp's first store, `sd ra, 0(a0)`,
       with a0 1.5 MB below the domain base -- so the jmp_buf address mruby
       passed is not where a stack local should be. Printing the frame address
       says whether the stack itself is misplaced or only that one pointer. */
    int n = snprintf(b, sizeof b, "MRUBY BASE: capstone_main=%lx frame=%lx\n",
                     (unsigned long)(uintptr_t)(void *)&capstone_main,
                     (unsigned long)(uintptr_t)__builtin_frame_address(0));
    if (n > 0)
      __capstone_hc_write(1, b, (unsigned long)n);
  }

  mrb_state *mrb = mrb_open_allocf(probe_allocf, 0);
  if (!mrb) {
    SAY("MRUBY FAIL: mrb_open returned null\n");
    say_mem("at-open-failure");
    return 1;
  }
  SAY("MRUBY S2: mrb_open ok\n");
  say_mem("after-open");
  say_arena("after-open");

  mrb_value v = mrb_load_irep(mrb, probe_irep);
  if (mrb->exc) {
    SAY("MRUBY FAIL: exception raised while running the irep\n");
    mrb_close(mrb);
    return 2;
  }
  SAY("MRUBY S3: irep executed\n");

  /* mrb_fixnum_p / mrb_fixnum, NOT the mrb_integer_* spelling: the corpus
     pins nine mruby versions spanning 2017-2026 and the newer names do not
     exist before 3.0, while the older ones are kept as aliases throughout.
     Modernising these breaks every pre-3.0 row. */
  if (!mrb_fixnum_p(v)) {
    SAY("MRUBY FAIL: result is not an Integer\n");
    mrb_close(mrb);
    return 3;
  }
  if (mrb_fixnum(v) != 400) {
    SAY("MRUBY FAIL: t[19] is not 400\n");
    mrb_close(mrb);
    return 4;
  }
  SAY("MRUBY S4: t[19] == 400\n");

#ifdef MRUBY_PROBE_MRBTEST
  {
    int rc = run_mrbtest(mrb);
    mrb_close(mrb);
    SAY("MRUBY S5: state closed\n");
    if (rc == 0)
      SAY("__CAPSTONE_MRBTEST_COMPLETED__\n");
    return rc;
  }
#endif
#ifdef MRUBY_PROBE_ROW
  run_row(mrb);
  mrb_close(mrb);
  return 0;
#endif
#ifdef MRUBY_PROBE_CDP
  run_cdp(mrb);
  mrb_close(mrb);
  return 0;
#endif

#ifdef MRUBY_PROBE_PARSER
  /* THE SAME CHUNK, AS RUBY SOURCE. Same text as probe.rb, so bytecode and
     parser differ in exactly one thing: who turned the source into an irep.
     This is what the bytecode route exists to avoid, and the reason it exists
     is one line in mruby's parser (see patch-parser.py). */
  SAY("MRUBY S6: parsing Ruby source\n");
  mrb_value w = mrb_load_string(mrb, PROBE_SOURCE);
  if (mrb->exc) {
    SAY("MRUBY FAIL: exception while parsing or running the source\n");
    mrb_close(mrb);
    return 5;
  }
  if (!mrb_fixnum_p(w) || mrb_fixnum(w) != 400) {
    SAY("MRUBY FAIL: parsed source did not produce 400\n");
    mrb_close(mrb);
    return 6;
  }
  SAY("MRUBY S7: parsed source produced 400\n");
#endif

  /* Closing exercises the GC's teardown over everything the chunk allocated,
     which is where a capability the collector cannot follow would show up. */
  mrb_close(mrb);
  SAY("MRUBY S5: state closed\n");
  say_mem("at-exit");
  say_arena("at-exit");

  SAY("__CAPSTONE_MRUBY_PROBE_PASSED__\n");
  return 0;
}
