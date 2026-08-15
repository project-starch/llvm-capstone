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
static void say_mem(const char *stage) {
  char b[128];
  int n = snprintf(b, sizeof b,
                   "MRUBY MEM %s: requested=%lu peak=%lu calls=%lu fails=%lu\n",
                   stage, mem_live, mem_peak, mem_calls, mem_fails);
  if (n > 0)
    __capstone_hc_write(1, b, (unsigned long)n);
}

/* Generated by mruby's build system from the gembox; a core-only domain has no
   gems, so the list is empty rather than absent. Defined here so the absence is
   deliberate and visible instead of a link error someone silences later. */
void mrb_init_mrbgems(mrb_state *mrb) { (void)mrb; }
void mrb_final_mrbgems(mrb_state *mrb) { (void)mrb; }

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
      long got = mrb_integer_p(rv) ? (long)mrb_integer(rv) : -1;
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
    if (mrb_integer_p(v6) && mrb_integer(v6) == 400)
      SAY("MRUBY STAGE 6: t[19] == 400\n");
    else
      SAY("MRUBY STAGE 6: ran but did not produce 400\n");
  }
#else
  {
    mrb_value v5 = mrb_load_irep(mrb, probe_irep);
    if (mrb->exc) {
      SAY("MRUBY STAGE 5: exception while running our own irep\n");
    } else if (mrb_integer_p(v5) && mrb_integer(v5) == 400) {
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

  mrb_value v = mrb_load_irep(mrb, probe_irep);
  if (mrb->exc) {
    SAY("MRUBY FAIL: exception raised while running the irep\n");
    mrb_close(mrb);
    return 2;
  }
  SAY("MRUBY S3: irep executed\n");

  if (!mrb_integer_p(v)) {
    SAY("MRUBY FAIL: result is not an Integer\n");
    mrb_close(mrb);
    return 3;
  }
  if (mrb_integer(v) != 400) {
    SAY("MRUBY FAIL: t[19] is not 400\n");
    mrb_close(mrb);
    return 4;
  }
  SAY("MRUBY S4: t[19] == 400\n");

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
  if (!mrb_integer_p(w) || mrb_integer(w) != 400) {
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

  SAY("__CAPSTONE_MRUBY_PROBE_PASSED__\n");
  return 0;
}
