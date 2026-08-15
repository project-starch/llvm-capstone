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

int capstone_main(void) {
  SAY("MRUBY S1: entered\n");

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
