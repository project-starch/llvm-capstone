/* Freestanding silicon-ladder BASELINE controller (FPGA/CapliFive).
 *
 * The denominator for the spatial-safety overhead measurement. `ladder_perf_ctl`
 * runs each rung as a pure-capability Capstone domain and reports the mcycle delta
 * across the compute; this binary runs the SAME kernels as ordinary RISC-V code in
 * Linux userspace and reports the same delta, so the ratio prices the capability
 * ABI + its hardware enforcement. Everything else is held fixed: same board, same
 * clock, same DRAM, same clang at the same -O level (see ladder_base_kern.c).
 *
 * It creates no domain and never opens /dev/capstone, so -- unlike the capability
 * sweep -- it needs no power-cycle per rung: the 2.5 min/rung cost there is domain
 * creation against a cold icache (the second-same-VA-domain hang), which does not
 * arise here. All seven rungs run in ONE boot, from one transferred binary.
 *
 *   ladder_base_ctl probe <cycle|time|instret>   -- can this counter be read at all?
 *   ladder_base_ctl all [ctr]                    -- run every rung
 *   ladder_base_ctl <rung> [ctr]                 -- run one rung
 *
 * COUNTER, and why `probe` exists. The domain half reads the M-mode `mcycle` CSR
 * (0xB00), which the monitor leaves domain-readable. Userspace cannot read 0xB00;
 * the U-mode mirror is `cycle` (0xC00), which on CVA6 counts the same underlying
 * cycles -- but only if [m|s]counteren.CY is set, and ladder_perf_domain.h records
 * that this board gates the unprivileged counter for the domain. Whether it is
 * gated for ordinary Linux userspace too is unknown, and a gated read traps as an
 * illegal instruction. With no libc there is no SIGILL handler to catch it, so
 * each probe is its OWN process: a trap kills that one invocation and prints
 * "Illegal instruction" to the shell, leaving the boot intact to try the next.
 * Probe before measuring; never assume the counter.
 *
 * Same freestanding soft-float model as ladder_perf_ctl.c: links NO glibc (the
 * board rejects glibc's hard-float fsd), built -nostdlib -static -march=rv64imac
 * -mabi=lp64, own _start, raw Linux syscalls, integer-only I/O.
 */

typedef unsigned long ulong;

/* ------------------------------------------------------------------ *
 *  Raw Linux syscalls (RISC-V 64, generic ABI) -- no libc.
 * ------------------------------------------------------------------ */
static inline long _sys(long n, long a0, long a1, long a2) {
  register long ra0 __asm__("a0") = a0;
  register long ra1 __asm__("a1") = a1;
  register long ra2 __asm__("a2") = a2;
  register long rn  __asm__("a7") = n;
  __asm__ volatile("ecall" : "+r"(ra0) : "r"(ra1), "r"(ra2), "r"(rn) : "memory");
  return ra0;
}
#define SYS_write       64
#define SYS_exit_group  94

static long sys_write(int fd, const void *b, ulong n) {
  return _sys(SYS_write, fd, (long)b, (long)n);
}
static void sys_exit(int code) { _sys(SYS_exit_group, code, 0, 0); }

/* ------------------------------------------------------------------ *
 *  Tiny libc. memset/memcpy are exported unmangled because clang may emit
 *  calls to them for aggregate init/copy inside the kernels even at -O0.
 * ------------------------------------------------------------------ */
void *memset(void *d, int c, ulong n) {
  unsigned char *p = d;
  while (n--) *p++ = (unsigned char)c;
  return d;
}
void *memcpy(void *d, const void *s, ulong n) {
  unsigned char *dp = d;
  const unsigned char *sp = s;
  while (n--) *dp++ = *sp++;
  return d;
}
static void puts_(const char *s) {
  ulong n = 0;
  while (s[n]) n++;
  sys_write(1, s, n);
}
static void putu_(ulong v) {         /* decimal */
  char buf[24];
  int i = 24;
  if (v == 0) buf[--i] = '0';
  while (v) { buf[--i] = (char)('0' + v % 10); v /= 10; }
  sys_write(1, buf + i, (ulong)(24 - i));
}
static int streq_(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return *a == *b;
}

/* ------------------------------------------------------------------ *
 *  Counters. Each read is a distinct inline asm so the CSR number is
 *  immediate; a gated CSR traps here rather than returning garbage.
 * ------------------------------------------------------------------ */
static inline ulong rd_cycle(void) {
  ulong v; __asm__ volatile("csrr %0, cycle" : "=r"(v)); return v;
}
static inline ulong rd_time(void) {
  ulong v; __asm__ volatile("csrr %0, time" : "=r"(v)); return v;
}
static inline ulong rd_instret(void) {
  ulong v; __asm__ volatile("csrr %0, instret" : "=r"(v)); return v;
}

enum { CTR_CYCLE = 0, CTR_TIME = 1, CTR_INSTRET = 2 };
static const char *const CTR_NAME[] = { "cycle", "time", "instret" };

static ulong rd_ctr(int which) {
  switch (which) {
    case CTR_TIME:    return rd_time();
    case CTR_INSTRET: return rd_instret();
    default:          return rd_cycle();
  }
}
static int ctr_by_name(const char *s) {
  for (int i = 0; i < 3; i++) if (streq_(s, CTR_NAME[i])) return i;
  return -1;
}

/* ------------------------------------------------------------------ *
 *  The rungs. One extern per kernel TU (see ladder_base_kern.c); the
 *  names match the capability sweep's rung names so the two result
 *  tables join on the first column.
 * ------------------------------------------------------------------ */
unsigned base_null(void);
unsigned base_matmult_int(void);
unsigned base_coremark_matrix(void);
unsigned base_rv8_primes(void);
unsigned base_beebs_crc32(void);
unsigned base_beebs_insertsort(void);
unsigned base_beebs_prime(void);
unsigned base_beebs_recursion(void);
unsigned base_beebs_bs(void);
unsigned base_beebs_janne(void);
unsigned base_beebs_fibcall(void);
unsigned base_beebs_fac(void);
unsigned base_beebs_cnt(void);
unsigned base_beebs_duff(void);
unsigned base_ctrsanity(void);
unsigned base_ctrsanity4(void);

struct rung { const char *name; unsigned (*fn)(void); };
static const struct rung RUNGS[] = {
  { "null",             base_null             },
  { "matmult_int",      base_matmult_int      },
  { "coremark_matrix",  base_coremark_matrix  },
  { "rv8_primes",       base_rv8_primes       },
  { "beebs_crc32",      base_beebs_crc32      },
  { "beebs_insertsort", base_beebs_insertsort },
  { "beebs_prime",      base_beebs_prime      },
  { "beebs_recursion",  base_beebs_recursion  },
  { "beebs_bs",         base_beebs_bs         },
  { "beebs_janne",      base_beebs_janne      },
  /* This table is hand-maintained and is SEPARATE from the RUNGS list in
     build-ladder-base-fpga.sh. Adding a rung there but not here builds fine and
     then reports "--" for every column at run time -- which cost a board boot on
     2026-07-27. Add to both, always. */
  { "beebs_fibcall",    base_beebs_fibcall    },
  { "beebs_fac",        base_beebs_fac        },
  { "beebs_cnt",        base_beebs_cnt        },
  { "beebs_duff",       base_beebs_duff       },
  { "ctrsanity",        base_ctrsanity        },
  { "ctrsanity4",       base_ctrsanity4       },
};
#define NRUNGS ((int)(sizeof RUNGS / sizeof RUNGS[0]))

/* Each rung is measured TWICE, back to back, and both passes are reported.
 *
 * Pass 1 is cold: it is the first touch of that kernel's own .bss arrays, so it
 * pays Linux demand-paging faults. The capability domain has no paging at all and
 * pays none of that, which is exactly why the naive pass-1-vs-domain ratio came
 * out at 0.54 for beebs_prime -- an impossible "capability is 1.8x faster". Pass 2
 * runs with those pages already mapped, so (pass1 - pass2) isolates the
 * paging cost and pass 2 is the number comparable to the domain.
 *
 * This is only legitimate where the kernel is IDEMPOTENT, and the retvals say so
 * rather than a comment: pass 2 reports its own retval, so a stateful kernel
 * (beebs_crc32, whose pseudo-random generator carries state) announces itself by
 * returning something different, and its warm number is discarded. Verified on the
 * host beforehand: 6 of 7 are idempotent across repeated calls, crc32 is not.
 *
 * Both counters bracket the SAME execution: cycles outermost, instret inside, so
 * the instret delta excludes the outer CSR reads. Reporting instructions next to
 * cycles separates "the capability build executes more instructions" from "the
 * same instructions cost more cycles" -- two very different findings that a cycle
 * count alone conflates. */
static void run_pass(const struct rung *r, int pass) {
  ulong c0 = rd_cycle(), i0 = rd_instret();
  unsigned v = r->fn();
  ulong i1 = rd_instret(), c1 = rd_cycle();
  puts_("BASE RESULT "); puts_(r->name);
  puts_(" pass=");       putu_((ulong)pass);
  puts_(" retval=");     putu_((ulong)v);
  puts_(" cycles=");     putu_(c1 - c0);
  puts_(" instret=");    putu_(i1 - i0);
  puts_("\n");
}

static void run_one(const struct rung *r, int ctr) {
  (void)ctr;                    /* both counters are read; see run_pass */
  run_pass(r, 1);
  run_pass(r, 2);
}

/* ------------------------------------------------------------------ *
 *  Entry: parse argc/argv off the stack, no libc runtime.
 * ------------------------------------------------------------------ */
void _start_c(long *sp) {
  long argc = sp[0];
  char **argv = (char **)(sp + 1);
  if (argc < 2) {
    puts_("usage: ladder_base_ctl probe <cycle|time|instret>\n"
          "       ladder_base_ctl <all|rung> [cycle|time|instret]\n");
    sys_exit(2);
  }

  if (streq_(argv[1], "probe")) {
    int c = (argc >= 3) ? ctr_by_name(argv[2]) : CTR_CYCLE;
    if (c < 0) { puts_("BASE PROBE bad-counter\n"); sys_exit(2); }
    /* If the CSR is gated this traps and the process dies here -- that IS the
       result, reported by the shell as an illegal instruction. */
    ulong a = rd_ctr(c);
    ulong b = rd_ctr(c);
    puts_("BASE PROBE "); puts_(CTR_NAME[c]);
    puts_(" ok v="); putu_(a);
    puts_(" delta="); putu_(b - a);
    puts_("\n");
    sys_exit(0);
  }

  int ctr = CTR_CYCLE;
  if (argc >= 3) {
    ctr = ctr_by_name(argv[2]);
    if (ctr < 0) { puts_("BASE bad-counter\n"); sys_exit(2); }
  }

  if (streq_(argv[1], "all")) {
    for (int i = 0; i < NRUNGS; i++) run_one(&RUNGS[i], ctr);
    puts_("BASE DONE all\n");
    sys_exit(0);
  }
  for (int i = 0; i < NRUNGS; i++) {
    if (streq_(argv[1], RUNGS[i].name)) { run_one(&RUNGS[i], ctr); sys_exit(0); }
  }
  puts_("BASE unknown-rung\n");
  sys_exit(2);
}

__asm__(
  ".section .text\n"
  ".globl _start\n"
  "_start:\n"
  "  .option push\n"
  "  .option norelax\n"
  "  lla gp, __global_pointer$\n"
  "  .option pop\n"
  "  mv a0, sp\n"
  "  andi sp, sp, -16\n"
  "  call _start_c\n"
  "  ebreak\n"
);
