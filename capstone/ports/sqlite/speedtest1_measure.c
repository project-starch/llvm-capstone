/* SQLite's own speedtest1 benchmark, run INSIDE a capability domain.
 *
 * Selected by DOMAIN_SRC, so build-sqlite-silicon.sh copies this file over
 * sqlite_capstone_domain.c and the amalgam translation unit #includes it. That placement is not
 * incidental: under -capstone-gp-captable globals are numbered per module and positionally against
 * one runtime cap-table, so speedtest1's file-scope state -- and the stdio shim's -- MUST share the
 * unit with SQLite's, or the two silently disagree about which slot is whose. That is why the shim
 * is #included as a .c below rather than linked as its own object: it owns five globals.
 *
 * Everything domain-specific is here and amounts to five things: where the arguments come from,
 * where the output goes, what times the run, how much heap the run used, and how a fatal error is
 * reported instead of wedging. speedtest1.c itself is unmodified.
 *
 * WHICH TESTSETS RUN (measured 2026-09-10 against our define set): main, orm, parsenumber. The
 * default, mix1, does NOT -- it contains json (omitted), rtree (not enabled), cte and star
 * (decimal literals, which the tokenizer rejects under SQLITE_OMIT_FLOATING_POINT) and fp (needs
 * round(), i.e. math functions). An invocation must therefore name --testset explicitly.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "capstone_sqlite_stdio.h"

/* The monitor's share selector, as sqlite_capstone_domain.c defines it. It is not in the hostcall
 * header because it belongs to the domain/monitor interface, not to the host transport. */
#define CAPSTONE_DPI_REGION_SHARE 1U

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif

/* Charged against dom_data, not image space, under the gp-captable ABI -- see the note in
 * sqlite_capstone_domain.c. The build passes -DSQLITE_HEAP_SIZE and domdata-budget.py is what says
 * whether the choice fits; do not raise it here on the assumption that it will -- 2.5 MiB passed
 * that gate and then faulted before the domain was entered.
 *
 * HIGHWATER prints "n/a", and that is the only value it can hold today: the define set carries
 * -DSQLITE_DEFAULT_MEMSTATUS=0, so the accounting is never updated. An earlier version of this
 * paragraph promised a high-water number, and the field read 0 on every run while claiming to
 * answer "did the heap fit". The heap MINIMUM per testset is measured natively instead; see
 * tools/speedtest1-heap-sweep.sh. */
#if defined(CAPSTONE_SPEEDTEST1_REGION_ARENA) && !defined(CAPSTONE_SPEEDTEST1_BASELINE)
/* THE ARENA COMES FROM A SHARED REGION, NOT FROM .bss. Under the gp-captable ABI a .bss array is
 * carved out of dom_data, and dom_data is ONE __get_free_pages allocation capped at 4 MiB on this
 * kernel (MAX_ORDER is 10 and INCLUSIVE from Linux 6.4, CONFIG_ARCH_FORCE_MAX_ORDER unset). That
 * ceiling is what blocks `json`, which needs ~6 MiB, and the whole --size axis. A region is CMA-
 * backed and has been demonstrated at 130 MiB.
 *
 * The pattern is not invented here: six domains already take a third region as an allocator arena,
 * and sqlite_row3_b2_domain.c uses one as SQLite's entire heap. The size is read off the grant's own
 * bounds rather than passed as a define, as revoke_on_free_alloc.h does -- so there is no second
 * place for it to disagree with the host.
 *
 * The baseline build never reaches this: it has no domain, no monitor and no regions, so it keeps
 * the static array unconditionally. */
static unsigned char *sqlite_arena;
static unsigned long sqlite_arena_len;
#define SPEED_ARENA_PTR ((void *)sqlite_arena)
#define SPEED_ARENA_LEN ((int)sqlite_arena_len)
#else
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
#define SPEED_ARENA_PTR ((void *)sqlite_heap)
#define SPEED_ARENA_LEN ((int)sizeof(sqlite_heap))
#endif

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
static char *baseline_args;
static unsigned long baseline_args_len;
static int baseline_sqlite_ready;
#endif

/* Output grows from offset 0; the arguments sit in the top half, exactly as an SLT input does, so
 * the two cannot collide. */
#define SPEED_OUT_LIMIT SQLITE_HC_SPEED_ARGS_OFF

/* A TAIL THE BENCHMARK CANNOT CONSUME, and it is the fix for a check that failed open.
 *
 * The report line -- cycles, DROPPED, the RAN marker -- is written through the SAME sink as
 * speedtest1's own output, and after it. So on a run that filled the region, the sink returned 0 for
 * everything that followed and the report was dropped too: `DROPPED 0` and a missing
 * SPEEDTEST1-CYCLES were the plan's two integrity checks, and BOTH were unobservable in exactly the
 * case they existed to detect. A truncated run looked like a run that had nothing to say.
 *
 * speedtest1's output is now bounded below the limit and the report writes into the reserve, so a
 * truncated run still reports its own truncation. 512 bytes is comfortably above the report's
 * longest form; the benchmark loses that much of a tail it was going to lose anyway. */
#define SPEED_REPORT_RESERVE 512UL

static unsigned long output_used;
#ifdef CAPSTONE_SPEEDTEST1_INSTRET
static unsigned long speed_instrs;
#endif
/* Set only while speedtest1_report is writing, which is the only thing allowed into the reserve. */
static int output_in_report;

static unsigned long speed_out_bound(void) {
  return output_in_report ? SPEED_OUT_LIMIT : SPEED_OUT_LIMIT - SPEED_REPORT_RESERVE;
}

/* The sink the stdio shim formats into. Returns bytes ACCEPTED so the shim can count what the
 * region refused: a report that quietly loses its tail is indistinguishable from one that had
 * nothing to say. */
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
/* THE BASELINE'S SINK. Same formatter, same bound, but the destination is a plain buffer the
   harness supplies instead of a shared region -- the baseline is ordinary Linux userspace and has
   no domain, no monitor and no regions. Everything between the counter brackets is otherwise the
   identical translation unit, which is the whole point of a matched denominator. */
static char *baseline_out;
static unsigned long baseline_out_cap;

unsigned long capstone_stdio_sink(const char *text, unsigned long n) {
  unsigned long i = 0;
  if (!baseline_out)
    return 0;
  {
    unsigned long cap = baseline_out_cap > SPEED_REPORT_RESERVE && !output_in_report
                            ? baseline_out_cap - SPEED_REPORT_RESERVE
                            : baseline_out_cap;
    while (i < n && output_used + 1 < cap)
      baseline_out[output_used++] = text[i++];
  }
  return i;
}
#else
unsigned long capstone_stdio_sink(const char *text, unsigned long n) {
  unsigned long i = 0;
  char *payload;
  if (!hostcall_metadata || !hostcall_payload)
    return 0;
  payload = (char *)hostcall_payload;
  while (i < n && output_used + 1 < speed_out_bound())
    payload[output_used++] = text[i++];
  hostcall_metadata->length = output_used;
  return i;
}
#endif

static void out(const char *s) {
  unsigned long n = 0;
  while (s[n])
    n++;
  (void)capstone_stdio_sink(s, n);
}

static void out_ulong(unsigned long v) {
  char d[24];
  unsigned n = 0;
  if (!v) {
    out("0");
    return;
  }
  while (v) {
    d[n++] = (char)('0' + (v % 10));
    v /= 10;
  }
  while (n)
    (void)capstone_stdio_sink(&d[--n], 1);
}

/* mcycle, for the same reason ladder_perf_domain.h reads it: the board gates the unprivileged
 * counter for domains. minstret is deliberately NOT read -- it is implemented for rungs but no
 * recorded board result carries an instret value, so this vehicle reports CYCLES ONLY until someone
 * demonstrates the other counter on this path. Decided here, in the source, rather than discovered
 * in a transcript. */
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
/* USERSPACE CANNOT READ 0xB00. The U-mode mirror `cycle` (0xC00) counts the same underlying cycles
   on CVA6, and `instret` beside it is what makes the comparison a ratio of instructions as well as
   of cycles. Both are gated by [m|s]counteren, and a gated read traps as an illegal instruction --
   with no libc there is no handler, so the harness probes each counter in its own invocation before
   measuring. Same reasoning, and the same wording, as ladder_base_ctl.c:20. */
static unsigned long rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, cycle" : "=r"(v));
  return v;
}
static unsigned long rd_instret_(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, instret" : "=r"(v));
  return v;
}
#else
static unsigned long rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

#ifdef CAPSTONE_SPEEDTEST1_INSTRET
/* minstret (0xB02), and it is OPT-IN AND SHIPS AS ITS OWN IMAGE for a specific reason.
 *
 * Why it is wanted: a cycle count alone cannot separate "the capability ABI retires MORE
 * instructions" from "the same instructions cost MORE cycles", and the measurements doc treats that
 * split as the point of the exercise. The baseline half already reports instret; only this side was
 * missing, so a board run without it answers half the question.
 *
 * Why it is not simply switched on: on 2026-07-26 adding two csrr minstret plus three stores to a
 * beebs_prime rung flipped it from correct to DETERMINISTICALLY MISCOMPUTING on silicon --
 * 1087631800 against an oracle of 582955588, reproduced bit-identically across two sessions, for
 * +172 cycles. The later bisection (ladder_perf_domain.h) cleared the CSR READS and left the STORE
 * as the surviving suspect, with the store's offset and "one more region store" both still live.
 * So the hazard here is a WRONG ANSWER, not a trap.
 *
 * Why that is tolerable: every arm carries a verification hash that must equal the native value, so
 * a miscomputing arm fails its own check and is discarded. The cost is a lost arm, not a published
 * wrong number. Ship it as a SEPARATE image and stage it LAST, so the three verified arms keep
 * their hashes byte-for-byte and nothing of value sits behind it. */
static unsigned long rd_minstret(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, minstret" : "=r"(v));
  return v;
}
#endif
#endif

/* ------------------------------------------------------------------- the abort path
 *
 * speedtest1's fatal_error calls exit(). A domain cannot exit a process and must not return into
 * speedtest1, whose caller carries on with a null statement.
 *
 * WHAT DOES NOT WORK, AND THE CLAIM THIS RETRACTS. An earlier version of this file said the
 * deliberate capability fault below "terminates the domain and returns to the host, so the core
 * survives and the boot continues". THAT IS WRONG FOR QEMU AND UNVERIFIED FOR THE BOARD. A domain
 * installs no ctvec -- only the monitor does -- so the fault cannot be delivered horizontally, and
 * capstone-qemu's cpu_helper.c:1866-1887 says so in as many words: it prints
 * "domain halted by capability fault" and calls exit(0), noting that returning control to the host
 * launcher "requires a monitor-side fault-return path that does not exist yet". The monitor's
 * fault_return_from_domain is therefore never reached on this path. Measured, not argued: arms A
 * and C of the 2026-09-10 bisection reached this function (heap too small -> SQLITE_NOMEM ->
 * fatal_error) and the emulator halted with no host-side return.
 *
 * WHY THE FAULT STAYS ANYWAY. The alternatives are worse. A spin hangs the emulator until its
 * timeout and wedges the core on the board; the fault at least ends QEMU immediately and flushes,
 * and on the board it is at worst equal to a wedge and at best a clean fault return, since the FPGA
 * monitor does install _cap_trap_entry as the domain's trap vector (sbi_capstone.c:991-1024) and
 * its unhandled-cause arm ends in fault_return_from_domain. Not an ILLEGAL instruction: on FPGA
 * that lands at the ILLX site, which still ends in while(1).
 *
 * WHAT ACTUALLY KEEPS A RUN ALIVE is not this path -- it is not reaching it. The heap minimum for
 * each testset and size is measured (see run-speedtest1-measure.sh), the host refuses an invocation
 * that cannot name a testset, and a stage that might abort goes LAST in a boot.
 *
 * The report is still written into the payload before faulting. It costs nothing, and it is the
 * only discriminator between a deliberate abort and a genuine capability defect if the board's
 * vector does take. */
static int exit_taken;
static int exit_code_seen;

static void speedtest1_report(unsigned long cycles, int aborted, int rc);

void capstone_stdio_on_exit(int code) {
  unsigned long tmp = 0;
  unsigned long notcap = 0;
  exit_code_seen = code;
  exit_taken = 1;
  /* Cycles are meaningless on this path (the run did not finish), so report 0 and let the marker
     say why. The fatal message fatal_error already printed is above it in the payload. */
  speedtest1_report(0UL, 1, code);
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
  /* A baseline is an ordinary process and CAN exit. Forging a capability fault here would be
     meaningless -- the target has no capability instructions, and the static gate on the baseline
     build would reject one.
     BUT IT MUST FLUSH FIRST. The harness writes the buffer only after speedtest1_baseline_run
     RETURNS, and this path never returns, so an aborted baseline arm previously emitted NOTHING:
     no report, no DROPPED, no ABORTED marker -- identical on the wire to a segfault or to the
     gated-CSR death the probes exist to rule out. The domain arm at least leaves its report in the
     shared region for the host to dump; this one had no such backstop. */
  (void)tmp; (void)notcap;
  if (baseline_out && output_used) {
    register long a0 __asm__("a0") = 1;                     /* stdout */
    register long a1 __asm__("a1") = (long)baseline_out;
    register long a2 __asm__("a2") = (long)output_used;
    register long a7 __asm__("a7") = 64;                    /* __NR_write */
    __asm__ volatile("ecall" :: "r"(a0), "r"(a1), "r"(a2), "r"(a7) : "memory");
  }
  {
    register long a0 __asm__("a0") = (long)code;
    register long a7 __asm__("a7") = 93;   /* __NR_exit */
    __asm__ volatile("ecall" :: "r"(a0), "r"(a7) : "memory");
  }
#else
  /* ldc tmp, 0(notcap) -- rs1 holds an integer, so the capability load raises UNEXPECTED_CAP_TYPE. */
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(tmp) : "r"(notcap));
#endif
  /* Not reached. Kept only so a target where that instruction is somehow serviced does not fall
     through into speedtest1's caller with a half-finished statement. */
  for (;;)
    ;
}

/* ------------------------------------------------------------------- allocation census
 *
 * WHY THIS EXISTS, and why it is a counting wrapper rather than the allocator arm itself.
 *
 * The revoke-on-free allocator (revoke_on_free_alloc.h) is the security-relevant arm: every
 * allocation is an independently revocable capability, and a pointer SQLite freed FAULTS on any
 * later use. Its cost is stated in its own header and is structural: the arena is consumed by
 * one-way SPLIT and xFree can never return space, so `rof_carved_total` only grows. That means the
 * arena it needs is not the workload's PEAK live bytes -- it is the SUM OF EVERY ALLOCATION THE RUN
 * EVER MAKES.
 *
 * Building that arm costs a third shared region, a host argument form, and a second SQLite image
 * that competes for the one-image-per-boot slot. All of it is wasted if the sum exceeds what a
 * region can hold, and the region ceiling is now known to be around 4 MiB. So measure the sum
 * first. This wrapper is ten lines and answers it.
 *
 * IT PERTURBS WHAT IT MEASURES -- a few instructions and a counter update per allocation -- so it is
 * opt-in and its runs are diagnostics, never the reported cycle number. Same rule as
 * CAPSTONE_SPEEDTEST1_MEMSTATUS above. */
#ifdef CAPSTONE_SPEEDTEST1_ALLOCSTATS
static sqlite3_mem_methods census_inner;
static unsigned long census_bytes_total;   /* every byte ever handed out */
static unsigned long census_calls;
static unsigned long census_live;
static unsigned long census_peak;

static void *census_malloc(int n) {
  void *p = census_inner.xMalloc(n);
  if (p) {
    unsigned long got = (unsigned long)census_inner.xSize(p);
    census_bytes_total += got;
    census_live += got;
    ++census_calls;
    if (census_live > census_peak)
      census_peak = census_live;
  }
  return p;
}

static void census_free(void *p) {
  if (p) {
    unsigned long got = (unsigned long)census_inner.xSize(p);
    census_live = census_live > got ? census_live - got : 0;
  }
  census_inner.xFree(p);
}

/* A realloc is a fresh carve under the revoke-on-free allocator -- it mallocs, copies and frees --
   so it is counted as one, which is what makes this census predictive of that arena rather than of
   memsys5's. */
static void *census_realloc(void *p, int n) {
  unsigned long old = p ? (unsigned long)census_inner.xSize(p) : 0UL;
  void *q = census_inner.xRealloc(p, n);
  if (q) {
    unsigned long got = (unsigned long)census_inner.xSize(q);
    census_bytes_total += got;
    ++census_calls;
    census_live = census_live > old ? census_live - old : 0;
    census_live += got;
    if (census_live > census_peak)
      census_peak = census_live;
  }
  return q;
}

static int census_size(void *p) { return census_inner.xSize(p); }
static int census_roundup(int n) { return census_inner.xRoundup(n); }
static int census_init(void *a) { return census_inner.xInit(a); }
static void census_shutdown(void *a) { census_inner.xShutdown(a); }

/* FIELD ORDER MATTERS AND xSize IS EASY TO DROP: the struct is
   xMalloc, xFree, xRealloc, xSize, xRoundup, xInit, xShutdown, pAppData.
   A first version omitted xSize and shifted everything after it by one, which the compiler caught
   as three incompatible function-pointer initialisers rather than as the missing field it was. */
static const sqlite3_mem_methods census_methods = {
    census_malloc, census_free, census_realloc, census_size,
    census_roundup, census_init, census_shutdown, (void *)0};

static void census_install(void) {
  if (sqlite3_config(SQLITE_CONFIG_GETMALLOC, &census_inner) != SQLITE_OK)
    return;
  if (!census_inner.xMalloc)
    return;
  (void)sqlite3_config(SQLITE_CONFIG_MALLOC, &census_methods);
}
#endif

/* ------------------------------------------------------------------- the run */

/* speedtest1's own entry point, from the pinned source #included at the bottom of this file. */
int main(int argc, char **argv);

#define SPEED_MAX_ARGV 16

/* Split the argument text the host published into argv. Whitespace-separated, NUL-terminated in
 * place; the text lives in the shared region and is ours to modify. argv[0] is synthesised because
 * speedtest1's option loop skips it. */
static int build_argv(char *text, unsigned long len, char **argv) {
  int argc = 0;
  unsigned long i = 0;
  argv[argc++] = (char *)"speedtest1";
  while (i < len && argc < SPEED_MAX_ARGV - 1) {
    while (i < len && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n'))
      text[i++] = 0;
    if (i >= len || !text[i])
      break;
    argv[argc++] = &text[i];
    while (i < len && text[i] != ' ' && text[i] != '\t' && text[i] != '\n')
      i++;
    if (i < len)
      text[i++] = 0;
  }
  argv[argc] = 0;
  return argc;
}

/* One line, always emitted, on both the clean and the aborted path. Machine-readable because the
 * driver reads a transcript, not a file. HIGHWATER is the heap question answered as a number. */
static void speedtest1_report(unsigned long cycles, int aborted, int rc) {
  output_in_report = 1;   /* the reserve is for this line and nothing else */
  out("SPEEDTEST1-CYCLES ");
  out_ulong(cycles);
  /* HIGHWATER IS PRINTED AS "n/a" UNLESS IT IS REAL. The define set carries
     -DSQLITE_DEFAULT_MEMSTATUS=0, so sqlite3_memory_highwater() returns 0 -- and a 0 here reads
     exactly like "the run used no heap", which is the shape of wrong result this project keeps
     paying for. Enabling the accounting costs a counter update per allocation, i.e. it perturbs the
     thing being measured, so it is opt-in and the reading is absent rather than false by default.
     The heap MINIMUM per testset and size is measured natively instead; see
     run-speedtest1-measure.sh. */
#ifdef CAPSTONE_SPEEDTEST1_INSTRET
  out(" INSTRS ");
  out_ulong(speed_instrs);
#endif
  out(" HIGHWATER ");
#ifdef CAPSTONE_SPEEDTEST1_MEMSTATUS
  out_ulong((unsigned long)sqlite3_memory_highwater(0));
#else
  out("n/a");
#endif
  out(" HEAP ");
  out_ulong((unsigned long)SPEED_ARENA_LEN);
#ifdef CAPSTONE_SPEEDTEST1_ALLOCSTATS
  /* CARVED is the number the revoke-on-free arena would have to hold: the sum of every allocation
     the run ever made, because that allocator never returns space. PEAK is what a coalescing
     allocator needs. The gap between them IS the generality cost of revoke-on-free. */
  out(" CARVED ");
  out_ulong(census_bytes_total);
  out(" PEAK ");
  out_ulong(census_peak);
  out(" ALLOCS ");
  out_ulong(census_calls);
#endif
  out(" DROPPED ");
  out_ulong(capstone_stdio_dropped);
  out(" RC ");
  out_ulong((unsigned long)(rc < 0 ? 0 : rc));
  out("\n");
  out(aborted ? "__CAPSTONE_SPEEDTEST1_ABORTED__\n" : "__CAPSTONE_SPEEDTEST1_RAN__\n");
  output_in_report = 0;
}

static unsigned run_speedtest1(void) {
  char *argv[SPEED_MAX_ARGV];
  int argc;
  unsigned long c0, c1;
#ifdef CAPSTONE_SPEEDTEST1_INSTRET
  unsigned long i0, i1;
#endif
  unsigned long args_len;
  char *args;
  int rc;
  int i;

#ifdef CAPSTONE_SPEEDTEST1_BASELINE
  /* No region, so no region gate. The harness has already validated the argument text. */
  args = baseline_args;
  args_len = baseline_args_len;
  if (args_len == 0)
    return (unsigned)SQLITE_HC_ERR_BAD_INPUT;
#else
  /* The agreement gate every domain here has: host and domain are separate compilations that must
     share one region size, and a silent mismatch is destructive in whichever direction it goes. */
  if (!hostcall_metadata ||
      hostcall_metadata->result != (sqlite_hostcall_s64_t)SQLITE_HC_REGION_SIZE)
    return (unsigned)SQLITE_HC_ERR_REGION_MISMATCH;

  /* THE OPCODE IS CHECKED, not assumed. domain_main dispatches on `func` alone, so an --slt
     invocation against this image would otherwise read the .test file's leading bytes out of the
     region's top half and run them as a speedtest1 command line. */
  if ((unsigned long)hostcall_metadata->opcode != SQLITE_HC_OP_SPEED)
    return (unsigned)SQLITE_HC_ERR_BAD_OPCODE;

  args_len = (unsigned long)hostcall_metadata->offset;
  if (args_len == 0 || args_len >= SQLITE_HC_SPEED_MAX_ARGS)
    return (unsigned)SQLITE_HC_ERR_BAD_INPUT;
  args = (char *)hostcall_payload + SQLITE_HC_SPEED_ARGS_OFF;
#endif

  /* SQLITE_OMIT_AUTOINIT is in the define set, so the domain configures and initializes SQLite
     itself. A probe that skipped this faulted at the first API call, which presented as an
     instruction-fetch bug and was not one. */
#ifdef CAPSTONE_SPEEDTEST1_MEMSTATUS
  /* Opt-in, and it PERTURBS THE MEASUREMENT: every allocation gains a counter update. Use it to
     answer a heap question, never in a run whose cycle count is being reported. */
  (void)sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 1);
#endif
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
  /* ONCE PER PROCESS, not once per pass. The baseline runs the benchmark twice in one process to get
     a warm number, and SQLite refuses sqlite3_config after it is initialised -- the first attempt at
     a warm pass came back as 158 cycles carrying the CONFIG_HEAP marker. sqlite3_shutdown between
     passes was tried and leaves speedtest1's own globals pointing at freed state, so the second pass
     dies outright. Skipping the already-done setup is what remains, and the cost it removes from the
     warm pass is init, which is a few thousand instructions against a run of ~4.7e8 -- visible as
     part of the cold-minus-warm difference, and not material to the ratio. */
  if (!baseline_sqlite_ready) {
    if (sqlite3_config(SQLITE_CONFIG_HEAP, SPEED_ARENA_PTR, SPEED_ARENA_LEN, 64) != SQLITE_OK)
      return (unsigned)SQLITE_HC_ERR_CONFIG_HEAP;
#ifdef CAPSTONE_SPEEDTEST1_ALLOCSTATS
    census_install();
#endif
    if (sqlite3_initialize() != SQLITE_OK)
      return (unsigned)SQLITE_HC_ERR_INITIALIZE;
    baseline_sqlite_ready = 1;
  }
#else
  if (sqlite3_config(SQLITE_CONFIG_HEAP, SPEED_ARENA_PTR, SPEED_ARENA_LEN, 64) != SQLITE_OK)
    return (unsigned)SQLITE_HC_ERR_CONFIG_HEAP;
#ifdef CAPSTONE_SPEEDTEST1_ALLOCSTATS
  census_install();
#endif
  if (sqlite3_initialize() != SQLITE_OK)
    return (unsigned)SQLITE_HC_ERR_INITIALIZE;
#endif

  argc = build_argv(args, args_len, argv);

  /* Echo the arguments back. A size sweep whose size knob silently failed to arrive would otherwise
     read as a valid measurement of the wrong thing. */
  out("SPEEDTEST1-ARGS");
  for (i = 1; i < argc; i++) {
    out(" ");
    out(argv[i]);
  }
  out("\n");

#ifdef CAPSTONE_SPEEDTEST1_INSTRET
  i0 = rd_minstret();
#endif
  c0 = rd_mcycle();
  rc = main(argc, argv);
  c1 = rd_mcycle();
#ifdef CAPSTONE_SPEEDTEST1_INSTRET
  i1 = rd_minstret();
  speed_instrs = i1 - i0;
#endif

  speedtest1_report(c1 - c0, 0, rc);

  /* The 0x4EB. marker family the host and the drivers already read as a result.
     THIS VALUE IS A CONSTANT AND CANNOT DISTINGUISH ANYTHING. An earlier comment offered the low bit
     as a text-free "did it complete" discriminator; it is always 0, because the only writer of
     exit_taken is capstone_stdio_on_exit, which never returns. The abort case is signalled by the
     monitor's fault retval plus the payload marker, and by nothing here -- see the host's
     speed_args branch, which is what actually implements it. */
  (void)exit_taken;
  return 0x4EB10000UL;
}

#ifdef CAPSTONE_SPEEDTEST1_BASELINE
/* The baseline's entry, called by speedtest1_baseline.c. Returns bytes written into `out`; the
   cycle and instruction deltas come back through the two pointers. The harness lives OUTSIDE the
   brackets, exactly as ladder_base_ctl does, so its codegen cannot enter the measurement. */
/* THE WARM-UP, and what it does and does not warm.
 *
 * The method requires the baseline's WARM pass, because the first run pays Linux first-touch page
 * faults inside the bracket while the capability domain has no paging at all -- charging the
 * baseline for them once made capabilities look 1.8x FASTER. The obvious way to get a warm pass is
 * to run the benchmark twice in one process. THAT DOES NOT WORK HERE: speedtest1 is not re-entrant,
 * and a second call to its main dies (measured -- the process exits 1 after a clean cold pass).
 * sqlite3_shutdown between passes leaves its globals pointing at freed state and dies sooner.
 *
 * So the warm-up is a SEPARATE small workload that touches the same things: it initialises SQLite,
 * runs a create/insert/index/select/drop cycle through the parser, VDBE, B-tree and pager, and
 * writes one byte per page across the whole arena. What it warms is the SQLite code pages, the
 * memsys5 arena and the process's own stack. What it does NOT warm is any code path unique to
 * speedtest1 itself, which is a small fraction of the image; cold minus warm is reported so the size
 * of what was removed is visible rather than asserted. */
void speedtest1_baseline_warmup(void) {
  sqlite3 *db = 0;
  unsigned long i;
  if (!baseline_sqlite_ready) {
    if (sqlite3_config(SQLITE_CONFIG_HEAP, SPEED_ARENA_PTR, SPEED_ARENA_LEN, 64) != SQLITE_OK)
      return;
    if (sqlite3_initialize() != SQLITE_OK)
      return;
    baseline_sqlite_ready = 1;
  }
  for (i = 0; i < sizeof sqlite_heap; i += 4096)
    sqlite_heap[i] = 0;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK || !db)
    return;
  (void)sqlite3_exec(db, "CREATE TABLE w(a INTEGER PRIMARY KEY, b TEXT, c INTEGER);", 0, 0, 0);
  (void)sqlite3_exec(db, "INSERT INTO w VALUES(1,'aaa',1),(2,'bbb',2),(3,'ccc',3);", 0, 0, 0);
  (void)sqlite3_exec(db, "CREATE INDEX wi ON w(c,b);", 0, 0, 0);
  (void)sqlite3_exec(db, "SELECT count(*), max(c) FROM w WHERE c BETWEEN 1 AND 3;", 0, 0, 0);
  (void)sqlite3_exec(db, "UPDATE w SET b='ddd' WHERE a=2;", 0, 0, 0);
  (void)sqlite3_exec(db, "DELETE FROM w WHERE a=3;", 0, 0, 0);
  (void)sqlite3_exec(db, "DROP TABLE w;", 0, 0, 0);
  (void)sqlite3_close(db);
}

unsigned long speedtest1_baseline_run(char *args, unsigned long args_len,
                                      char *out, unsigned long out_cap,
                                      unsigned long *cycles, unsigned long *instrs,
                                      unsigned *marker) {
  unsigned long c0, c1, i0, i1;
  baseline_out = out;
  baseline_out_cap = out_cap;
  output_used = 0;
  baseline_args = args;
  baseline_args_len = args_len;
  i0 = rd_instret_();
  c0 = rd_mcycle();
  *marker = run_speedtest1();
  c1 = rd_mcycle();
  i1 = rd_instret_();
  if (cycles) *cycles = c1 - c0;
  if (instrs) *instrs = i1 - i0;
  return output_used;
}
#else
void domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)res;
#if defined(CAPSTONE_SPEEDTEST1_REGION_ARENA)
    else if (shared_region_count == 2) {
      /* DELIN FIRST. The grant arrives LINEAR and a linear capability is CONSUMED BY COPY, so
       * handing it to memsys5 -- which copies it into mem5.zPool and then derives every allocation
       * from it -- would destroy the original. sqlite_row5_domain.c:71-75 is the worked example.
       *
       * The LENGTH comes from the grant's own bounds, so the domain cannot disagree with the host
       * about the arena size the way the region-size define once let it. Taken BEFORE the delin,
       * while the bounds are still on the capability we were handed. */
      unsigned long base = (unsigned long)__builtin_capstone_cap_get_base((void *)res);
      unsigned long end = (unsigned long)__builtin_capstone_cap_get_end((void *)res);
      sqlite_arena = (unsigned char *)__builtin_capstone_cap_delin((void *)res);
      sqlite_arena_len = (end > base) ? (end - base) : 0UL;
    }
#endif
    ++shared_region_count;
    return;
  }
  if (hostcall_metadata)
    hostcall_metadata->length = 0;
  output_used = 0;
  *res = run_speedtest1();
}
#endif

/* The shim's five globals must be in THIS translation unit; see the header comment. */
#include "capstone_sqlite_stdio.c"

/* speedtest1 itself, last, so every stdio macro above is already in scope for it. The build
 * script stages the pinned source under this fixed name in the object directory (see
 * SQLITE_SPEEDTEST1_SRC in build-sqlite-silicon.sh), so there is no path knob to get wrong. */
#include "speedtest1.c"
