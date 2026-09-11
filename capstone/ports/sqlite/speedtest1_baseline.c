/* THE MATCHED BASELINE for the speedtest1 overhead measurement, and nothing else.
 *
 * The capability half runs speedtest1 inside a Capstone domain and reports the cycle delta across
 * the benchmark. This binary runs THE SAME TRANSLATION UNIT as ordinary RISC-V code in Linux
 * userspace and reports the same delta, so the ratio prices the capability ABI and its hardware
 * enforcement rather than two compilers or two source trees. Same clang, same -O, same defines,
 * same board, same clock, same DRAM; only -target differs. See
 * docs/ref/fpga-silicon-measurements-for-paper.md for the method and for why the baseline's WARM
 * pass is the one that counts.
 *
 * THIS FILE IS SCAFFOLDING AND SITS OUTSIDE THE COUNTER BRACKETS. It is built with buildroot gcc,
 * exactly as ladder_base_ctl.c is, precisely so that its codegen cannot enter the measurement. The
 * measured code is speedtest1_baseline_run() in the amalgam TU, built with the same clang as the
 * domain.
 *
 *   speedtest1_baseline probe <cycle|instret>       -- can this counter be read at all?
 *   speedtest1_baseline run <speedtest1 args...>    -- measure
 *
 * WHY `probe` EXISTS, and it is not a formality. The domain half reads the M-mode mcycle CSR
 * (0xB00), which the monitor leaves domain-readable. Userspace cannot read 0xB00; the U-mode
 * mirrors are `cycle` (0xC00) and `instret` (0xC02), and both are gated by [m|s]counteren. This
 * board is RECORDED as gating the unprivileged counter for domains, and whether it does so for
 * ordinary Linux userspace is unknown. A gated read traps as an illegal instruction, and with no
 * libc there is no handler -- so each probe is its own process: a trap kills that invocation and
 * leaves the boot intact for the next. PROBE BEFORE MEASURING; never assume the counter.
 *
 * Freestanding soft-float, like every other baseline here: no glibc (the board rejects glibc's
 * hard-float fsd), -nostdlib -static -march=rv64imac_zicsr -mabi=lp64, own _start, raw syscalls,
 * integer-only I/O.
 */

typedef unsigned long ulong;

static long _sys(long n, long a0, long a1, long a2) {
  register long x10 __asm__("a0") = a0;
  register long x11 __asm__("a1") = a1;
  register long x12 __asm__("a2") = a2;
  register long x17 __asm__("a7") = n;
  __asm__ volatile("ecall" : "+r"(x10) : "r"(x11), "r"(x12), "r"(x17) : "memory");
  return x10;
}

#define SYS_write 64
#define SYS_exit  93

static void wr(const char *s, ulong n) { (void)_sys(SYS_write, 1, (long)s, (long)n); }

static ulong slen(const char *s) { ulong n = 0; while (s[n]) n++; return n; }
static void puts_(const char *s) { wr(s, slen(s)); }

static void put_ulong(ulong v) {
  char d[24];
  int n = 0;
  if (!v) { puts_("0"); return; }
  while (v) { d[n++] = (char)('0' + (v % 10)); v /= 10; }
  while (n) { char c = d[--n]; wr(&c, 1); }
}

static int streq_(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return *a == *b;
}

/* Each read is its own inline asm so the CSR number is an immediate; a gated CSR then traps here
 * rather than quietly returning something. */
static ulong rd_cycle(void)   { ulong v; __asm__ volatile("csrr %0, cycle"   : "=r"(v)); return v; }
static ulong rd_instret(void) { ulong v; __asm__ volatile("csrr %0, instret" : "=r"(v)); return v; }
/* THE ONE READ THAT CANNOT WEDGE THE MACHINE. sbi_capstone.c:1850-1856 takes
 * CAUSE_ILLEGAL_INSTRUCTION in M-mode, services a `time` CSR read and RETURNS; everything else it
 * does not handle goes to the ILLX site and ends in while(1). So a gated `cycle` or `instret` from
 * userspace takes the whole boot, not just this process -- the comment in ladder_base_ctl.c that
 * says otherwise predates this monitor. `time` counts the mtime tick, half the core clock. */
static ulong rd_time(void)    { ulong v; __asm__ volatile("csrr %0, time"    : "=r"(v)); return v; }

/* The measured half, in the amalgam TU, and the warm-up that precedes it. */
ulong speedtest1_baseline_run(char *args, ulong args_len,
                              char *out, ulong out_cap,
                              ulong *cycles, ulong *instrs, unsigned *marker);
void speedtest1_baseline_warmup(void);

static char args_buf[512];
static char out_buf[64 * 1024];

/* NOT `main`. speedtest1.c defines its own main inside the amalgam TU, and this file links against
   that TU -- two definitions, and the linker says so. The entry point is _start below, which calls
   this directly, so the name is ours to choose. */
int baseline_main(int argc, char **argv) {
  ulong at = 0;
  ulong cycles = 0, instrs = 0, n, ticks = 0, t0 = 0;
  unsigned marker = 0;
  int i;
  int warm = 0;

  if (argc < 2) {
    puts_("usage: speedtest1_baseline probe <cycle|instret>\n");
    return 2;
  }
  if (argc >= 3 && streq_(argv[1], "probe")) {
    /* AN UNRECOGNISED COUNTER IS AN ERROR, NOT A SILENT FALLBACK. This used to read `cycle` for any
       name that was not exactly "instret", so `probe instrret` printed
       "BASELINE-PROBE instrret = 1234567" -- a successful-looking probe of a counter it never
       touched. The whole purpose of this subcommand is to refuse to assume the counter. */
    ulong v;
    if (streq_(argv[2], "instret"))      v = rd_instret();
    else if (streq_(argv[2], "cycle"))   v = rd_cycle();
    else if (streq_(argv[2], "time"))    v = rd_time();
    else {
      puts_("BASELINE-PROBE-ERROR unknown counter '");
      puts_(argv[2]);
      puts_("' (want cycle, instret or time)\n");
      return 2;
    }
    puts_("BASELINE-PROBE ");
    puts_(argv[2]);
    puts_(" = ");
    put_ulong(v);
    puts_("\n");
    return 0;
  }
  {
    int is_run  = streq_(argv[1], "run");
    int is_warm = streq_(argv[1], "warm");
    if (argc < 3 || (!is_run && !is_warm)) {
      puts_("usage: speedtest1_baseline probe <cycle|instret>\n"
            "       speedtest1_baseline run  <speedtest1 args...>   -- cold\n"
            "       speedtest1_baseline warm <speedtest1 args...>   -- warmed first; THIS is the denominator\n");
      return 2;
    }
    warm = is_warm;
  }

  /* Everything after `run` is speedtest1's command line, joined with single spaces -- the same
     shape the capability host accepts, so the two arms are invoked identically. */
  for (i = 2; i < argc; i++) {
    ulong len = slen(argv[i]);
    if (at && at + 1 < sizeof args_buf) args_buf[at++] = ' ';
    if (at + len >= sizeof args_buf) { puts_("BASELINE-ERROR args too long\n"); return 2; }
    for (ulong k = 0; k < len; k++) args_buf[at + k] = argv[i][k];
    at += len;
  }
  args_buf[at] = 0;

  /* ONE MEASURED RUN PER PROCESS, because speedtest1 is not re-entrant: calling its main a second
     time in the same process dies, and sqlite3_shutdown between the two dies sooner. So `warm` runs
     a separate warm-up workload first rather than running the benchmark twice. Invoke the binary
     once each way and report the WARM number as the denominator; the difference between them is
     what the warm-up removed, and it is printed rather than assumed. */
  if (warm)
    speedtest1_baseline_warmup();

  /* TICKS brackets the run with `time`, the counter the monitor emulates and which therefore cannot
     wedge the machine. It is the denominator that survives whatever the cycle/instret probes say.
     The bracket is here rather than inside the amalgam because `time` is not the counter the
     capability arm reads, so it is scaffolding, not a matched measurement. */
  t0 = rd_time();
  n = speedtest1_baseline_run(args_buf, at, out_buf, sizeof out_buf, &cycles, &instrs, &marker);
  ticks = rd_time() - t0;
  wr(out_buf, n);
  puts_(warm ? "BASELINE-WARM CYCLES " : "BASELINE-COLD CYCLES ");
  put_ulong(cycles);
  puts_(" INSTRS ");
  put_ulong(instrs);
  puts_(" TICKS ");
  put_ulong(ticks);
  puts_(" MARKER ");
  put_ulong(marker);
  puts_("\n");
  /* A LINE THE EXISTING DRIVERS CAN SCORE. run_baked_rungs_fpga.py and run_sqlite_stages_fpga.py both
     require `RESULT <name> retval=<n>`; without it neither can record this arm, and the only driver
     that ever ran a plain Linux binary is deprecated. Emitting it here is a smaller and more honest
     change than teaching a driver a second scoring path, and it keeps both arms reporting the same
     way. The value is the domain-side marker, so a scored 0 means the run did not reach the end. */
  puts_("RESULT speedtest1_baseline retval=");
  put_ulong(marker);
  puts_("\n__CAPSTONE_SPEEDTEST1_BASELINE_RAN__\n");
  return 0;
}

/* No libc, so the entry point is ours. argc/argv come off the initial stack per the RISC-V Linux
 * ABI: sp points at argc, argv follows immediately. */
__asm__(
  ".section .text.entry\n"
  ".global _start\n"
  "_start:\n"
  "  ld a0, 0(sp)\n"
  "  addi a1, sp, 8\n"
  "  call baseline_main\n"
  "  li a7, 93\n"
  "  ecall\n");
