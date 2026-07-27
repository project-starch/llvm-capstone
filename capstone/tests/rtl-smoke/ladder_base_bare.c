/* BARE-METAL baseline controller: the plain-RISC-V half of the overhead
 * measurement, run as an S-mode payload with NO operating system underneath.
 *
 * WHY THIS EXISTS (issue I-2). The Linux-hosted baseline services timer
 * interrupts inside the measurement bracket. That is not a small effect and it is
 * not a guess: a control kernel of pure register arithmetic, compiled to the
 * identical five RISC-V instructions on both targets, runs at a metronomic
 * 6.003/6.001 cycles per iteration inside a capability domain and 7.29 under
 * Linux -- ~1.21x slow -- with the excess scaling proportionally with work
 * (3.9x for 4x the iterations) at ~14 cycles per extra instruction, which is
 * interrupt entry/exit plus cache disruption rather than the kernel itself.
 *
 * It inflates the DENOMINATOR of every overhead ratio, so it makes capability
 * overhead look SMALLER than it is -- the error has been running in our favour,
 * which is the honest reason to remove it rather than model it.
 *
 * Repeating each measurement 16x and keeping the least-disturbed pass fixes
 * kernels shorter than a timer tick (beebs_bs: 15/15 passes tied at minimum
 * instret, 45-cycle spread) but does NOTHING for long ones: for the calibration
 * kernel it recovered 7.290 against the old 7.287, where the true value is 6.003.
 * rv8_primes at 16.5M cycles is firmly in that regime and is still unmeasured.
 * The only way to fix it is to stop running the baseline under an OS.
 *
 * WHAT REPLACES LINUX. OpenSBI boots this directly in S-mode as the fw_payload's
 * payload, in place of the Linux Image. The rung kernels are the SAME
 * ladder_base_kern.c objects the Linux controller links -- they are plain
 * freestanding RISC-V with no OS dependency, so the measured code is unchanged and
 * only the harness around it differs. Output goes through the SBI legacy console
 * call instead of a write(2) syscall; the counters are the same unprivileged CSRs
 * (mcounteren lets S-mode read them, as the Linux userspace probe already
 * confirmed for cycle/time/instret).
 *
 * BONUS, and it is a large one: the payload replaces a ~15 MB Linux Image with a
 * ~15 KB program, so fw_payload drops from ~15 MB to ~2 MB. The JTAG reload of
 * that image dominates every board boot (~2 min), so this makes the board
 * substantially faster as a side effect of making it correct.
 *
 * We deliberately do NOT enable any interrupt here. No timer is programmed, no
 * trap vector is installed beyond the default, so nothing preempts a measurement.
 * If a trap ever does fire, it is a bug and it will hang rather than silently
 * perturb a number -- which is the failure mode we want.
 *
 * Output format is byte-identical to the Linux controller's, so
 * run_ladder_base_fpga.py parses both without changes.
 */
typedef unsigned long ulong;

/* ---- SBI console. Two extensions, chosen by probing, because the legacy one is
 * OPTIONAL and this board's firmware is a custom build.
 *
 * The first attempt used only the legacy console call (EID 0x01). It works under
 * QEMU's stock OpenSBI v1.3.1 -- the whole payload runs and prints there -- and
 * produced ABSOLUTELY NO OUTPUT on the board, whose CapliFive OpenSBI evidently
 * does not implement it. Silence is the worst possible failure mode: it is
 * indistinguishable from "the payload never started", and it cost a board session
 * to tell those apart. So probe, and prefer the modern DBCN extension. */
static long sbi_ecall(long eid, long fid, long a0_, long a1_, long a2_) {
  register long a0 __asm__("a0") = a0_;
  register long a1 __asm__("a1") = a1_;
  register long a2 __asm__("a2") = a2_;
  register long a6 __asm__("a6") = fid;
  register long a7 __asm__("a7") = eid;
  __asm__ volatile("ecall" : "+r"(a0), "+r"(a1)
                   : "r"(a2), "r"(a6), "r"(a7) : "memory");
  return a0;   /* SBI error code for non-legacy; legacy returns in a0 too */
}

#define SBI_EXT_BASE 0x10
#define SBI_EXT_DBCN 0x4442434EL      /* "DBCN" -- debug console */
#define SBI_EXT_SRST 0x53525354L      /* "SRST" -- system reset */

static int have_dbcn;

/* SBI returns {error in a0, VALUE in a1}. probe_extension's answer is the VALUE,
 * so a helper that only returns a0 always reports SBI_SUCCESS (0) and the caller
 * concludes "not available" -- which is exactly what happened on the first
 * attempt: DBCN was never selected and the board fallback was dead code. */
static long sbi_ecall_val(long eid, long fid, long a0_) {
  register long a0 __asm__("a0") = a0_;
  register long a1 __asm__("a1") = 0;
  register long a6 __asm__("a6") = fid;
  register long a7 __asm__("a7") = eid;
  __asm__ volatile("ecall" : "+r"(a0), "+r"(a1) : "r"(a6), "r"(a7) : "memory");
  return a0 == 0 ? a1 : 0;      /* value only if the call itself succeeded */
}

static void probe_console(void) {
  have_dbcn = (int)sbi_ecall_val(SBI_EXT_BASE, 3, SBI_EXT_DBCN);
}

static void put_c(char c) {
  if (have_dbcn) {
    /* DBCN FID 2 = console_write_byte */
    sbi_ecall(SBI_EXT_DBCN, 2, (unsigned char)c, 0, 0);
  } else {
    /* legacy console_putchar: EID 0x01, no FID */
    register long a0 __asm__("a0") = (unsigned char)c;
    register long a7 __asm__("a7") = 0x01;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
  }
}

static void sbi_shutdown(void) {
  sbi_ecall(SBI_EXT_SRST, 0, 0, 0, 0);          /* SRST: shutdown */
  register long a0 __asm__("a0") = 0;
  register long a7 __asm__("a7") = 0x08;        /* legacy shutdown */
  __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

/* ---- Direct ns16550a UART: the console that actually works on this board. -----
 *
 * The board's OpenSBI reports "Runtime SBI Version: 1.0". DBCN arrived in SBI 2.0,
 * so it CANNOT exist there, and the legacy console is evidently not built in
 * either -- two board sessions produced the OpenSBI banner followed by complete
 * silence from our payload. Rather than keep guessing at firmware features, drive
 * the UART directly.
 *
 * The parameters are not guessed; they come from the firmware's own device tree:
 *   /soc/uart@10000000  compatible=ns16550a  reg=0x10000000  reg-shift=2  io-width=4
 * PMP Domain0 Region05 grants S/U mode R,W,X over all memory, so S-mode may touch
 * these registers with no SBI involvement at all.
 *
 * QEMU's virt machine has the SAME chip at the SAME address but with reg-shift 0
 * and byte-wide registers, so shift/width are BUILD parameters and QEMU validation
 * must pass UART_SHIFT=0. Getting it wrong writes the wrong register and prints
 * nothing -- the identical silent failure this is meant to escape.
 */
#ifndef UART_BASE
#define UART_BASE 0x10000000UL
#endif
#ifndef UART_SHIFT
#define UART_SHIFT 2          /* board default; QEMU virt needs 0 */
#endif

#define UART_THR 0
#define UART_LSR 5
#define LSR_THRE 0x20         /* transmit holding register empty */

static inline unsigned uart_rd(int reg) {
#if UART_SHIFT == 2
  return *(volatile unsigned int *)(UART_BASE + ((unsigned long)reg << 2));
#else
  return *(volatile unsigned char *)(UART_BASE + (unsigned long)reg);
#endif
}
static inline void uart_wr(int reg, unsigned v) {
#if UART_SHIFT == 2
  *(volatile unsigned int *)(UART_BASE + ((unsigned long)reg << 2)) = v;
#else
  *(volatile unsigned char *)(UART_BASE + (unsigned long)reg) = (unsigned char)v;
#endif
}

static void uart_putc(char c) {
  /* Bounded spin: a wedged UART must not hang the measurement silently. */
  for (int i = 0; i < 100000; i++)
    if (uart_rd(UART_LSR) & LSR_THRE) break;
  uart_wr(UART_THR, (unsigned char)c);
}

static void puts_(const char *s) {
  while (*s) {
    if (*s == '\n') uart_putc('\r');
    uart_putc(*s);
    s++;
  }
}
/* MUST use the same console as puts_. It previously used put_c (the SBI path),
 * so on the board every label printed and every NUMBER came out blank:
 *   "BASE RESULT ctrsanity4 pass= retval= cycles= instret="
 * The run had actually succeeded; only the digits were lost. */
static void putu_(ulong v) {
  char buf[24];
  int i = 24;
  if (!v) { uart_putc('0'); return; }
  while (v) { buf[--i] = (char)('0' + (v % 10)); v /= 10; }
  while (i < 24) uart_putc(buf[i++]);
}

/* ---- Counters. Unprivileged CSRs, readable from S-mode via mcounteren. ------- */
static inline ulong rd_cycle(void) {
  ulong v; __asm__ volatile("csrr %0, cycle" : "=r"(v)); return v;
}
static inline ulong rd_instret(void) {
  ulong v; __asm__ volatile("csrr %0, instret" : "=r"(v)); return v;
}

/* ---- Rung table, GENERATED from ladder-rungs.spec. ---------------------------
 * Generated rather than hand-maintained: the Linux controller keeps its own table
 * separate from the build script's rung list, and adding a rung to one but not the
 * other builds cleanly and then reports "--" for every column. That cost a board
 * boot on 2026-07-27. Deriving both from the spec makes the drift impossible. */
#include "ladder_rungs_table.h"

#ifndef BASE_PASSES
#define BASE_PASSES 16
#endif

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

void bare_main(void) {
  probe_console();
  puts_("BARE BASELINE START\n");
  puts_("console=uart-mmio\n");
  for (int i = 0; i < NRUNGS; i++)
    for (int p = 1; p <= BASE_PASSES; p++)
      run_pass(&RUNGS[i], p);
  puts_("BARE BASELINE DONE\n");
  sbi_shutdown();
  for (;;) { }
}
