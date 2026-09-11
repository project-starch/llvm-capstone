#include "capstone_sqlite_vfs.h"

static struct capstone_sqlite_vfs_bundle capstone_sqlite_bundle;

#ifdef CAPSTONE_SQLITE_MCYCLE_CLOCK
/* A REAL CLOCK, for benchmarks that time themselves.
 *
 * The skeleton VFS registers xCurrentTime and xCurrentTimeInt64 as stubs that return 0, which is
 * right for SQLite (it needs no clock with the datetime functions omitted) and useless for
 * speedtest1, which reports every phase as a difference of two timestamps -- all of them zero.
 *
 * WHY HERE AND NOT IN THE SKELETON: capstone_sqlite_vfs.c is shared with the VFS-skeleton test
 * domain, and this file is the only other consumer. Overriding after bootstrap changes exactly one
 * build and leaves that test alone.
 *
 * WHY BEHIND A DEFINE: without it, every existing SQLite domain image would change, and the board
 * results on record are tied to those images. Opt-in keeps them reproducible.
 *
 * mcycle, not rdcycle: the board gates the unprivileged counter for domains (counteren.CY off), so
 * the M-mode CSR is the one a domain can read. Same reason ladder_perf_domain.h reads 0xB00.
 *
 * UNITS: SQLite's contract is milliseconds since the Julian epoch, and speedtest1 only ever
 * subtracts one reading from another, so an arbitrary origin is fine but the SCALE must be right or
 * every reported duration is wrong by that factor. The board core runs at 25 MHz. Note the device
 * tree also carries timebase-frequency = 12500000; that is the mtime tick, half the core clock, and
 * it is NOT what mcycle counts -- a reader checking this constant against the DTS will find 12.5
 * and should not "correct" it. */
#ifndef CAPSTONE_SQLITE_CLOCK_HZ
#define CAPSTONE_SQLITE_CLOCK_HZ 25000000UL
#endif

static unsigned long capstone_sqlite_rd_mcycle(void) {
  unsigned long v;
#ifdef CAPSTONE_SPEEDTEST1_BASELINE
  /* THE BASELINE RUNS IN LINUX USERSPACE AND CANNOT READ 0xB00. Reading it there is an ILLEGAL
   * INSTRUCTION, and with no libc there is no handler: the process dies and the trap surfaces from
   * M-mode, which reads like a monitor bug rather than like the two-line mistake it is. Measured
   * 2026-09-10 -- the baseline's first QEMU run died exactly here, after both counter probes had
   * already succeeded. The U-mode mirror `cycle` (0xC00) counts the same underlying cycles. */
  __asm__ volatile("csrr %0, cycle" : "=r"(v));
#else
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
#endif
  return v;
}

/* Captured at registration so the reported times start near zero rather than at whatever the core
 * had already counted; a benchmark's first phase should not carry the boot. */
static unsigned long capstone_sqlite_clock_base;

static int capstone_sqlite_mcycle_time_int64(sqlite3_vfs *vfs, sqlite3_int64 *out) {
  (void)vfs;
  if (out)
    *out = (sqlite3_int64)((capstone_sqlite_rd_mcycle() - capstone_sqlite_clock_base)
                           / (CAPSTONE_SQLITE_CLOCK_HZ / 1000UL));
  return SQLITE_OK;
}

/* The v1 method too. speedtest1 takes the Int64 path -- speedtest1.c:309 requires
 * `iVersion>=2 && xCurrentTimeInt64!=0`, and capstone_sqlite_vfs.c:266 sets iVersion to 3 -- but
 * SQLite internals may reach for either, and leaving one live stub beside one real clock is the
 * kind of half-change that later reads as a defect. Under SQLITE_OMIT_FLOATING_POINT `double` is an
 * integer type here, which is why this signature works at all.
 *
 * The v1 fallback is also where speedtest1 has a genuine type bug under these defines: its local is
 * a real `double` (sqlite3.h #undefs the remap at :11352) while the method writes an int64, which
 * gcc reports when building the native baseline. It is DEAD CODE on both our paths because
 * iVersion is 3, and it is written down here so the warning is not re-derived as a finding. */
static int capstone_sqlite_mcycle_time(sqlite3_vfs *vfs, double *out) {
  sqlite3_int64 ms = 0;
  capstone_sqlite_mcycle_time_int64(vfs, &ms);
  if (out)
    *out = (double)ms;
  return SQLITE_OK;
}
#endif /* CAPSTONE_SQLITE_MCYCLE_CLOCK */

int sqlite3_os_init(void) {
  capstone_sqlite_vfs_bootstrap(&capstone_sqlite_bundle);
#ifdef CAPSTONE_SQLITE_MCYCLE_CLOCK
  capstone_sqlite_clock_base = capstone_sqlite_rd_mcycle();
  capstone_sqlite_bundle.vfs.xCurrentTimeInt64 = capstone_sqlite_mcycle_time_int64;
  capstone_sqlite_bundle.vfs.xCurrentTime = capstone_sqlite_mcycle_time;
#endif
  return sqlite3_vfs_register(&capstone_sqlite_bundle.vfs, 1);
}

int sqlite3_os_end(void) { return SQLITE_OK; }
