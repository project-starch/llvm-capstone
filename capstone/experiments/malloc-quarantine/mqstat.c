/* LD_PRELOAD probe for unmodified programs: at exit print the revocation
 * epochs, jemalloc's final ledger and the kernel's sweep counters.
 * Peak memory comes from the kernel (time -l).
 *   MQ-EXIT-STATS revocation= enqueue= dequeue= allocated= active= resident= mapped=
 *   MQ-SWEEP-STATS calls= taken= passes= pages_scan_ro= ... page_scan_cycles= fault_cycles=
 *
 * Sweep counters: the kernel accumulates them per process until a call with
 * CHERI_REVOKE_TAKE_STATS copies them out and zeroes them.  MRS's synchronous
 * path passes TAKE_STATS (with a NULL info pointer) on every call, so reading
 * them only at exit sees zero there.  libc reaches cheri_revoke through its
 * PLT, so this file interposes it: every TAKE_STATS call is given an info
 * block and its counters are added up before they are lost.  The exit-time
 * read (without TAKE_STATS) adds whatever the asynchronous path left behind.
 * calls= counts interposed calls; it must be non-zero whenever passes is,
 * otherwise the interposition did not happen and the totals are incomplete.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cheri/revoke.h>
#include <malloc_np.h>

int __sys_cheri_revoke(int, cheri_revoke_epoch_t, struct cheri_revoke_syscall_info *);

#define FIELDS(X) X(pages_scan_ro) X(pages_scan_rw) X(pages_faulted_ro) X(pages_faulted_rw) \
  X(fault_visits) X(pages_skip_fast) X(pages_skip_nofill) X(pages_skip) X(caps_found) \
  X(caps_found_revoked) X(caps_cleared) X(lines_scan) X(pages_mark_clean) \
  X(page_scan_cycles) X(fault_cycles)
#define DECL(f) unsigned long long f;
static struct { FIELDS(DECL) } sum;
static unsigned long long calls, taken;

static void add(const struct cheri_revoke_stats *s) {
#define ADD(f) sum.f += s->f;
  FIELDS(ADD)
}

int cheri_revoke(int flags, cheri_revoke_epoch_t start, struct cheri_revoke_syscall_info *crsi) {
  calls++;
  if (!(flags & CHERI_REVOKE_TAKE_STATS))
    return __sys_cheri_revoke(flags, start, crsi);
  struct cheri_revoke_syscall_info mine;
  struct cheri_revoke_syscall_info *p = crsi ? crsi : &mine;
  int r = __sys_cheri_revoke(flags, start, p);
  if (r == 0) { add(&p->stats); taken++; }
  return r;
}

static size_t stat(const char *name) {
  size_t v = 0, n = sizeof v;
  return mallctl(name, &v, &n, NULL, 0) ? (size_t)-1 : v;
}

__attribute__((destructor)) static void mq_exit(void) {
  const struct cheri_revoke_info *info = NULL;
  void *p = NULL;
  if (!cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_INFO_STRUCT, NULL, &p)) info = p;
  uint64_t e = 1; size_t len = sizeof e;
  mallctl("epoch", &e, &len, &e, sizeof e);
  char line[1024];
  int n = snprintf(line, sizeof line,
    "MQ-EXIT-STATS revocation=%d enqueue=%llu dequeue=%llu allocated=%zu active=%zu "
    "resident=%zu mapped=%zu\n", malloc_revoke_enabled(),
    info ? (unsigned long long)info->epochs.enqueue : 0ull,
    info ? (unsigned long long)info->epochs.dequeue : 0ull,
    stat("stats.allocated"), stat("stats.active"), stat("stats.resident"), stat("stats.mapped"));
  if (n > 0) write(2, line, (size_t)n);
  /* With no completed pass there is nothing to read.  A start epoch of
   * dequeue-2 has already cleared, so this takes the kernel's fast-out path:
   * it starts no pass and, without TAKE_STATS, resets nothing. */
  if (!info || info->epochs.dequeue < 2) return;
  struct cheri_revoke_syscall_info crsi;
  memset(&crsi, 0, sizeof crsi);
  if (__sys_cheri_revoke(0, info->epochs.dequeue - 2, &crsi)) {
    write(2, "MQ-SWEEP-STATS error\n", 21);
    return;
  }
  add(&crsi.stats);
  n = snprintf(line, sizeof line, "MQ-SWEEP-STATS calls=%llu taken=%llu passes=%llu",
               calls, taken, (unsigned long long)info->epochs.dequeue / 2);
#define PR(f) if (n > 0 && (size_t)n < sizeof line) \
    n += snprintf(line + n, sizeof line - n, " " #f "=%llu", sum.f);
  FIELDS(PR)
  if (n > 0 && (size_t)n < sizeof line - 1) { line[n++] = '\n'; write(2, line, (size_t)n); }
}
