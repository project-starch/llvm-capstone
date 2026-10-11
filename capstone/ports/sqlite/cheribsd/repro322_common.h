/* repro322_common.h (CheriBSD port) -- drop-in replacement for the Capstone
 * freestanding scaffolding, so every case_*.c compiles UNCHANGED.
 *
 * Same contract as the Capstone header:
 *     #include "repro322_common.h"
 *     static int run_case(void) { ...drive SQLite to the freed-then-used path... }
 *     REPRO322_MAIN("<tag>")
 *
 * Differences from the Capstone version, and why:
 *   - out_text/out_uint write to stdout instead of a shared hostcall region.
 *     No __builtin_capstone_cap_delin: that exists only to work around the
 *     Capstone RTL's DELIN behaviour on non-linear capabilities.
 *   - REPRO322_MAIN expands to a real main() instead of domain_main(), and
 *     prints a machine-readable terminator the runner greps for.
 *   - repro_init() is IDENTICAL: same memsys5 arena, same 64-byte min alloc,
 *     so the allocator under test is the same one on both platforms.
 *
 * Arena size and SQLite build flags are supplied by build-corpus-cheri.sh and
 * mirror the Capstone build (-DSQLITE_HEAP_SIZE=262144, -O0).
 */
#ifndef REPRO322_COMMON_H
#define REPRO322_COMMON_H

#include "sqlite3.h"
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));

/* LadyBug reachability probe, no-op fallback.
 * The probe lives in sqlite3-probe.c; a case source that marks its own defect site
 * must still compile on the arms that build plain sqlite3-cheri.c, so the macros
 * are defined away here unless the probed amalgamation already defined them. */
/* LadyBug reachability probe.
 * -DLB_PROBE_BUILD is set only by the probe build, which links sqlite3-probe.c and
 * therefore supplies lb_site / lb_hit_slow. Every other build gets the no-ops, so a
 * case source that marks its own defect site still compiles on all three arms. */
#ifdef LB_PROBE_BUILD
extern int lb_site;
extern int lb_hit_slow(int id, const void *p, long n);
extern void lb_report(void);     /* hit count; _exit() below would skip atexit */
#define LB_HIT(id, p, n)  do{ if(lb_site==(id)) (void)lb_hit_slow((id),(const void*)(p),(long)(n)); }while(0)
#define LB_CHK(id, p, n)  (lb_site==(id) ? lb_hit_slow((id),(const void*)(p),(long)(n)) : 1)
#else
#define LB_HIT(id, p, n) ((void)0)
#define LB_CHK(id, p, n) (1)
#endif

static void out_text(const char *text) { fputs(text, stdout); fflush(stdout); }
static void out_uint(unsigned long v)  { printf("%lu", v); fflush(stdout); }


/* Diagnostic fault handler. Purely additive: it runs only AFTER a fault that would
 * have killed the process anyway, and it re-exits with 128+signal so the runner's
 * verdict mapping is unchanged. Its job is to print the CHERI si_code, which is the
 * only way to tell the fault MECHANISMS apart:
 *   PROT_CHERI_TAG    -> an untagged value was dereferenced. On this corpus that is
 *                        memsys5 clearing granule 0 of a freed block with its in-band
 *                        Mem5Link freelist ints (handoff mechanism M1).
 *   PROT_CHERI_BOUNDS -> a still-valid capability was walked past its bounds, e.g. a
 *                        COUNT read out of a freed block is now a freelist int (M2).
 * Without this the two are indistinguishable: both arrive as bare signal 34.
 */
static const char *repro_prot_code(int c) {
  switch (c) {
    case PROT_CHERI_BOUNDS: return "PROT_CHERI_BOUNDS/valid-cap-out-of-bounds";
    case PROT_CHERI_TAG:    return "PROT_CHERI_TAG/untagged-deref";
    case PROT_CHERI_SEALED: return "PROT_CHERI_SEALED";
    case PROT_CHERI_TYPE:   return "PROT_CHERI_TYPE";
    case PROT_CHERI_PERM:   return "PROT_CHERI_PERM";
    case PROT_CHERI_IMPRECISE: return "PROT_CHERI_IMPRECISE";
    case PROT_CHERI_UNALIGNED_BASE: return "PROT_CHERI_UNALIGNED_BASE";
    default: return "other";
  }
}
static void repro_fault_handler(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  char buf[200];
  int n = snprintf(buf, sizeof buf,
                   "\n<<FAULT>> signal=%d si_code=%d (%s) pc=%p capreg=%d\n",
                   sig, si->si_code,
                   sig == SIGPROT ? repro_prot_code(si->si_code) : "not-SIGPROT",
                   (void *)si->si_addr, si->si_trapno);
  if (n > 0) (void)write(2, buf, (size_t)n);
#ifdef LB_PROBE_BUILD
  lb_report();   /* before _exit: a faulting run would otherwise report no hit count,
                  * and the hit count is what says the fault was at the probed site */
#endif
  _exit(128 + sig);
}
static void repro_install_fault_handler(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = repro_fault_handler;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  (void)sigaction(SIGPROT, &sa, 0);
  (void)sigaction(SIGSEGV, &sa, 0);
  (void)sigaction(SIGBUS,  &sa, 0);
}

/* Forced-revocation shim for the system-allocator arm. See repro_init(). */
static sqlite3_mem_methods repro_base_mem;
static int repro_revoke_on = 0;

static void repro_free_revoking(void *p) {
  repro_base_mem.xFree(p);
  /* Declared in <stdlib.h> on CheriBSD. Sweeps the quarantine now, so a
  ** capability to the block just freed is revoked before the case's stale
  ** access rather than at some later threshold. */
  (void) malloc_revoke_quarantine_force_flush();
}

static void *repro_realloc_revoking(void *p, int n) {
  void *q = repro_base_mem.xRealloc(p, n);
  (void) malloc_revoke_quarantine_force_flush();
  return q;
}

static int run_case(void);

/* config memsys5 + init; return 0 on success. Same as the Capstone arm. */
static int repro_init(void) {
  /* A/B switch. With NOMEM5=1 in the environment the case runs on the SYSTEM
  ** allocator instead of memsys5. On purecap that changes the capability bounds a
  ** buffer carries -- its own allocation, instead of the whole 256 KiB arena -- and
  ** that is the difference which decides whether a heap overflow is catchable at all.
  ** memsys5 returns &mem5.zPool[i*szAtom], a pointer derived from the arena, so every
  ** sub-allocation inherits the ARENA bounds and an overflow inside it is in bounds. */
  if (getenv("NOMEM5")) {
    int r;
    /* REVOCATION MUST BE FORCED, or this arm measures the quarantine rather
    ** than the mechanism. CheriBSD's runtime_revocation_every_free_default is
    ** 0 on this guest, so a sweep runs only once a quarantine threshold is
    ** crossed. poscontrol.c has to churn 4096 allocations to cross it, and its
    ** own comment names what happens without that: "a default (non-zero)
    ** quarantine may not have swept yet and the dangling load would still be
    ** allowed -- that is use-after-REALLOCATION semantics, not
    ** use-after-free". None of the corpus cases churn, so without this every
    ** one of them reports PASS for a reason that has nothing to do with
    ** whether revocation can see the defect.
    **
    ** We force it at SQLite's own allocator boundary rather than by
    ** interposing libc free(): that is the layer the study is about, it needs
    ** no dlsym and no recursion guard, and it leaves libc's allocator exactly
    ** as the platform ships it. NOREVOKE=1 restores the stock quarantine, so
    ** the two can be compared. */
    if (!getenv("NOREVOKE")) {
      sqlite3_mem_methods m;
      if (sqlite3_config(SQLITE_CONFIG_GETMALLOC, &m) == SQLITE_OK && m.xFree) {
        repro_base_mem = m;
        m.xFree = repro_free_revoking;
        if (m.xRealloc) m.xRealloc = repro_realloc_revoking;
        if (sqlite3_config(SQLITE_CONFIG_MALLOC, &m) == SQLITE_OK)
          repro_revoke_on = 1;
      }
    }
    r = sqlite3_initialize();
    out_text("repro allocator=system\n");
    out_text(repro_revoke_on ? "repro revoke=on-every-free\n"
                             : "repro revoke=stock-quarantine\n");
    return r;
  }
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK) { out_text("repro ERROR config-heap rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK) { out_text("repro ERROR initialize rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  return 0;
}

/* The runner classifies on these two markers plus the wait status:
 *   "<tag> BEGIN"      -- the case started
 *   "<tag> RETURNED rc=N" -- run_case() returned, i.e. nothing trapped
 * A missing RETURNED line together with a signal status is a FAULT. */
#define REPRO322_MAIN(TAG)                                    \
  int main(void) {                                            \
    setvbuf(stdout, NULL, _IONBF, 0);                         \
    repro_install_fault_handler();                            \
    out_text(TAG " BEGIN\n");                                 \
    int rc = run_case();                                      \
    out_text(TAG " RETURNED rc="); out_uint((unsigned long)(rc < 0 ? -rc : rc)); \
    out_text("\n");                                           \
    return 0;                                                 \
  }

#define FAILRC(stage, rc) (out_text(stage), out_text(" rc="), out_uint((unsigned long)((rc)<0?-(rc):(rc))), out_text("\n"), (rc)?(rc):1)

#endif
