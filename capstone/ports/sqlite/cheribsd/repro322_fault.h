/* repro322_fault.h (CheriBSD port) -- the diagnostic fault handler, shared by every
 * binary a run produces: the case images through repro322_common.h, and poscontrol
 * directly.
 *
 * It is its own header for one reason. The gate (poscontrol) is what proves the
 * handler's dladdr resolver works in the boot that is about to be read, and a gate
 * built from different source than the thing it gates proves nothing. Keeping the
 * handler in repro322_common.h made that impossible: that header pulls in sqlite3.h
 * and defines the memsys5 arena, which poscontrol neither has nor needs. Nothing
 * here depends on SQLite.
 */
#ifndef REPRO322_FAULT_H
#define REPRO322_FAULT_H

#include <stdio.h>
#include <signal.h>
#include <dlfcn.h>
#include <unistd.h>
#include <string.h>

#ifdef LB_PROBE_BUILD
extern void lb_report(void);     /* hit count; _exit() below would skip atexit */
#endif

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
  char buf[512];
  int n = snprintf(buf, sizeof buf,
                   "\n<<FAULT>> signal=%d si_code=%d (%s) pc=%p capreg=%d\n",
                   sig, si->si_code,
                   sig == SIGPROT ? repro_prot_code(si->si_code) : "not-SIGPROT",
                   (void *)si->si_addr, si->si_trapno);
  if (n > 0) (void)write(2, buf, (size_t)n);
  /* WHERE the fault is, resolved INSIDE the faulting process, which is the only
  ** place it can be resolved: the guest runs with kern.elf64.aslr.enable=1, so a pc
  ** in a shared object cannot be mapped to a symbol afterwards from the pc alone --
  ** five cells of the 2026-10-29 non-nested run are NO-READING:unattributed for
  ** exactly that. dladdr gives the object, its load base and the nearest symbol, so
  ** the attribution needs no load-base table and no second run.
  **
  ** dladdr is not async-signal-safe: it takes rtld's lock. We call it anyway, and
  ** the trade is deliberate. The faults this corpus produces are in the program's
  ** own code or in a libc string primitive, never inside rtld, so the lock is free;
  ** and if it ever is not, the handler blocks and the runner's timeout reports
  ** TIMEOUT -- a visible non-reading rather than a wrong one. _exit follows
  ** immediately either way. The poscontrol gate faults before every run, so this
  ** line is exercised once per run whatever the cases do: a run whose gate prints
  ** no FAULT-WHERE has a broken resolver, not an unresolvable fault. */
  {
    Dl_info at, self;
    int k = dladdr((const void *)si->si_addr, &at);
    int m = dladdr((const void *)(const char *)repro_fault_handler, &self);
    n = snprintf(buf, sizeof buf,
                 "<<FAULT-WHERE>> obj=%s base=%p sym=%s symaddr=%p image=%s imagebase=%p\n",
                 k && at.dli_fname ? at.dli_fname : "?",
                 k ? at.dli_fbase : (void *)0,
                 k && at.dli_sname ? at.dli_sname : "?",
                 k ? at.dli_saddr : (void *)0,
                 m && self.dli_fname ? self.dli_fname : "?",
                 m ? self.dli_fbase : (void *)0);
    if (n > 0) (void)write(2, buf, (size_t)n);
  }
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

#endif /* REPRO322_FAULT_H */
