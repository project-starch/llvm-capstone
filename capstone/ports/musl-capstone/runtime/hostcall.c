/* __capstone_hostcall: the one place musl's syscall layer meets the domain
 * boundary, and the application's entry. arch-capstone64/syscall_arch.h routes
 * every __syscall0..6 here, and every call goes on to the delegated stub
 * (delegate.c, docs/plans/delegation-abi.md): the launcher's task runs it as
 * Linux. There is one application runtime, the delegated one (ABI v2); the
 * HostCall v0 emulation that used to live here (a file table, pipes, a working
 * directory and path operations kept in the domain) was removed on 2026-09-30.
 *
 * What stays in this file is what the stub does not do itself:
 *   - domain_main: the shared regions, the thread pointer, the transport and
 *     the launch block, then constructors, main and exit();
 *   - the regions the launcher shares after its own, for the program
 *     (__capstone_region);
 *   - the at-exit hook, and the record and report of what was not served;
 *   - .init_array and .fini_array.
 * The file keeps its name because the ports' build scripts look for it.
 */
#include "capstone/launch.h"
#include <errno.h>
#include <stdlib.h>
#include <sys/syscall.h>
/* syscall_arg_t and the __capstone_hostcall prototype both live in the arch
 * overlay. Including it here rather than re-declaring the type is what makes a
 * signature drift between the two a compile error instead of a silent ABI
 * mismatch at the one boundary that cannot be debugged from C. */
#include <syscall_arch.h>

extern int __capstone_application_prepare(const void *, size_t);
static void *hc_startup;

/* shared_region_annotated() enters the domain with func == 1 and the region
   capability as the first argument. */
#define CAPSTONE_DPI_REGION_SHARE 1

extern int capstone_main(void);
/* Installs the domain's single thread pointer. Everything in musl that reports
   an error needs it, so it runs before the program and not on demand. */
extern int __capstone_init_tls(void);

/* The launcher's regions, in the order it shares them: the entry block, the
   exchange region, the launch block. */
static void *hc_entry;
static void *hc_exchange;
static unsigned hc_shared_region_count;

/* Regions the launcher shares AFTER its three belong to the program: parked here
   as they arrive and handed over once each by __capstone_region. The first use is a heap the
   launcher transfers LINEAR (REV_TRANSFERRED), which the Sublet heap (sublet_heap.c) carves and
   revokes; without this they would be counted and dropped. A linear capability moves when it
   is loaded on hardware that enforces linearity, so the slot is read once and cleared. */
#define HC_PROGRAM_REGIONS 2
static void *hc_program_region[HC_PROGRAM_REGIONS];

void *__capstone_region(unsigned index) {
  if (index >= HC_PROGRAM_REGIONS)
    return 0;
  void *r = hc_program_region[index];
  hc_program_region[index] = 0;
  return r;
}

/* Optional: what a domain wants done when the program exits, run inside the
   exit_group call before the task ends (delegate.c). A domain overrides it with
   a strong definition; this weak one is what every other domain gets.

   DEFINED, not merely declared weak. An undefined weak symbol's address is not
   NULL in a domain (C-56): it is formed pc-relative against gp, the linker
   resolves the symbol to 0 at the link address, and the image runs at another
   base without relocation, so the old `if (__capstone_at_exit)` was always true
   and exit() called the image base. */
__attribute__((__weak__)) int __capstone_at_exit(int status) { return status; }

/* RECORDED, NOT JUST REFUSED. A libc is never finished in the sense that every
   syscall is served; it is finished in the sense that what is not served is
   visible. ENOSYS alone is not visible: musl turns most of them into a
   plausible-looking failure and the caller carries on, so a missing call
   surfaces later as wrong behaviour somewhere unrelated. The stub notes every
   call it refuses here, and the list is reported at exit. */
#define HC_UNSERVED_MAX 16
static long hc_unserved[HC_UNSERVED_MAX];
static unsigned long hc_unserved_n;

/* Syscalls answered with a success that has no service behind it, recorded like
   the unserved ones and printed at exit beside them, so that "served" never
   quietly comes to mean "pretended". */
#define HC_NOOP_MAX 16
static long hc_noop[HC_NOOP_MAX];
static unsigned long hc_noop_n;

void __capstone_hc_note_unserved(long n) {
  if (hc_unserved_n < HC_UNSERVED_MAX)
    hc_unserved[hc_unserved_n] = n;
  hc_unserved_n++;
}
void __capstone_hc_note_noop(long n) {
  if (hc_noop_n < HC_NOOP_MAX)
    hc_noop[hc_noop_n] = n;
  hc_noop_n++;
}

/* The delegated stub (delegate.c): every call is Linux's, run by the task. */
long __capstone_delegate_call(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f);
void __capstone_delegate_write2(const char *buf, unsigned long n);

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d, syscall_arg_t e,
                         syscall_arg_t f) {
  return __capstone_delegate_call(n, a, b, c, d, e, f);
}

/* Said out loud at exit, because recording is not reporting.
 *
 * A refused syscall comes back as -ENOSYS, musl turns that into an ordinary
 * failed call, and a program that does not check carries on with a wrong
 * answer. Every domain therefore says on its way out what it refused.
 *
 * Straight to the task's fd 2 through the stub, not printf and not write(2):
 * this runs inside exit_group, after the program is finished and musl's stdio
 * has been torn down, and write(1) would go through the program's descriptors:
 * a program that closed fd 1 (a full tshark run does) lost the report, and a
 * missing line reads as "nothing unserved" (ISSUES I-11).
 *
 * Numbers, not names: a table of three hundred names is not worth the bytes in
 * every domain image when the readers already translate. The delegated
 * libc-test runner prints them by name, and check-domain-support.py says which
 * CALL needs each one, which is the question a person actually has.
 */
static void hc_report_list(const char *head, const long *list,
                           unsigned long total, unsigned long max) {
  if (total == 0)
    return;
  char buf[256];
  unsigned long p = 0;
  for (unsigned long i = 0; head[i]; i++)
    buf[p++] = head[i];
  unsigned long shown = total < max ? total : max;
  for (unsigned long i = 0; i < shown && p + 32 < sizeof buf; i++) {
    unsigned long seen = 0, times = 0;
    for (unsigned long j = 0; j < shown; j++) {
      if (list[j] != list[i])
        continue;
      if (j < i)
        seen = 1;
      times++;
    }
    if (seen)   /* one entry per distinct number, with how often it was asked */
      continue;
    buf[p++] = ' ';
    long v = list[i];
    char d[20];
    unsigned long k = 0;
    do { d[k++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (k)
      buf[p++] = d[--k];
    if (times > 1) {
      buf[p++] = 'x';
      k = 0;
      do { d[k++] = (char)('0' + times % 10); times /= 10; } while (times);
      while (k)
        buf[p++] = d[--k];
    }
  }
  if (total > shown && p + 8 < sizeof buf) {
    static const char more[] = " ...";
    for (unsigned long i = 0; i < sizeof more - 1; i++)
      buf[p++] = more[i];
  }
  buf[p++] = '\n';
  __capstone_delegate_write2(buf, p);
}

/* The report is a diagnostic and lands on the application's stderr, which
   otherwise carries application bytes only; it is printed under the same
   switch as the launcher's counters. */
void __capstone_hc_report_unserved(void) {
  if (!getenv("CAPSTONE_DELEGATE_STATS"))
    return;
  hc_report_list("capstone-domain: UNSERVED syscalls:", hc_unserved, hc_unserved_n,
                 HC_UNSERVED_MAX);
  hc_report_list("capstone-domain: NO-OP syscalls:", hc_noop, hc_noop_n, HC_NOOP_MAX);
}

/* Read by a program's at-exit hook, never during the run. Returns the total
   seen, which may exceed what was kept. */
unsigned long __capstone_unserved_count(void) { return hc_unserved_n; }
long __capstone_unserved_at(unsigned long i) {
  return i < HC_UNSERVED_MAX && i < hc_unserved_n ? hc_unserved[i] : -1;
}

/* Constructors and destructors: .init_array and .fini_array (ISSUES C-64).
 *
 * Nothing else in a domain runs them. start-musl.S runs only .capstone_cap_init,
 * and musl's __libc_start_main, which would, is not used. musl's exit() walks
 * .fini_array through uintptr_t,
 *
 *     uintptr_t a = (uintptr_t)&__fini_array_end;
 *     for (; a > (uintptr_t)&__fini_array_start; a -= sizeof(void(*)()))
 *         (*(void (**)())(a - sizeof(void(*)())))();
 *
 * and loads each slot through an integer address: cause 24 on the first one.
 * Both stayed invisible while no domain had either array. The tshark port was
 * the first, with GLib's and libgpg-error's constructors and libxml2's
 * destructor.
 *
 * THE SLOTS ARE NOT CAPABILITIES. A static domain image carries no relocations,
 * and the capability initialisers do not cover these arrays, so each 16-byte slot
 * holds the function's LINK address as a plain integer in its low 8 bytes, and
 * the domain runs at another base. The callable capability is derived from the
 * code capability of a function in this file, the anchor, moved by the distance
 * between the two link addresses. The anchor's own link address is written into
 * .rodata by the assembler (`.quad`), which the static link resolves exactly as
 * it resolves the slots. A slot that does hold a tagged capability is called as
 * it is.
 *
 * The array markers are defined by my_first_domain/link.ld, the script every musl
 * domain links with, so their addresses are real ones (an undefined weak symbol's
 * would not be: C-56). __libc_exit_fini replaces musl's weak alias of the same
 * name (src/exit/exit.c); musl's version also calls _fini(), which in a domain is
 * its empty default. */
extern const unsigned char __init_array_start[], __init_array_end[];
extern const unsigned char __fini_array_start[], __fini_array_end[];

void __capstone_init_fini_anchor(void);
void __capstone_init_fini_anchor(void) {}

extern const unsigned long __capstone_init_fini_anchor_link;
__asm__(".section .rodata\n"
        ".p2align 3\n"
        ".globl __capstone_init_fini_anchor_link\n"
        "__capstone_init_fini_anchor_link:\n"
        ".quad __capstone_init_fini_anchor\n"
        ".previous\n");

typedef void (*hc_array_fn)(void);

static void hc_call_array_slot(const unsigned char *slot) {
  hc_array_fn f = *(const hc_array_fn *)slot;
  if (!__builtin_capstone_cap_get_tag(f)) {
    unsigned long link = *(const unsigned long *)slot;
    f = (hc_array_fn)((const char *)__capstone_init_fini_anchor +
                      (long)(link - __capstone_init_fini_anchor_link));
  }
  f();
}

/* The environment the program starts with. In C, environ exists before any
 * constructor runs, and a constructor may read it: GLib's reads G_DEBUG and
 * G_MESSAGES_PREFIXED. The application entry (runtime/domain/application.c)
 * defines this with the environment of the launch block; the weak default is
 * an empty environment. DEFINED weak, not declared (C-56). */
extern char **__environ;
__attribute__((__weak__)) char **__capstone_domain_environ(void) {
  static char *none[] = { 0 };
  return none;
}

static void hc_run_init_array(void) {
  for (const unsigned char *p = __init_array_start; p < __init_array_end;
       p += sizeof(hc_array_fn))
    hc_call_array_slot(p);
}

void __libc_exit_fini(void) {
  for (const unsigned char *p = __fini_array_end; p > __fini_array_start;
       p -= sizeof(hc_array_fn))
    hc_call_array_slot(p - sizeof(hc_array_fn));
}

/* Domain entry. The region shares carry the entry block, the exchange region,
   the launch block and then the program's regions; the entry after them runs
   the program. */
void domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (hc_shared_region_count == 0)
      hc_entry = res;
    else if (hc_shared_region_count == 1)
      hc_exchange = res;
    else if (hc_shared_region_count == 2)
      hc_startup = res;
    else if (hc_shared_region_count - 3 < HC_PROGRAM_REGIONS)
      hc_program_region[hc_shared_region_count - 3] = res;
    ++hc_shared_region_count;
    return;
  }

  /* Before the program, because errno has to exist the first time a syscall
     fails, and that can be the program's first line. A failure here is worth
     more than the program's own status: it means every later error report
     would have faulted instead. */
  if (__capstone_init_tls() != 0) {
    if (res)
      *res = (unsigned)-1;
    return;
  }

  /* Region 0 is the entry block, region 1 the exchange region. Before the
     launch block is applied, because its chdir() is already a delegated
     call. The first request tells the launcher where this image runs. */
  {
    extern void __capstone_delegate_regions(void *, void *);
    extern long __capstone_delegate_hello(unsigned long, unsigned long, unsigned long);
    extern int __capstone_delegate_ready(void);
    void (*anchor)(unsigned *, unsigned) = domain_main;
    __capstone_delegate_regions(hc_entry, hc_exchange);
    if (!__capstone_delegate_ready()) {
      if (res)
        *res = (unsigned)-1;
      return;
    }
    __capstone_delegate_hello((unsigned long)__builtin_capstone_cap_get_cursor(anchor),
                              (unsigned long)__builtin_capstone_cap_get_base(anchor),
                              (unsigned long)__builtin_capstone_cap_get_end(anchor));
  }
  size_t startup_bytes = hc_startup ?
      __builtin_capstone_cap_get_end(hc_startup) -
      __builtin_capstone_cap_get_cursor(hc_startup) : 0;
  int startup_error = __capstone_application_prepare(hc_startup, startup_bytes);
  if (startup_error)
    _Exit(125);
  {
    extern const struct capstone_launch_task *__capstone_launch_task(void);
    extern void __capstone_set_tid(int);
    const struct capstone_launch_task *task = __capstone_launch_task();
    if (task && task->pid) __capstone_set_tid((int)task->pid);
  }

  /* The environment, then the constructors, then main, as in C. A program that
     returns ends as returning from main ends in C: through exit(), so its
     atexit handlers run and stdio is flushed; exit_group then ends the task
     and nothing resumes here. */
  __environ = __capstone_domain_environ();
  hc_run_init_array();
  exit(capstone_main());
}
