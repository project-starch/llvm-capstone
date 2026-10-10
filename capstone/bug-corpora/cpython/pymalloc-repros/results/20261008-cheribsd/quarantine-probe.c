/* Is a freed object in CheriBSD's revocation quarantine, and is it reused before a
 * sweep clears it?
 *
 * WHY IT LIVES HERE AND NOT IN A RESULT BUNDLE. It answers the one question a
 * CheriBSD MISS always raises -- did the mechanism fail to see this, or had the
 * asynchronous sweep simply not run yet -- and that question belongs to every
 * corpus, not to the one whose bundle it first appeared in. It is
 * program-independent: put it in front of any case.
 *
 * HOW TO READ IT, in the order the counters have to be read. `sweeps` first: with
 * sweeps=0 nothing freed during the case was ever cleared, so a stale read into a
 * freed block succeeds WITHOUT the block being handed out again -- and then a zero
 * in reused_while_quarantined says nothing about whether the object entered the
 * quarantine. That is why the counter exists, and why a reading without a sweep
 * control beside it is not evidence: a counter that is always zero looks the same.
 *
 * So the probe separates the two silences only together with the program's own
 * account of where its memory went. memcached's eight (2026-10-08): 4 to 6 libc
 * frees in the whole process, all quarantined, while the program's own counters
 * show the reuse on its cache and slab freelists -- with four frees in total the
 * item cannot have passed through free(), so NEVER-FREED. Perl's extra case
 * (2026-10-06): the SV head goes on PL_sv_root and is never returned, which is
 * the code path, and 2,983 frees with zero reissues is corroboration, not the
 * proof on its own. mruby's four (2026-10-08): mruby frees straight through libc,
 * 1,238 to 1,994 frees all quarantined, and sweeps=0 -- the block did enter the
 * quarantine and no sweep ever cleared it, so the stale read went through
 * quarantined memory. QUARANTINED-UNSWEPT: the mechanism had its chance and the
 * asynchronous window lost the race.
 *
 * Reading the shadow bitmap is what makes this attributable where forcing a
 * sweep is not: `_RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1` makes Perl's eleven
 * all fault, including one that is an in-bounds read of a LIVE object which no
 * bounds and no revocation may legitimately catch -- so that configuration
 * cannot tell a real catch from an artefact, and its verdicts were withdrawn.
 * The bitmap can: the kernel exposes one bit per 16-byte granule, set while that
 * granule is quarantined. Reading a bit is O(1), so this can run on every
 * allocator call without forcing a sweep or changing anything else about the run.
 *
 * TWO WAYS IN, because a purecap program is not always dynamically linked.
 *   default           a shared object defining malloc/free/realloc/calloc.
 *                     LD_PRELOAD it. Needs a dynamic binary.
 *   -DQPROBE_WRAP     an object defining __wrap_* instead, for a STATIC binary:
 *                     link it in with -Wl,--wrap=malloc,--wrap=free,
 *                     --wrap=realloc,--wrap=calloc. mruby's CheriBSD arm must be
 *                     static (dynamically linked it dies in the loader with
 *                     "Traditional TLS not supported"), which is why this exists.
 * Both report the same counters, to stderr, at exit:
 *   frees             how many free() calls happened, plus the implicit frees a
 *                     moving realloc performs
 *   quarantined       of those, how many had their shadow bit SET right after
 *   reused_unswept    how many allocation results came back with the bit STILL
 *                     set -- memory handed out again while quarantined, which is
 *                     exactly the window an async sweep leaves open
 *   sweeps            how far the DEQUEUE epoch advanced while the case ran, i.e.
 *                     how many revocation sweeps completed. Read it before the
 *                     other three: with sweeps=0 nothing freed during the case
 *                     was ever cleared, so a stale read into a freed block
 *                     succeeds without any reissue and reused_unswept=0 proves
 *                     nothing about whether the object entered the quarantine
 * A stale pointer into memory that never reaches free() shows up as neither, and
 * that is the structural answer rather than a missing measurement.               */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <cheri/cheric.h>
#include <cheri/revoke.h>

static unsigned char *shadow;
static int shadow_err;
static unsigned long n_alloc, n_free, n_quarantined, n_reused_unswept;
static int ready;
/* The dequeue epoch is what gates removal FROM the quarantine, so its advance
 * counts completed sweeps. It is the difference between the two silences this
 * probe has to tell apart. If no sweep ran, every block freed during the case
 * was still quarantined when the case ended, and a stale read into one of them
 * succeeds WITHOUT any reissue -- so a zero in reused_while_quarantined does
 * not mean the object stayed out of the quarantine. If a sweep did run, a stale
 * capability into a swept block is untagged and the read would have faulted. */
static struct cheri_revoke_info *info;
static unsigned long long epoch_first;

#ifdef QPROBE_WRAP
/* --wrap gives the real ones a name, so there is nothing to look up. */
extern void *__real_malloc(size_t);
extern void *__real_calloc(size_t, size_t);
extern void *__real_realloc(void *, size_t);
extern void __real_free(void *);
#define REAL_MALLOC __real_malloc
#define REAL_CALLOC __real_calloc
#define REAL_REALLOC __real_realloc
#define REAL_FREE __real_free
#define PROBE(name) __wrap_##name
#else
#include <dlfcn.h>
static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void (*real_free)(void *);
#define REAL_MALLOC real_malloc
#define REAL_CALLOC real_calloc
#define REAL_REALLOC real_realloc
#define REAL_FREE real_free
#define PROBE(name) name
#endif

static void init(void) {
  if (ready) return;
  ready = 1;
#ifndef QPROBE_WRAP
  real_malloc = dlsym(RTLD_NEXT, "malloc");
  real_calloc = dlsym(RTLD_NEXT, "calloc");
  real_realloc = dlsym(RTLD_NEXT, "realloc");
  real_free = dlsym(RTLD_NEXT, "free");
#endif
  void *s = NULL;
  if (cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_NOVMEM_ENTIRE, NULL, &s) != 0) shadow_err = errno;
  else shadow = s;
  void *i = NULL;
  if (cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_INFO_STRUCT, NULL, &i) == 0) {
    info = i;
    epoch_first = info->epochs.dequeue;
  }
}

/* The fine-grained map is one bit per capability granule, which revoke.h gives as
 * VM_CHERI_REVOKE_GSZ_MEM_NOMAP (16 bytes here), indexed from the returned pointer. */
static int bit_set(const void *p) {
  if (!shadow) return -1;
  unsigned long g = (unsigned long)cheri_getaddress(p) / VM_CHERI_REVOKE_GSZ_MEM_NOMAP;
  return (shadow[g / 8] >> (g % 8)) & 1;
}
int quarantine_bit(const void *p) { init(); return bit_set(p); }

/* Every allocation result is asked the same question: did this memory come back
 * while its quarantine bit was still set? */
static void *issued(void *p) {
  if (p) { n_alloc++; if (bit_set(p) == 1) n_reused_unswept++; }
  return p;
}

/* And every release: did it enter the quarantine at all? */
static void released(void *p) {
  n_free++;
  if (bit_set(p) == 1) n_quarantined++;
}

void *PROBE(malloc)(size_t n) { init(); return issued(REAL_MALLOC(n)); }

void *PROBE(calloc)(size_t n, size_t m) { init(); return issued(REAL_CALLOC(n, m)); }

void PROBE(free)(void *p) {
  init();
  if (!p) { REAL_FREE(p); return; }
  REAL_FREE(p);
  released(p);
}

/* realloc is where an interpreter does most of its freeing: a grow that moves
 * releases the old block, and counting only free() would miss it. The old
 * pointer is asked its question only when the block actually moved. */
void *PROBE(realloc)(void *old, size_t n) {
  init();
  void *p = REAL_REALLOC(old, n);
  if (old && p != old) released(old);
  return issued(p);
}

__attribute__((destructor)) static void report(void) {
  char line[256];
  int k = snprintf(line, sizeof line,
    "QUARANTINE shadow=%s allocs=%lu frees=%lu quarantined_after_free=%lu "
    "reused_while_quarantined=%lu sweeps=%llu\n",
    shadow ? "mapped" : (shadow_err ? "ERRNO" : "null"),
    n_alloc, n_free, n_quarantined, n_reused_unswept,
    info ? (unsigned long long)(info->epochs.dequeue - epoch_first) : 0ULL);
  if (k > 0) write(2, line, (size_t)k);
}
