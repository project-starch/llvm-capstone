/* PoisonCap backend for the port's backing API. Every system request wmem
 * makes -- a block, a jumbo object, a descriptor -- is one mapped region that
 * keeps SW_VMEM and POISON authority for the manager; published objects lose
 * both, so a sweep revokes them and nothing else. Mode 0 bounds objects
 * exactly and invalidates nothing. Mode 1 invalidates a retained block at every
 * reset, a region at every release, and a recycler chunk at every individual
 * free -- the hook Sublet has no use for. Trusted, serial, with a synchronous
 * sweep before storage can be reused. */
#include "poisoncap.h"
#include "port.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef CHERI_PERM_POISON
#error "WM_POISONCAP requires the PoisonCap SDK"
#endif

struct entry {
  void *region;
  size_t size;
  unsigned live;
};
static struct entry entries[WM_REGIONS];
static unsigned count, live, peak, mode, initialized;
static uint64_t created;
static size_t sweeps, poisoned, epochs, released_chunks, releases, unrepresentable;

static _Noreturn void refuse(const char *why) {
  fprintf(stderr, "WM_POISONCAP refused: %s\n", why);
  exit(1);
}

void wm_poisoncap_init(unsigned selected) {
  if (initialized || selected > 1 || !feature_present("cheri_caprevoke_poison"))
    refuse("initialization or platform");
  initialized = 1;
  mode = selected;
}

/* The port's backing entry. The payload is unused: regions are mapped here. */
void wm_init_backing(void *metadata, void *payload, unsigned selected) {
  (void)metadata;
  (void)payload;
  if (!initialized)
    wm_poisoncap_init(selected);
  else if (selected != mode)
    refuse("mode mismatch");
}

/* Poison every granule, sweep, clear, zero: stale capabilities into [ptr, ptr+n)
 * are dead afterwards and the storage can be handed out again. */
static void invalidate(void *ptr, size_t n) {
  if (!mode || !n)
    return;
  if ((cheri_getaddress(ptr) & 15) || (n & 15) ||
      (cheri_getperm(ptr) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
          (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    refuse("poison geometry or authority");
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                       CHERI_REVOKE_TAKE_STATS,
                   0, &info))
    refuse("sweep failed; storage cannot be reused");
  ++sweeps;
  poisoned += n;
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  /* Clearing access state does not erase the poison capability in memory. */
  memset(ptr, 0, n);
}

static struct entry *find(void *p, unsigned exact) {
  ptraddr_t address = cheri_getaddress(p);
  for (unsigned i = 0; i < count; ++i) {
    struct entry *e = &entries[i];
    if (!e->live)
      continue;
    ptraddr_t base = cheri_getaddress(e->region);
    if (exact ? address == base : address >= base && address < base + e->size)
      return e;
  }
  return NULL;
}

static void *issue(struct entry *e) {
  e->live = 1;
  if (++live > peak)
    peak = live;
  return e->region;
}

/* Regions are retained until process exit, in both modes, and reissued only
 * at their exact size, so libc free or revocation can never explain a pair. */
void *wm_sys_alloc(size_t n) {
  if (!n || n > WM_PAYLOAD_BYTES)
    refuse("system request size");
  size_t rounded = (n + 4095) & ~(size_t)4095;
  for (unsigned i = 0; i < count; ++i)
    if (!entries[i].live && entries[i].size == rounded)
      return issue(&entries[i]);
  if (count == WM_REGIONS)
    refuse("region table exhausted");
  void *region = mmap(NULL, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (region == MAP_FAILED)
    refuse("mmap");
  /* libc allocation removes SW_VMEM; a mapping keeps it. Without it the sweep
   * would revoke the manager's own pointers too. */
  if ((cheri_getperm(region) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
      (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    refuse("mapping lacks poison authority");
  struct entry *e = &entries[count++];
  e->region = region;
  e->size = rounded;
  ++created;
  return issue(e);
}

void wm_sys_free(void *p) {
  if (!p)
    return;
  struct entry *e = find(p, 1);
  if (!e)
    refuse("release of an unknown region");
  invalidate(e->region, e->size);
  ++releases;
  e->live = 0;
  --live;
}

void *wm_sys_realloc(void *p, size_t n) {
  if (!p)
    return wm_sys_alloc(n);
  if (!n) {
    wm_sys_free(p);
    return NULL;
  }
  struct entry *e = find(p, 1);
  if (!e)
    refuse("resize of an unknown region");
  if (((n + 4095) & ~(size_t)4095) <= e->size)
    return e->region;
  void *q = wm_sys_alloc(n);
  memcpy(q, e->region, e->size);
  wm_sys_free(p);
  return q;
}

/* A retained block starts a new epoch: every published alias into it dies. */
void *wm_epoch(void *p) {
  struct entry *e = find(p, 0);
  if (!e)
    refuse("epoch of an unknown region");
  invalidate(e->region, e->size);
  ++epochs;
  return (char *)e->region + (cheri_getaddress(p) - cheri_getaddress(e->region));
}

/* A pointer handed back to the allocator must still carry authority of its
 * own before block-wide authority is looked up by address. */
void *wm_widen(void *p) {
  (void)*(const volatile unsigned char *)p;
  struct entry *e = find(p, 0);
  if (!e)
    refuse("widen of an unknown pointer");
  return (char *)e->region + (cheri_getaddress(p) - cheri_getaddress(e->region));
}

/* Exact bounds, and neither poison nor mapping authority: the sweep must see
 * the published object as revocable. */
void *wm_publish(void *p, size_t n) {
  void *bounded = cheri_setboundsexact(p, n);
  if (!cheri_gettag(bounded)) {
    bounded = cheri_setbounds(p, n);
    ++unrepresentable;
  }
  return cheri_clearperm(bounded, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
}

/* The recycler's individual free: the chunk's data granules are invalidated
 * before the allocator writes its free-list node into them. */
void wm_release_chunk(void *p, size_t n) {
  if (!mode)
    return;
  invalidate(p, n);
  ++released_chunks;
}

void wm_backing_stats(struct wm_header *out) {
  out->regions_created = created;
  out->regions_peak = peak;
}

void wm_poisoncap_report(void) {
  printf("WM_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu epochs=%zu "
         "released_chunks=%zu region_releases=%zu regions=%llu "
         "unrepresentable=%zu pointer_bytes=%zu\n",
         mode, sweeps, poisoned, epochs, released_chunks, releases,
         (unsigned long long)created, unrepresentable, sizeof(void *));
  fflush(stdout);
}
