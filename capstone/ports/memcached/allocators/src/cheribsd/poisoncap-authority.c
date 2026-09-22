/* PoisonCap: the authority layer under the shared ledger, built from the
 * published platform's poison, sweep and clear primitives.
 *
 * The ledger in src/shared/leases.c is unchanged and is the same file the
 * Capstone domain runs: it owns the page map, the chunk and object states and
 * the counters, and it calls down here for authority. Only this file differs
 * between the two systems, so the arms measure one bookkeeping and two
 * mechanisms.
 *
 * Who holds what. One mmap'd arena carries POISON and SW_VMEM: that is the
 * MANAGER authority, it stays in the ledger's slots, and the kernel's revoker
 * skips it. Every alias handed to memcached is derived from it, bounded to the
 * unit, and stripped of both permissions -- so a sweep can take it away, and
 * an ordinary consumer can never poison anything.
 *
 * What a release costs in mode 1: cpoison on each 16-byte granule of the unit,
 * one synchronous sweep, cclearpoison on each granule, and a memset. The
 * memset is not hygiene: cclearpoison resets access state but leaves the
 * poison capability stored in the payload, which a later sweep would read as a
 * fresh lease. Mode 0 does none of it, which is the point of the pair. */
#include "port.h"
#include "poisoncap.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef CHERI_PERM_POISON
#error "MCP_POISONCAP requires the published PoisonCap SDK and its kernel"
#endif

#define MANAGER_PERMS (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)

static unsigned char *arena;
static uintptr_t base[2], cursor[2], limit[2];
static unsigned protected_mode, initialized;
static size_t sweeps, poison_bytes, clear_bytes, zeroed_bytes;

void mcp_authority_init(void *payload) {
  /* The region is this layer's own: libc's allocator strips SW_VMEM, and
   * without it the sweep revokes the manager too and cclearpoison then faults
   * on its own dead capability. The hosted entry passes NULL for that reason. */
  if (payload || initialized)
    mcp_fail(532);
  if (!feature_present("cheri_caprevoke_poison"))
    mcp_fail(531);
  void *region = mmap(NULL, MCP_PAYLOAD_BYTES, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
  if (region == MAP_FAILED)
    mcp_fail(533);
  if ((cheri_getperm(region) & MANAGER_PERMS) != MANAGER_PERMS ||
      cheri_getlen(region) < MCP_PAYLOAD_BYTES ||
      (cheri_getaddress(region) & (MCP_GRAIN - 1)))
    mcp_fail(534);
  arena = region;
  base[MCP_PAGES] = cursor[MCP_PAGES] = cheri_getaddress(arena);
  limit[MCP_PAGES] = base[MCP_PAGES] + MCP_PAGE_HALF;
  base[MCP_OBJECTS] = cursor[MCP_OBJECTS] = limit[MCP_PAGES];
  limit[MCP_OBJECTS] = base[MCP_PAGES] + MCP_PAYLOAD_BYTES;
  initialized = 1;
}
int mcp_authority_can_revoke(void) { return 1; }
void mcp_authority_set_mode(unsigned mode) { protected_mode = mode ? 1 : 0; }
uintptr_t mcp_authority_base(unsigned half) { return base[half]; }
size_t mcp_authority_used(unsigned half) { return cursor[half] - base[half]; }

/* Manager authority over [address, address+size). Exact, never rounded up: a
 * capability longer than its unit would put a live neighbour's base inside the
 * granules this one poisons, and the sweep would take the neighbour with it.
 * A size the format cannot express exactly is refused here rather than widened
 * silently -- see the README's note on the representability boundary. */
static void *bounded(uintptr_t address, size_t size) {
  void *p = cheri_setboundsexact(arena + (address - base[MCP_PAGES]), size);
  if (!cheri_gettag(p) || cheri_getaddress(p) != address ||
      cheri_getlen(p) != size)
    mcp_fail(535);
  return p;
}

int mcp_authority_carve(unsigned half, size_t size, capstone_cap_slot *out,
                        uintptr_t *at) {
  if (half > 1 || !size)
    return 0;
  /* Align the cursor so the bounds below are expressible; the units the
   * allocators ask for are 16-byte multiples and the mask is a power of two. */
  size_t mask = CHERI_REPRESENTABLE_ALIGNMENT_MASK(size);
  uintptr_t start = (cursor[half] + ~mask) & mask;
  if (start < cursor[half] || start > limit[half] - size)
    return 0;
  cursor[half] = start + size;
  *at = start;
  out->c = bounded(start, size);
  return 1;
}

/* The prefix up to `split_end` leaves as its own authority; `from` advances to
 * the remainder. The ledger dictates the geometry -- chunk i at
 * page_base + i * chunk_size -- so this adds no padding of its own. */
void mcp_authority_split(capstone_cap_slot *from, uintptr_t split_end,
                         capstone_cap_slot *out) {
  unsigned char *p = from->c;
  uintptr_t start = cheri_getaddress(p);
  if (split_end <= start)
    mcp_fail(535);
  out->c = bounded(start, split_end - start);
  from->c = p + (split_end - start);
}

/* The client alias: the unit's bounds without the two permissions that make a
 * capability a manager. Stripping POISON is what stops a consumer poisoning
 * its own storage; stripping SW_VMEM is what lets the sweep take this alias. */
void *mcp_authority_take(capstone_cap_slot *region) {
  void *client = cheri_clearperm(region->c, MANAGER_PERMS);
  if (!cheri_gettag(client) || (cheri_getperm(client) & MANAGER_PERMS))
    mcp_fail(536);
  return client;
}

static void invalidate(capstone_cap_slot *region) {
  unsigned char *manager = region->c;
  size_t n = cheri_getlen(manager);
  if (!protected_mode)
    return;
  if (!n || (n & 15) || (cheri_getaddress(manager) & 15) ||
      (cheri_getperm(manager) & MANAGER_PERMS) != MANAGER_PERMS)
    mcp_fail(537);
  for (size_t i = 0; i < n; i += 16) {
    void *granule = manager + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(granule) : "memory");
  }
  poison_bytes += n;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS,
                   0, &info))
    mcp_fail(538); /* storage is never reused after a sweep that did not run */
  ++sweeps;
  for (size_t i = 0; i < n; i += 16) {
    void *granule = manager + i;
    __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(granule) : "memory");
  }
  clear_bytes += n;
  memset(manager, 0, n);
  zeroed_bytes += n;
}

void *mcp_authority_renew(capstone_cap_slot *region) {
  invalidate(region);
  return mcp_authority_take(region);
}
void mcp_authority_reclaim(capstone_cap_slot *region) { invalidate(region); }

void mcp_poisoncap_report(void) {
  printf("MCP_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
         "zeroed_bytes=%zu page_bytes=%zu object_bytes=%zu pointer_bytes=%zu\n",
         protected_mode, sweeps, poison_bytes, clear_bytes, zeroed_bytes,
         mcp_authority_used(MCP_PAGES), mcp_authority_used(MCP_OBJECTS),
         sizeof(void *));
}
