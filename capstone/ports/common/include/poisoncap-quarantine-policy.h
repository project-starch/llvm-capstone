#ifndef POISONCAP_QUARANTINE_POLICY_H
#define POISONCAP_QUARANTINE_POLICY_H

#include <stddef.h>

/* SQLite MEMSYS5 policy in PoisonCap artifact 731c5720d1a6321d436a8d7b85bf3aa2e4971fe0.
 * This is a transfer to other nested allocators, not an author-provided port.
 * Keep the full-queue revocation correction explicit at the call site.
 * See experiments/study/poisoncap-policy.md for provenance and scope. */
#define POISONCAP_QUARANTINE_ENTRIES 4096u
#define POISONCAP_MIN_HELD_BYTES (16UL * 1024 * 1024)

static inline int
poisoncap_quarantine_threshold(size_t held, size_t quarantined)
{
  /* Equivalent to 4*quarantined >= held without size_t overflow. */
  return held >= POISONCAP_MIN_HELD_BYTES &&
         quarantined >= held / 4 + (held % 4 != 0);
}

#endif
