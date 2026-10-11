/* What the two PoisonCap adapters share: the manager's authority over a
 * mapped node, and the three operations on it. */
#ifndef APRP_POISON_H
#define APRP_POISON_H
#include <stddef.h>
/* The manager's pointer for the node containing p: full bounds, POISON and
 * SW_VMEM retained. Fails the run for an address outside every node. */
void *aprp_poison_authority(const void *p);
/* Exact bounds, neither poison nor mapping authority: revocable by a sweep. */
void *aprp_poison_publish(void *manager, size_t n);
/* Mode 1: poison every granule of [p, p+n), sweep, clear, zero. Mode 0: nothing. */
void aprp_poison_invalidate(void *manager, size_t n);
void aprp_poison_report(void);
#endif
