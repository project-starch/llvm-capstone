#ifndef CAPSTONE_CAPABILITY_SLOT_H
#define CAPSTONE_CAPABILITY_SLOT_H

/* Owned capabilities live in slots: copying a linear capability consumes it.
 * Pass the slot's address to operations; never copy a slot or its enclosing
 * record. Native control builds use the same shape with an ordinary pointer.
 * This header defines storage only and can be included by either target. */
typedef struct capstone_cap_slot {
  void *c;
} capstone_cap_slot;

#endif
