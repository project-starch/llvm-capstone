#ifndef FFPOOL_ALIAS_SCATTER_H
#define FFPOOL_ALIAS_SCATTER_H

#include <capstone/capability-slot.h>
#include <stdint.h>

/* Synthetic two-pool hierarchy for the alias-scatter security probe. Slots
 * hold linear authority and must never be copied. The sibling remains outside
 * the parent subtree so it can hold one of the scattered child aliases. */
struct ff2_alias_scatter {
  capstone_cap_slot parent, parent_handle, child;
  capstone_cap_slot sibling, new_child;
  volatile unsigned char *child_alias;
  volatile unsigned char *sibling_alias;
  volatile unsigned char *new_alias;
  uintptr_t child_address;
};

void ff2_alias_scatter_setup(struct ff2_alias_scatter *);
void ff2_alias_scatter_transition(struct ff2_alias_scatter *, unsigned, unsigned);
unsigned ff2_alias_scatter_register_span(volatile unsigned char *,
    capstone_cap_slot *, unsigned, unsigned, unsigned,
    volatile unsigned char *, unsigned long, uintptr_t);

#endif
