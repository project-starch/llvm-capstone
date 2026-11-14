#ifndef LIFE_INJECT_H
#define LIFE_INJECT_H
#define capstone_cap_derive life_real_derive
#include <capstone/capability.h>
#undef capstone_cap_derive
extern unsigned long life_calls, life_fail;
static inline void *capstone_cap_derive(void *parent, unsigned long off, unsigned long size)
{
  if (++life_calls == life_fail) return 0;
  return life_real_derive(parent, off, size);
}
#endif
