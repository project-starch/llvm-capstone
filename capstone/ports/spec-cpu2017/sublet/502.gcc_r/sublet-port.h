/* Sublet lifetimes for the allocators of SPEC CPU2017 502.gcc_r: the three
   instructions. apply.py copies this file next to the benchmark's sources.  */
#ifndef SUBLET_PORT_H
#define SUBLET_PORT_H

/* CDERIVE: a child lifetime of AT's object, SIZE bytes from AT's address.  */
static inline void *
sublet_derive (void *at, unsigned long size)
{
  void *child;
  __asm__ volatile (".insn r 0x5b, 0x1, 0x51, %0, %1, %2"
		    : "=r" (child) : "r" (at), "r" (size) : "memory");
  return child;
}

/* CREVOKE: end OBJECT's lifetime.  */
static inline void
sublet_release (void *object)
{
  __asm__ volatile (".insn r 0x5b, 0x1, 0x52, x0, %0, x0"
		    : : "r" (object) : "memory");
}

/* CRENEW: end OBJECT's lifetime and begin a new one in the same place.  */
static inline void *
sublet_renew (void *object)
{
  void *renewed;
  __asm__ volatile (".insn r 0x5b, 0x1, 0x54, %0, %1, x0"
		    : "=r" (renewed) : "r" (object) : "memory");
  return renewed;
}

/* CDERIVE refuses bounds that the reference format cannot hold exactly.
   Every length below 4 KiB is exact.  */
#define SUBLET_EXACT_LIMIT 4096

#endif
