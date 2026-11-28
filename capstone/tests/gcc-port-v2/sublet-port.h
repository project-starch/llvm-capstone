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

/* CRENEW: end OBJECT's lifetime and begin a new one in the same place.
   Fail-stop (nested lane, 2026-11-26, CFP-g3 v2): CRENEW returns an untagged 0
   and leaves the old lifetime live when the table cannot give the renewal a new
   identity and cannot grow. Every caller stores through the result at once, and
   bitmap_elt_clear_from would otherwise end its free chain silently with the old
   lifetime still live. So on 0 the old lifetime is ended (CREVOKE needs no new
   identity), one line goes to stderr, and the process aborts. Under
   SUBLET_RENEW_FORCE_FAIL (the control build) every renewal reads as that
   failure without executing CRENEW.  */
extern long write (int, const void *, unsigned long);
static inline void *
sublet_renew (void *object)
{
  void *renewed;
#ifdef SUBLET_RENEW_FORCE_FAIL
  renewed = 0;
#else
  __asm__ volatile (".insn r 0x5b, 0x1, 0x54, %0, %1, x0"
		    : "=r" (renewed) : "r" (object) : "memory");
#endif
  if (__builtin_expect (renewed == 0, 0))
    {
      sublet_release (object);
      write (2, "SUBLET-RENEW-FAILED\n", 20);
      __builtin_abort ();
    }
  return renewed;
}

/* CDERIVE refuses bounds that the reference format cannot hold exactly.
   Every length below 4 KiB is exact.  */
#define SUBLET_EXACT_LIMIT 4096

#endif
