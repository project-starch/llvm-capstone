/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * The contract is the one in ../../SCHEMA.md; this
 * header is its Wireshark seam. shared/driver.c supplies the entry point, the
 * scopes, the per-dissection packet pool and the labelled probes; a case
 * supplies only its sequence, inside WM_CASE(NN).
 *
 * The allocator is real: wmem's core and its four allocators from the pinned
 * 4.6.8 release, compiled unmodified but for the Sublet patch the port
 * applies in its protected build. The consumers are reduced to the allocator calls the upstream
 * defect makes, in the same order, because reaching them in place needs a
 * dissector, a capture file and the whole of epan.
 */
#ifndef WM_CORPUS_H
#define WM_CORPUS_H

#include "port.h"
#include "scopes.h"
#include "wmem_core.h"
#include <string.h>

/* A case declares the number its directory carries. The driver refuses a
 * fixture that names another case rather than silently running it. */
#define WM_CASE(n)                                                             \
  const unsigned wm_case_number = (n);                                         \
  void wm_case_run(void)

extern const unsigned wm_case_number;
void wm_case_run(void);

/* The per-dissection packet pool the driver acquired, as pinfo->pool. */
extern wmem_allocator_t *wm_packet;

/* Dissect the next packet: epan_dissect_reset frees the packet pool, which
 * ends every packet-scoped object at once. This is the event every case in
 * this corpus turns on. */
void wm_next_packet(void);

/* Where a case parks the alias it will read after the lifetime ends. Volatile
 * and external, so the sequence cannot be optimised into nothing. */
extern unsigned char *volatile wm_held;

/* The stale access, labelled so a run can require the fault to land HERE
 * rather than merely somewhere. wm_mark() prints the case's ready mark and
 * must be the LAST thing before the access: its presence is the evidence that
 * the setup ran. */
unsigned wm_probe(const volatile unsigned char *p);
void wm_write_probe(volatile unsigned char *p);
void wm_mark(void);

_Noreturn void wm_give_up(unsigned long code);

/* THE NATIVE FIX DIFFERENTIAL (added 2026-10-09): `program 0 N buggy|fixed`.
 * Every other invocation -- `program 0 N`, which every other arm runs -- leaves all three flags 0,
 * so every allocation, store and the labelled access those arms measure are unchanged; the only
 * addition on their path is that the probe's result is kept (WM_READ/WM_WRITE), after the
 * access.
 *   wm_fixed    run the upstream fix's behaviour where the case models it (`if (wm_fixed) ...`).
 *   wm_observe  make a stale access VISIBLE natively: after a lifetime ends, the next dissection's
 *               first allocation reoccupies the freed storage and is filled with WM_MARKER, so a
 *               read through the stale pointer returns the marker (wm_reoccupy).
 *   wm_defect   the case's own verdict: the access reached storage that is not its object's --
 *               a read returned another object's marker, or a write landed past the object
 *               (the case's CHECKs establish where). */
#define WM_MARKER 0x5a
extern int wm_fixed, wm_observe, wm_defect;
void wm_reoccupy(const void *stale, unsigned long bytes);
#define WM_READ(p, marker) (wm_defect = (wm_probe(p) == (unsigned)(marker)))
#define WM_WRITE(p) (wm_write_probe(p), wm_defect = 1)
/* Spatial rows: the verdict is WHERE the access went, the definition the plain-heap corpora use for
 * `crossed`. The address arithmetic the case CHECKs is what places it outside [obj, obj + n), so a
 * fix that clamps the access to its object reads FIXED while still performing it. */
#define WM_OUTSIDE(p, obj, n)                                                  \
  ((const unsigned char *)(p) < (const unsigned char *)(obj) ||               \
   (const unsigned char *)(p) >= (const unsigned char *)(obj) + (n))
#define WM_READ_AT(p, obj, n) ((void)wm_probe(p), wm_defect = WM_OUTSIDE(p, obj, n))
#define WM_WRITE_AT(p, obj, n) (wm_write_probe(p), wm_defect = WM_OUTSIDE(p, obj, n))
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      wm_give_up(n);                                                           \
  } while (0)

#endif
