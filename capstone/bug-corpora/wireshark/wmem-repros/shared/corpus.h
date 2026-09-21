/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * The contract is the one in ../../cpython/pymalloc-repros/SCHEMA.md; this
 * header is its Wireshark seam. shared/driver.c supplies the entry point, the
 * scopes, the per-dissection packet pool and the labelled probes; a case
 * supplies only its sequence, inside WM_CASE(NN).
 *
 * The allocator is real: wmem's core and its four allocators from the pinned
 * 4.6.8 release, compiled unmodified but for the guarded authority hooks the
 * port applies. The consumers are reduced to the allocator calls the upstream
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
 * rather than merely somewhere. wm_mark() publishes the case and the probe
 * addresses and must be the LAST thing before the access: its presence is the
 * evidence that the setup ran. */
unsigned wm_probe(const volatile unsigned char *p);
void wm_write_probe(volatile unsigned char *p);
void wm_mark(void);

_Noreturn void wm_give_up(unsigned long code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      wm_give_up(n);                                                           \
  } while (0)

#endif
