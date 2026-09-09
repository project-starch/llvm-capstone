/* Compatibility shim for building the 2024 MicroPython tree with our 2026-era port.
 *
 * Between ce491ab0d1 (2024-04-22) and the pin, the mp_int_t/mp_uint_t typedefs moved
 * INTO py/mpconfig.h. Before that move they were the port's responsibility. Our
 * mpconfigport.h is written against the pin and therefore correctly omits them, which
 * leaves the 2024 tree with no definition at all -- the symptom is a cascade of
 * "function cannot return function type 'mp_int_t'" errors.
 *
 * Injected with -include so the committed port header stays untouched.
 */
#ifndef MPY2024_COMPAT_H
#define MPY2024_COMPAT_H
#include <stdint.h>
typedef intptr_t mp_int_t;
typedef uintptr_t mp_uint_t;

/* The 2024 py/stream.c uses SEEK_SET/SEEK_CUR/SEEK_END, which the freestanding
   shim does not provide and which this domain has no filesystem to need. */
#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

#endif
