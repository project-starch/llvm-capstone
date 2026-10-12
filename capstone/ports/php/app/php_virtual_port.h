/* Geometry shared by the virtual port's own files, in one place and checked once.
 *
 * ONE NUMBER, because two copies of it is how a port stops matching the program it is porting.
 * php_virtual_lower.c's zend_cap_widen needs the offset from PHP's object pointer back to the
 * block the heap handed out, and php_virtual_main.c is where PHP's own headers are in scope, so
 * the constant lives here and the assertion lives there.
 *
 * WHY ASSERT IT AT ALL. sizeof(zend_mem_header) is 48 only given the configuration this corpus
 * tree actually has: ZEND_DEBUG 0 and MEMORY_LIMIT 0 (main/php_config.h:2373,2382) leave the
 * header as {pNext, pLast, size:31, cached:1}, which at __alignof__(void *) == 16 is 48 bytes
 * with MEM_HEADER_PADDING 0. A tree configured with ZEND_DEBUG on grows the header, and every
 * derivation in the port would then be silently off -- the widen would look up the wrong block
 * and _efree would read a neighbour's header. That must fail the BUILD, not the run.
 */
#ifndef PHP_VIRTUAL_PORT_H
#define PHP_VIRTUAL_PORT_H

#define ZEND_CAP_HEADER_BYTES 48UL

/* A C89-compatible compile-time assertion: a negative array size is an error, and gnu89 has no
 * _Static_assert. The name carries the claim, so the compiler's message names it too. */
#define PHP_VIRTUAL_ASSERT(name, cond) typedef char php_virtual_assert_##name[(cond) ? 1 : -1]

#endif
