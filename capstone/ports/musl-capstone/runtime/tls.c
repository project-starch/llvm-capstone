/* The one thread a domain has.
 *
 * musl reaches errno, the locale and the cancellation state through the thread
 * pointer, so a domain that never sets one faults at the first failing syscall
 * rather than at the first thread. Measured 2026-09-16: write() to a bad fd
 * returned -EBADF correctly, __syscall_ret recognised it correctly, and the
 * domain then halted in __errno_location with tp reading 0.
 *
 * WHY __init_tp AND NOT A HAND-ROLLED STRUCT. __init_tp is musl's own entry
 * point for exactly this, the one pthread_create uses for every thread it
 * makes. It sets self, detach state, locale, the robust list and the thread
 * ring, and a hand-written substitute would have to be kept in step with all of
 * them across musl versions. The block below mirrors musl's own builtin_tls:
 * static storage, natural alignment, never freed, because a domain has exactly
 * one thread for its whole life.
 *
 * __init_tls() is the wrong entry point here, not a stricter one: it reads the
 * TLS program headers out of auxv to size the image, and a domain has no auxv.
 * musl's core uses the __thread keyword zero times; a domain's own __thread
 * variables (C-47) are laid out below instead, from symbols the domain linker
 * script defines around the PT_TLS segment.
 *
 * set_tid_address is answered with 1 rather than refused. A domain has one
 * thread and 1 is its identifier; nothing outside the domain consumes a tid and
 * musl only stores what comes back. It was refused at first, and the
 * unserved-syscall instrument in hostcall.c is what made the choice visible: it
 * reported syscall 96 as the one thing a full stdio run asked for and did not
 * get. A list with one permanent entry in it is a list people stop reading.
 */
/* pthread_impl.h declares a locale_t member and does not pull the type in
   itself; musl's own users of it get locale_t from their preamble. Naming the
   header that supplies it is clearer than copying someone else's include list. */
#include <locale.h>
#include "pthread_impl.h"

int __capstone_init_tls(void);

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static struct pthread __capstone_main_thread;

/* The TLS template, placed by my_first_domain/link.ld. Only their addresses are
   used, and only as differences and as a page offset: see the script. */
extern char __capstone_tls_image[], __capstone_tdata_end[], __capstone_tls_end[];

/* A domain with __thread variables (C-47). The compiler reaches one as
 * tp + %tprel(sym), with tp a capability, so tp must point at a block laid out
 * as lld computed %tprel -- RISC-V variant I, the TLS block starting AT tp and
 * struct pthread just below it, which is musl's TLS_ABOVE_TP with a zero gap --
 * and tp's bounds must cover the whole block. Built here the way musl's static
 * __init_tls builds builtin_tls, from one allocation, never freed: a domain has
 * one thread for its whole life.
 *
 * Alignment. link.ld aligns the segment's start to its own alignment, so lld's
 * offsets need tp congruent to the template's address modulo that alignment. It
 * is not known here, but it is at most 4096 (link.ld asserts it) and the image is
 * loaded on a page boundary, so tp is put at the template's offset within a page,
 * which satisfies every alignment up to a page at the cost of up to 4 KiB of
 * slack. Pointer arithmetic throughout: tp keeps the allocation's capability.
 *
 * domain_main calls this on EVERY entry, so the block is made once and kept:
 * thread-locals then live as long as the domain's globals do, and a domain
 * entered many times does not leak a block per entry.
 *
 * Without thread-locals the segment is empty and this is the static struct above,
 * exactly as before. */
static char *__capstone_tp;

int __capstone_init_tls(void)
{
	size_t tdata = __capstone_tdata_end - __capstone_tls_image;
	size_t memsz = __capstone_tls_end - __capstone_tls_image;
	if (memsz == 0)
		return __init_tp(&__capstone_main_thread);
	if (__capstone_tp)
		return __init_tp(__capstone_tp - sizeof(struct pthread));

	size_t page_off = (uintptr_t)__capstone_tls_image & 4095;
	/* calloc: __init_tp expects every field it does not set to be zero, as the
	   static struct above is, and .tbss is the zeroed tail of the block. */
	char *mem = calloc(1, sizeof(struct pthread) + 4095 + memsz);
	if (!mem)
		return -1;
	char *tp = mem + sizeof(struct pthread);
	tp += (page_off - ((uintptr_t)tp & 4095)) & 4095;
	memcpy(tp, __capstone_tls_image, tdata);
	__capstone_tp = tp;
	return __init_tp(tp - sizeof(struct pthread));
}
