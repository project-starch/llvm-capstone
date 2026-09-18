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
 * TLS program headers out of auxv to size the image, and a domain has no auxv
 * and no TLS image. musl's core uses the __thread keyword zero times, so there
 * is nothing for it to size.
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

static struct pthread __capstone_main_thread;

int __capstone_init_tls(void)
{
	return __init_tp(&__capstone_main_thread);
}
