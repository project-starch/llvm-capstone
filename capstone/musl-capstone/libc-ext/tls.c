/* THE DOMAIN'S struct __pthread, initialised.
 *
 * start-musl.S gives tp a capability with writable space below it, and its own
 * comment says what that space was for: "not a TLS image: just addressable,
 * writable space so musl's errno has somewhere to live". Errno was the only
 * consumer anyone had, and libc-ext/errno.c does not even use tp -- so the block
 * stayed all zeros and nothing noticed.
 *
 * musl reaches a great deal more than errno through __pthread_self(). The one
 * that found this is strerror:
 *
 *     #define CURRENT_LOCALE (__pthread_self()->locale)     locale_impl.h:40
 *     char *strerror(int e) { return __strerror_l(e, CURRENT_LOCALE); }
 *
 * `locale` is a POINTER field, so reading it is a capability load out of the
 * block, and out of a zeroed block it comes back untagged. The next use of it
 * takes a cause-24 -- not where the zero is, but one dereference later:
 *
 *     movc a3, tp            fine, tp is a real capability (measured: tag=1)
 *     ldc  a3, -0x60(a3)     fine, reads the locale field
 *     ldc  a1, 0x50(a3)      cause 24: the field was zero
 *
 * WHY NOT musl's OWN __init_tp, which is in the archive and sets exactly this
 * field: it also calls __set_thread_area, which on riscv64 sets tp with an
 * ordinary integer move and would strip the tag we just went to the trouble of
 * keeping (arch-capstone64/pthread_arch.h), and SYS_set_tid_address, which a
 * domain does not have. What is left of it after removing those is this file.
 *
 * The LAYOUT comes from musl's own header, not from the offsets above. Those
 * are in the comment because they are what the fault report shows; hard-coding
 * them would break the next time musl moves a field.
 *
 * Single-threaded on purpose: self, next and prev all point at this one thread
 * because that is the truth in a domain, not a placeholder.
 */
#include "pthread_impl.h"
#include "locale_impl.h"

void __capstone_init_tp(void)
{
	pthread_t td = __pthread_self();

	td->self = td;
	td->detach_state = DT_JOINABLE;
	/* A zeroed global_locale is the C locale: every cat[] entry is null and
	   __lctrans returns the untranslated message, which is what a domain
	   wants. The field has to be non-null; its CONTENTS may be zero. */
	td->locale = &libc.global_locale;
	td->robust_list.head = &td->robust_list.head;
	td->next = td->prev = td;
}
