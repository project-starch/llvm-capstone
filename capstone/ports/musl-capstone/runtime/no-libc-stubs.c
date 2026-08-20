/* The two symbols start-musl.S needs, for a domain that links NO libc.
 *
 * The glue publishes dom_data and asks for the thread-pointer struct to be
 * populated; libc-capstone.a provides both. yield-probe wants only the resumable
 * yield and links start-musl.o plus its own domain, no archive -- so it links
 * this instead, and the requirement stays visible on the link line.
 *
 * NOT weak definitions inside the glue, which is what this replaces: a weak
 * definition satisfies the reference, the linker then never extracts tls.o from
 * the archive, and a domain that DOES link the libc silently loses its real
 * thread-pointer initialisation. See the note in start-musl.S.
 *
 * A domain using these gets a zeroed dom_data slot, so libc-ext/malloc.c (if it
 * were linked, which it is not) would fall back to its static heap, and an
 * uninitialised tp -- both correct for a domain that has neither. */

/* Sixteen bytes and 16-aligned: the glue stores a CAPABILITY here with stc. A
 * void * is exactly that on this target. */
void *__capstone_dom_data;

void __capstone_init_tp(void) {}
