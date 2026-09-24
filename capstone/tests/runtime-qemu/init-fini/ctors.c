/* Constructors and destructors in a musl domain (ISSUES C-64).
 *
 * One of each with a priority and two without, and one constructor reads the
 * environment, which in C exists before any constructor runs. The order they
 * print in is the order they ran; run.sh compares it with the same file built natively, so the
 * reference is the toolchain's, not an order written down here. Before C-64 no
 * constructor ran in a domain, and exit() halted (cause 24) on the first
 * .fini_array slot, loaded through an integer address.
 */
#include <stdio.h>
#include <stdlib.h>

static int order;

__attribute__((constructor(101))) static void ctor_p(void) { printf("CTOR p %d\n", ++order); }
__attribute__((constructor)) static void ctor_a(void)
{
	/* In C the environment exists before any constructor runs. */
	const char *e = getenv("INIT_FINI_ENV");
	printf("CTOR a %d env=%s\n", ++order, e ? e : "(unset)");
}
__attribute__((constructor)) static void ctor_b(void) { printf("CTOR b %d\n", ++order); }
__attribute__((destructor(101))) static void dtor_p(void) { printf("DTOR p %d\n", ++order); }
__attribute__((destructor)) static void dtor_a(void) { printf("DTOR a %d\n", ++order); }
__attribute__((destructor)) static void dtor_b(void) { printf("DTOR b %d\n", ++order); }

int main(void)
{
	printf("MAIN %d\n", ++order);
	return 0;
}
