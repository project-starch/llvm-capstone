/* __thread in a musl domain (C-47).
 *
 * Built by run.sh at -O0 and -O2, and twice more as controls. Every check prints
 * one TLS-TEST line; the status is the number that failed, so LT-RESULT status=0
 * is the pass and a missing line is a failure, never "no result".
 *
 *   initialized  .tdata: a thread-local starts at its initial value, and a write
 *                sticks. The runtime copied the template.
 *   zeroed       .tbss: 100 bytes start zero.
 *   align64      a 64-byte-aligned thread-local is 64-aligned at run time.
 *   align4096    a page-aligned one is page-aligned: the case that needs tp placed
 *                at the template's offset within a page, not just 16-aligned.
 *   other-tu     a thread-local defined in another translation unit, which clang
 *                asks for as initial-exec; the backend lowers it local-exec and
 *                both units agree on its address.
 *   capability   a pointer stored in a thread-local keeps its tag: read back
 *                through it after a call.
 *   bounds       the capability for a thread-local covers that variable and no
 *                more (-capstone-shrink-globals, as for a global).
 *   errno        errno still works: musl reaches it through tp, and tp is now the
 *                start of the TLS block with struct pthread below it.
 *
 * -DOVERRUN (the control tls-overrun) writes one byte past `zeroed` and must halt
 * on the capability's bounds.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

__thread int counter = 41;
__thread char zeroed[100];
/* Not static, and written at run time: a static thread-local that is only read
   is folded to a constant and deleted at -O2, and so is its alignment test --
   the compiler knows the address it would have. */
__thread long a64 __attribute__((aligned(64))) = 7;
__thread char a4096 __attribute__((aligned(4096))) = 9;
__thread char *held;
extern __thread int other;
int *other_address(void);
unsigned long address_of(void *p); /* tls_other.c: opaque to this unit */

static int failures;

static void check(int ok, const char *name, long got, long want)
{
	printf("TLS-TEST %s %s got=%ld want=%ld\n", ok ? "PASS" : "FAIL", name, got, want);
	failures += !ok;
}

__attribute__((noinline)) static char *read_held(void) { return held; }

int main(void)
{
#ifdef OVERRUN
	volatile int i = 100;
	printf("TLS-TEST overrun: writing zeroed[%d]\n", i);
	fflush(stdout);
	zeroed[i] = 1;
	printf("TLS-TEST overrun: NOT stopped\n");
	fflush(stdout);
	return 1;
#else
	check(counter == 41, "initialized", counter, 41);
	counter++;
	check(counter == 42, "initialized-write", counter, 42);

	int nonzero = 0;
	for (int i = 0; i < 100; i++)
		nonzero += zeroed[i] != 0;
	check(nonzero == 0, "zeroed", nonzero, 0);

	long mis64 = (long)(address_of(&a64) & 63);
	check(mis64 == 0 && a64 == 7, "align64", mis64 * 1000 + a64, 7);
	long mis4096 = (long)(address_of(&a4096) & 4095);
	check(mis4096 == 0 && a4096 == 9, "align4096", mis4096 * 1000 + a4096, 9);
	a64 += 1;
	a4096 += 1;

	check(other == 1234 && other_address() == &other, "other-tu",
	      other + (other_address() == &other ? 0 : 1000000), 1234);

	held = malloc(32);
	strcpy(held, "kept");
	char *back = read_held();
	check(back && strcmp(back, "kept") == 0, "capability", back ? back[0] : -1, 'k');

	long zb = (long)(__builtin_capstone_cap_get_end(zeroed) - __builtin_capstone_cap_get_base(zeroed));
	long cb = (long)(__builtin_capstone_cap_get_end(&counter) - __builtin_capstone_cap_get_base(&counter));
	check(zb == 100 && cb == (long)sizeof counter, "bounds", zb * 1000 + cb,
	      100 * 1000 + (long)sizeof counter);

	errno = 0;
	long w = write(99, "x", 1);
	check(w == -1 && errno == EBADF, "errno", errno, EBADF);

	printf("TLS-TEST-DONE failures=%d\n", failures);
	fflush(stdout);
	return failures;
#endif
}
