/* Runs one libc-test program inside a pure-capability domain.
 *
 * libc-test's contract is small and this file is all of it: each test is a
 * main() that calls t_error() on every failure, t_error() bumps t_status and
 * prints the location, and main() returns t_status. The test is compiled with
 * -Dmain=libc_test_main so that this file can own the domain entry, call it,
 * and hand the status back through metadata->result like every other probe.
 *
 * argv is supplied because a few tests take it: main(int, char **) called
 * with (1, {"test", 0}) is what the harness's own runtest.c would do, and a
 * main(void) ignores the extras under this ABI.
 *
 * The unserved-syscall list is printed here, after the test, because a test
 * that "passed" while a syscall it needed was refused has not passed anything.
 * It is folded into the status so the host sees it without parsing text.
 */
#include <stdio.h>

extern volatile int t_status;
int libc_test_main(int argc, char **argv);
unsigned long __capstone_unserved_count(void);
long __capstone_unserved_at(unsigned long i);

/* Status codes above 0xff cannot collide with t_status, which is 0 or 1. */
#define LT_UNSERVED 0x100

/* musl's __environ is defined in src/env/__environ.c as a null pointer and is
   set by __libc_start_main, which a domain does not run. An empty environment
   is a valid one; a null environ is not, and env.c says so in its first check.
   This belongs in the domain entry generally, and it is here first because the
   test is what found it. */
extern char **__environ;

/* The report, written once whether the test returns from main or calls exit().
   A test that fails an assertion and exits would otherwise leave the runner with
   no line to read, which reads as a hang and costs the boot; libc-test's mntent
   is such a test. __capstone_at_exit is the runtime's hook for that path. */
static int lt_reported;

static int lt_report(int r)
{
	if (lt_reported)
		return r;
	lt_reported = 1;

	unsigned long n = __capstone_unserved_count();
	if (n) {
		printf("libc-test: UNSERVED syscalls:");
		for (unsigned long i = 0; i < n; i++) {
			long nr = __capstone_unserved_at(i);
			if (nr < 0) {   /* the ring keeps the first few, the count is exact */
				printf(" (+%lu more)", n - i);
				break;
			}
			printf(" %ld", nr);
		}
		printf("\n");
		fflush(stdout);
		r |= LT_UNSERVED;
	}
	fflush(stdout);
	return r;
}

int __capstone_at_exit(int status)
{
	return lt_report(status ? status : t_status);
}

int capstone_main(void)
{
	static char *argv[] = { "libc-test", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	int r = libc_test_main(1, argv);
	if (r == 0)
		r = t_status;
	return lt_report(r);
}
