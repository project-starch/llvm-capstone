/* The runtime's report of unserved syscalls, from a program that closed fd 1
 * (ISSUES I-11).
 *
 * Two uname() calls, which the runtime does not serve, then close(1). The report
 * is written after the program ends. Before I-11 it went through the program's
 * own descriptors, was refused with EBADF, and the missing line read as "nothing
 * unserved".
 */
#include <stdio.h>
#include <unistd.h>
#include <sys/utsname.h>

int main(void)
{
	struct utsname u;
	uname(&u);
	uname(&u);
	printf("UNSERVED-TEST two uname calls, closing fd 1\n");
	fflush(stdout);
	close(1);
	return 0;
}
