/* The runtime's report of unserved syscalls, from a program that closed fd 1
 * (ISSUES I-11).
 *
 * Two brk calls, which the delegated runtime refuses by design (memory is the
 * domain allocator's and never crosses: the MEMORY group of the shape table,
 * runtime/common/delegate.c), then close(1). The report is written inside
 * exit_group, after the program has ended, to the task's fd 2. Before I-11 it
 * went through the program's own fd 1, was refused with EBADF, and the missing
 * line read as "nothing unserved". ../run-delegated-probes.py wants exactly
 * "capstone-domain: UNSERVED syscalls: 214x2" on stderr.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(void)
{
	long a = syscall(SYS_brk, 0);
	long b = syscall(SYS_brk, 0);
	printf("UNSERVED-TEST two brk calls (%ld %ld), closing fd 1\n", a, b);
	fflush(stdout);
	close(1);
	return 0;
}
