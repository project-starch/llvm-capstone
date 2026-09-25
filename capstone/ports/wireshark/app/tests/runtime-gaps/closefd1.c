/* ISSUES I-11: the runtime reports unserved syscalls on fd 1 after the program ends, so a program
 * that closed fd 1 loses the report. Two unserved calls (uname), then close(1). */
#include <stdio.h>
#include <unistd.h>
#include <sys/utsname.h>
int main(void)
{
	struct utsname u;
	uname(&u);
	uname(&u);
	printf("CLOSEFD1\n");
	fflush(stdout);
	close(1);
	return 0;
}
