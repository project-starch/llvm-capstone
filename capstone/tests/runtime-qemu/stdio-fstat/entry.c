/* Domain entry for this test: main() under C's exit semantics, so stdio is
   flushed (musl buffers stdout fully in a domain, where the tty ioctl fails).
   __capstone_at_exit is DEFINED here: on a runtime that only declares it weak,
   the address of an undefined weak symbol is not NULL in a domain (C-56). */
#include <stdlib.h>
extern char **__environ;
int main(int argc, char **argv);
int __capstone_at_exit(int status) { return status; }
int capstone_main(void)
{
	static char *argv[] = { "stdio-fstat", 0 };
	static char *envp[] = { 0 };
	__environ = envp;
	exit(main(1, argv));
}
