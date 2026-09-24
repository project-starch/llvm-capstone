/* A domain that RETURNS from its entry, never calling exit(): three lines of
 * stdout still in musl's buffer and an atexit handler registered. C says
 * returning from main is exit(), so all three lines and the handler's line must
 * reach the host, and the status must be the returned 5. run.sh checks that,
 * and that a runtime which returns straight to the host (the control) loses
 * them. __capstone_at_exit is DEFINED here: the control's runtime only declares
 * it weak, and an undefined weak symbol's address is not NULL in a domain (C-56). */
#include <stdio.h>
#include <stdlib.h>

int __capstone_at_exit(int status) { return status; }

static void on_exit_handler(void) { puts("RETURN-FLUSH atexit handler ran"); }

int capstone_main(void)
{
	atexit(on_exit_handler);
	puts("RETURN-FLUSH line 1");
	puts("RETURN-FLUSH line 2");
	puts("RETURN-FLUSH line 3");
	return 5;
}
