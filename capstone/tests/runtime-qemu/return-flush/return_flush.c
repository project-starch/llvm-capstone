/* A program that RETURNS from main, never calling exit(): three lines of
 * stdout still in musl's buffer (stdout is a pipe, so fully buffered) and an
 * atexit handler registered. C says returning from main is exit(), so all three
 * lines and the handler's line must reach the launcher's stdout, and the status
 * must be the returned 5. ../run-delegated-probes.py checks that. A runtime that
 * returned straight to its caller lost them: only the first line got out, and no
 * atexit handler ran (FFmpeg and CPython each worked around it). */
#include <stdio.h>
#include <stdlib.h>

static void on_exit_handler(void) { puts("RETURN-FLUSH atexit handler ran"); }

int main(void)
{
	atexit(on_exit_handler);
	puts("RETURN-FLUSH line 1");
	puts("RETURN-FLUSH line 2");
	puts("RETURN-FLUSH line 3");
	return 5;
}
