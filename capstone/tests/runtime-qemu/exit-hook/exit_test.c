/* exit() from main in a musl domain (C-56).
 *
 * Built twice by run.sh. exit-default defines no __capstone_at_exit, so the
 * runtime's own weak default runs; before C-56 the runtime only DECLARED it
 * weak, `if (__capstone_at_exit)` was true for the undefined symbol in a
 * domain, and exit() called the image base (cause 2). exit-hook defines the
 * hook and turns 7 into 42, which shows the hook path runs and its result is
 * what the host reads. The line before exit() is buffered (musl buffers stdout
 * fully in a domain); seeing it shows exit() flushed stdio. The atexit handler's
 * line shows exit() called it through a pointer that kept its capability: musl's
 * own atexit() passes the handler through uintptr_t and loses the tag, which
 * runtime/atexit_capability_safe.c replaces.
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef WITH_HOOK
int __capstone_at_exit(int status) { return status + 35; }
#endif

static void on_exit_handler(void) { puts("EXIT-TEST atexit handler ran"); }

int main(void)
{
	atexit(on_exit_handler);
	printf("EXIT-TEST before exit\n");
	exit(7);
}
