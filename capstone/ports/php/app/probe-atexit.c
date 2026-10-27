/* Does an atexit handler run on this profile, and which stream still works when it does?
 *
 * The seam's revoke counters are the measurement's strongest gate -- stock 0, block 0, object
 * more than 0 -- and on real PHP there is nowhere to print them from except an exit hook,
 * because PHP owns main(). The first attempt registered one through atexit() and produced
 * NOTHING: the image references atexit (llvm-nm says `U atexit` in the object), the seam is
 * demonstrably live in the same run (it printed its mode line and the platform reported
 * collections=3 reclaimed=28255 against the control's 0/0), and still not one line appeared.
 *
 * Two candidate explanations, and they call for opposite fixes, so guessing is not allowed:
 *   (a) the handler never runs -- the process leaves through a path that skips them;
 *   (b) the handler runs but its output is lost -- PHP's shutdown has already taken stdout
 *       down by then, which would make stderr the right stream and stdout the wrong one.
 *
 * This probe separates them. It writes to stdout, to stderr, and straight to fd 2 with
 * write(2), from a handler and from a destructor, and prints a marker just before returning
 * from main so that "nothing at all" and "nothing after main" are distinguishable. The raw
 * write() matters: it is the one channel that does not depend on stdio still being up.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void raw(const char *s) { (void)!write(2, s, strlen(s)); }

static void on_exit_first(void)
{
    raw("ATEXIT_PROBE handler-1 raw-write-fd2\n");
    fprintf(stderr, "ATEXIT_PROBE handler-1 stderr\n");
    fflush(stderr);
    printf("ATEXIT_PROBE handler-1 stdout\n");
    fflush(stdout);
}

/* Registered second, so it runs FIRST: a handler list that runs partially is its own finding. */
static void on_exit_second(void) { raw("ATEXIT_PROBE handler-2 raw-write-fd2\n"); }

__attribute__((destructor))
static void on_destructor(void) { raw("ATEXIT_PROBE destructor raw-write-fd2\n"); }

int main(void)
{
    int rc1 = atexit(on_exit_first);
    int rc2 = atexit(on_exit_second);
    printf("ATEXIT_PROBE registered rc=%d,%d\n", rc1, rc2);
    fflush(stdout);
    raw("ATEXIT_PROBE main returning\n");
    return 0;
}
