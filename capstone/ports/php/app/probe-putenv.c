/* Does this libc keep an empty-name putenv string in environ?
 *
 * WHY THIS PROBE EXISTS. CRASH-084 (bug #67399) is a use-after-free that needs putenv("=") to
 * leave its string referenced from libc's environ: PHP frees that string on the second
 * empty-name call and the following environ scan reads it. That is what the corpus's native
 * oracle (glibc, ASan) saw. musl 1.2.5's putenv (src/env/putenv.c) takes a different branch for
 * an empty name:
 *
 *     size_t l = __strchrnul(s, '=') - s;
 *     if (!l || !s[l]) return unsetenv(s);
 *
 * and unsetenv refuses a name that contains '=' with EINVAL, so nothing is stored. On this libc
 * the dangling reference the defect needs does not exist, and a run that completes is not a
 * miss of the protection: it is a case whose precondition this platform does not meet.
 *
 * A declaration like that must be CHECKED, not asserted: this probe makes the two calls with
 * buffers that stay alive, scans environ for the exact pointer it supplied, and prints what it
 * found with the return value and errno. The control proves the probe can see retention at all:
 * an ordinary named variable put the same way MUST be found, or the probe's negative means
 * nothing. The runner reads PUTENV_PROBE lines; one of three outcomes, never a silent zero:
 *
 *   PUTENV_PROBE empty-name retained=0 ...   and   control retained=1   -> precondition absent
 *   PUTENV_PROBE empty-name retained=1 ...                              -> precondition present
 *   anything else (control not retained, no DONE line)                  -> the probe failed
 *
 * No capability builtin, no allocator of PHP's: the question is the libc's alone. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

static char empty_name[] = "=";            /* what CRASH-084's first putenv supplies */
static char empty_value[] = "=foo";        /* and its second */
static char control[] = "PHPPROBE_CTL=1";  /* an ordinary variable, put the same way */

static int retained(const char *s)
{
    char **e;
    for (e = environ; e && *e; e++) {
        if (*e == s) { return 1; }
    }
    return 0;
}

static void put(const char *label, char *s)
{
    int rc, err;
    errno = 0;
    rc = putenv(s);
    err = errno;
    printf("PUTENV_PROBE %s retained=%d rc=%d errno=%d (%s)\n", label, retained(s), rc, err,
           err ? strerror(err) : "ok");
    fflush(stdout);
}

int main(void)
{
    int failures = 0;
    put("control", control);
    if (!retained(control)) {
        printf("PUTENV_PROBE FAIL the control variable was not retained; the probe cannot see "
               "retention, so its reading of the empty name means nothing\n");
        failures++;
    }
    put("empty-name", empty_name);
    put("empty-name-value", empty_value);
    printf("PUTENV_PROBE DONE failures=%d\n", failures);
    fflush(stdout);
    return failures ? 1 : 0;
}
