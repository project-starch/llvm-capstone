/* Guest-side runner for the staged mallocbench gate (native Linux, not a Capstone program):
 *   mb-run SECONDS PROGRAM ARGS...
 * Runs PROGRAM (capstone-vexec) as a child with the caller's stdin/stdout/stderr, kills it
 * after SECONDS, and prints on stderr after it ends
 *   MB_RUSAGE status=<exit status or 128+signal> killed=<0|1> maxrss_kib= utime_ms= stime_ms=
 * maxrss is the launcher process's peak resident set: the capability image, its stack and
 * mappings, and the native mallocng heap with its metadata all live in that one mm, so this
 * is the counterpart of CheriBSD's time -l maximum resident set size. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t child;
static volatile sig_atomic_t killed;
static void expire(int sig) { (void)sig; killed = 1; kill(child, SIGKILL); }

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: mb-run SECONDS PROGRAM ARGS...\n"); return 2; }
    unsigned limit = (unsigned)strtoul(argv[1], NULL, 10);
    child = fork();
    if (child < 0) { perror("fork"); return 2; }
    if (child == 0) { execv(argv[2], argv + 2); perror("execv"); _exit(127); }
    signal(SIGALRM, expire);
    alarm(limit);
    int status;
    struct rusage ru;
    while (wait4(child, &status, 0, &ru) < 0) {}
    alarm(0);
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    fprintf(stderr, "MB_RUSAGE status=%d killed=%d maxrss_kib=%ld utime_ms=%ld stime_ms=%ld\n", rc,
            (int)killed, ru.ru_maxrss, ru.ru_utime.tv_sec * 1000 + ru.ru_utime.tv_usec / 1000,
            ru.ru_stime.tv_sec * 1000 + ru.ru_stime.tv_usec / 1000);
    return rc;
}
