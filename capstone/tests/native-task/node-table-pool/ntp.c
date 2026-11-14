/* The node-table pool's directed test (capstone_vm node_table_pool): one context leaves its flat
 * lifetime table dirty -- a long free list (table[2]), an allocation count (table[3]) and a bump
 * pointer far above the floor -- and exits; the next context, created right after, must get a
 * table that behaves as a fresh one.
 *
 * Modes (argv[1]; anything else is the parent, so the program runs as a SPEC stage's binary):
 *   dirty N   allocate N 32-byte objects (each a lifetime record: CDERIVE at malloc), free every
 *             other one (CREVOKE: N/2 records on the free list), exit without freeing the rest;
 *   fresh N   allocate N objects, write and check a pattern through each, free them all, three times.
 *   parent    ROUNDS times: spawn dirty, then fresh, as separate processes (one context each).
 * A stale free list makes the fresh child's context_create refuse (node_available() reads table[2])
 * or its first allocation fault (cause 29), so its spawn or its exit fails. A stale record the new
 * context never names is invisible from here: the module's node_table_pool_verify scan, read back by
 * the stage, is the check for those. */
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;
#define ROUNDS 4
#define OBJECTS 6000
/* The stage copies the binary here, inside dom_grant_root (as the perl stages copy perl to
 * /tmp/nt/sr/.perl-native): a native spawn cannot name argv[0]'s /tmp/nt/b, which is outside it. */
#define SELF "/tmp/nt/sr/.ntp-native"

static int dirty(long n)
{
    void **p = malloc(n * sizeof(*p));
    if (!p) return 3;
    for (long i = 0; i < n; ++i)
        if (!(p[i] = malloc(32))) return 4;
    for (long i = 0; i < n; i += 2) free(p[i]);
    printf("NTP_DIRTY allocated %ld freed %ld\n", n, (n + 1) / 2);
    fflush(stdout);
    return 0;   /* the rest stay allocated: the table is released at exit with them live */
}

static int fresh(long n)
{
    void **p = malloc(n * sizeof(*p));
    if (!p) return 3;
    for (int round = 0; round < 3; ++round) {
        for (long i = 0; i < n; ++i) {
            unsigned char *q = malloc(32);
            if (!q) return 4;
            memset(q, (int)(i ^ round) & 0xff, 32);
            p[i] = q;
        }
        for (long i = 0; i < n; ++i) {
            unsigned char *q = p[i];
            for (int j = 0; j < 32; ++j)
                if (q[j] != ((i ^ round) & 0xff)) return 5;
        }
        for (long i = 0; i < n; ++i) free(p[i]);
    }
    printf("NTP_FRESH ok %ld x 3\n", n);
    fflush(stdout);
    return 0;
}

static int spawn(const char *self, const char *mode, const char *count)
{
    char *argv[] = {(char *)self, (char *)mode, (char *)count, NULL};
    pid_t pid;
    int status, rc = posix_spawn(&pid, self, NULL, NULL, argv, environ);
    if (rc) return 1000 + rc;
    if (waitpid(pid, &status, 0) != pid) return 2000;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 3000 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

int main(int argc, char **argv)
{
    char count[24];
    int failed = 0;
    if (argc > 2 && !strcmp(argv[1], "dirty")) return dirty(atol(argv[2]));
    if (argc > 2 && !strcmp(argv[1], "fresh")) return fresh(atol(argv[2]));
    snprintf(count, sizeof count, "%d", OBJECTS);
    for (int round = 0; round < ROUNDS; ++round) {
        int d = spawn(SELF, "dirty", count);
        int f = spawn(SELF, "fresh", count);
        printf("NTP_ROUND %d dirty=%d fresh=%d\n", round, d, f);
        failed |= d || f;
    }
    printf("NTP_DONE %s rounds %d objects %d\n", failed ? "FAIL" : "PASS", ROUNDS, OBJECTS);
    return failed;
}
