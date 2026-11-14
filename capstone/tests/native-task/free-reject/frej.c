/* free()'s rejection of pointers it did not issue, recorded case by case so that a cheaper implementation of the
 * same allocator policy can be held to exactly today's outcomes (the allocator-lean work: alloc-lean branch).
 *
 * The parent spawns one child per case (a rejection kills its process) and prints each child's exit status or
 * terminating signal. Each child prints FREJ_SETUP <case> before the free under test and FREJ_RETURNED <case> if
 * free returned, so a case that dies in its setup is told apart from one that dies in free.
 *
 * Cases (the check each is expected to reach, from heap_sublet.h heap_access and the lifetime revoke):
 *   ok            control: free of a live, issued pointer returns
 *   int           an integer address (untagged): the probe load faults
 *   interior      cursor 16 bytes past the issued base: in bounds, so the probe passes; cursor != base traps
 *   pastend       cursor at the object's end: the probe is out of bounds
 *   global        a pointer to a global object: no mapping-table entry
 *   stack         a pointer to a stack object: no mapping-table entry
 *   double        the same pointer freed twice: revoked at the first free (its group is then released too)
 *   double-live   as double, with a sibling object kept live in the same group, so the group survives: with
 *                 lifetimes the pointer is revoked; without them (the spatial arm) the second free reaches get_meta,
 *                 whose freed-mask assert rejects it
 *   reused        a freed pointer after its slot was reallocated: revoked
 *   remapped      a large (own-mapping) object freed, the same size allocated again; if the new object has the same
 *                 address, the OLD pointer is freed: revoked (the case a last-mapping cache must not change)
 *   remapped-new  as remapped, but the NEW pointer is freed: returns */
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;
#define SELF "/tmp/nt/sr/.frej-native"
static char global_object[64];

static void setup(const char *c) { printf("FREJ_SETUP %s\n", c); fflush(stdout); }

static int child(const char *c)
{
    if (!strcmp(c, "ok")) { void *p = malloc(48); setup(c); free(p); }
    else if (!strcmp(c, "int")) { void *p = (void *)(uintptr_t)0x12345670; setup(c); free(p); }
    else if (!strcmp(c, "interior")) { char *p = malloc(48); setup(c); free(p + 16); }
    else if (!strcmp(c, "pastend")) { char *p = malloc(48); setup(c); free(p + 48); }
    else if (!strcmp(c, "global")) { setup(c); free(global_object); }
    else if (!strcmp(c, "stack")) { char s[64]; memset(s, 1, sizeof s); setup(c); free(s); }
    else if (!strcmp(c, "double")) { void *p = malloc(48); free(p); setup(c); free(p); }
    else if (!strcmp(c, "double-live")) {
        void *p = malloc(48), *q = malloc(48);
        memset(q, 4, 48); free(p);
        printf("FREJ_INFO double-live sibling-alive %d\n", ((char *)q)[0] == 4); setup(c); free(p);
    }
    else if (!strcmp(c, "reused")) {
        void *p = malloc(48); free(p);
        void *q = malloc(48);
        printf("FREJ_INFO reused same-address %d\n", (uintptr_t)p == (uintptr_t)q); setup(c); free(p);
    } else if (!strcmp(c, "remapped") || !strcmp(c, "remapped-new")) {
        size_t n = 1 << 20;
        char *p = malloc(n); memset(p, 2, n); free(p);
        char *q = malloc(n); memset(q, 3, n);
        printf("FREJ_INFO remapped same-address %d\n", (uintptr_t)p == (uintptr_t)q);
        setup(c); free(c[8] ? q : p);
    } else { printf("FREJ_UNKNOWN %s\n", c); return 2; }
    printf("FREJ_RETURNED %s\n", c); fflush(stdout);
    return 0;
}

int main(int argc, char **argv)
{
    static const char *cases[] = {"ok", "int", "interior", "pastend", "global", "stack", "double", "double-live", "reused",
                                  "remapped", "remapped-new"};
    if (argc > 2 && !strcmp(argv[1], "case")) return child(argv[2]);
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; ++i) {
        char *av[] = {SELF, "case", (char *)cases[i], NULL};
        pid_t pid; int st, rc = posix_spawn(&pid, SELF, NULL, NULL, av, environ);
        if (rc) { printf("FREJ_RESULT %s spawn-error %d\n", cases[i], rc); continue; }
        if (waitpid(pid, &st, 0) != pid) { printf("FREJ_RESULT %s wait-error\n", cases[i]); continue; }
        if (WIFEXITED(st)) printf("FREJ_RESULT %s exit %d\n", cases[i], WEXITSTATUS(st));
        else if (WIFSIGNALED(st)) printf("FREJ_RESULT %s signal %d\n", cases[i], WTERMSIG(st));
        else printf("FREJ_RESULT %s status 0x%x\n", cases[i], st);
        fflush(stdout);
    }
    printf("FREJ_DONE\n");
    return 0;
}
