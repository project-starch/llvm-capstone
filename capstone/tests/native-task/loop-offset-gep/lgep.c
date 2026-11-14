/* -lsr-capability-byte-iv-fallback's directed test: loops of the shapes LSR's capability address groups take, run
 * with the cursor at the extremes of a 24 MiB heap object. Built twice, with the fallback off and on; both builds must
 * print the same LGEP_RESULT lines. With the fallback, fwd, bwd, shifted, field and idx32 take the byte form and
 * negidx keeps the capability recurrence (the compiler's group log says so for this source). On the native twin
 * (compact bounds) a CINCOFFSET whose result is outside the source's representable window writes an untagged
 * integer (R-60), so an out-of-window intermediate capability faults at its next use: a codegen that steps or
 * hoists a capability out of window dies here rather than reading as variance.
 *
 * The parent spawns one child per case and prints its checksum line and exit status. Cases:
 *   fwd       lsr4.c's loop (S_regcppop): a[i].start conditional, a[i].end always, i over the whole array
 *   bwd       a[i].tmp, i from the last element down to 0
 *   shifted   a[i - K].tmp with K a large constant: the case where a hoisted base would be base - K * 24
 *   negidx    p = a + N (one past the end), p[-i].start for i = 1..N
 *   field     an array member at a constant index of each element: s[i].v[3]
 *   idx32     a signed 32-bit index, sign-extended to the GEP's width
 *   ctl-out   POSITIVE CONTROL: a capability moved 256 MiB below its base and back, then stored through: must die
 *             (cause 24 at the store) on compact bounds, or the test cannot see an out-of-window value
 *   ctl-in    its pair: the same round trip by 12 MiB, inside the object: must return */
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;
#define SELF "/tmp/nt/sr/.lgep-native"
#define N (1L << 20)
#define K 3000000L
#define REPS 6

typedef struct { long start, end, tmp; } pair;
typedef struct { long x; int v[4]; } rec;

__attribute__((noinline)) void fwd(pair *a, unsigned long lo, unsigned long hi, unsigned long m)
{
    for (unsigned long i = lo; i <= hi; i++) {
        if (i > m)
            a[i].start = (long)i;
        a[i].end = ~(long)i;
    }
}
__attribute__((noinline)) void bwd(pair *a, long hi)
{
    for (long i = hi; i >= 0; i--)
        a[i].tmp += i;
}
__attribute__((noinline)) void shifted(pair *a, long n)
{
    for (long i = K; i < K + n; i++)
        a[i - K].tmp ^= i;
}
__attribute__((noinline)) void negidx(pair *end, long n)
{
    for (long i = 1; i <= n; i++)
        end[-i].start += i;
}
__attribute__((noinline)) void field(rec *s, long n)
{
    for (long i = 0; i < n; i++)
        s[i].v[3] += (int)i;
}
__attribute__((noinline)) void idx32(pair *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i].end -= i;
}

static unsigned long sum(const long *w, size_t n)
{
    unsigned long h = 1469598103934665603UL;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (unsigned long)w[i]) * 1099511628211UL;
    return h;
}

static int child(const char *c)
{
    size_t bytes = N * sizeof(pair);
    pair *a = malloc(bytes);
    if (!a) { printf("LGEP_RESULT %s malloc-failed\n", c); return 3; }
    memset(a, 0, bytes);
    for (int r = 0; r < REPS; r++) {
        if (!strcmp(c, "fwd")) fwd(a, 0, N - 1, N / 2);
        else if (!strcmp(c, "bwd")) bwd(a, N - 1);
        else if (!strcmp(c, "shifted")) shifted(a, N);
        else if (!strcmp(c, "negidx")) negidx(a + N, N);
        else if (!strcmp(c, "field")) field((rec *)a, (long)(bytes / sizeof(rec)));
        else if (!strcmp(c, "idx32")) idx32(a, (int)N);
        else if (!strcmp(c, "ctl-out") || !strcmp(c, "ctl-in")) {
            long d = c[4] == 'o' ? 256L << 20 : 12L << 20;
            char *p = (char *)a;
            p -= d;
            __asm__ volatile("" : "+r"(p));
            p += d;
            __asm__ volatile("" : "+r"(p));
            *(long *)p = r;
        } else { printf("LGEP_UNKNOWN %s\n", c); return 2; }
    }
    printf("LGEP_RESULT %s sum %016lx\n", c, sum((const long *)a, bytes / sizeof(long)));
    return 0;
}

int main(int argc, char **argv)
{
    static const char *cases[] = {"fwd", "bwd", "shifted", "negidx", "field", "idx32", "ctl-in", "ctl-out"};
    if (argc > 2 && !strcmp(argv[1], "case")) return child(argv[2]);
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        char *av[] = {SELF, "case", (char *)cases[i], NULL};
        pid_t pid; int st, rc = posix_spawn(&pid, SELF, NULL, NULL, av, environ);
        if (rc) { printf("LGEP_STATUS %s spawn-error %d\n", cases[i], rc); continue; }
        fflush(stdout);
        if (waitpid(pid, &st, 0) != pid) { printf("LGEP_STATUS %s wait-error\n", cases[i]); continue; }
        if (WIFEXITED(st)) printf("LGEP_STATUS %s exit %d\n", cases[i], WEXITSTATUS(st));
        else if (WIFSIGNALED(st)) printf("LGEP_STATUS %s signal %d\n", cases[i], WTERMSIG(st));
        fflush(stdout);
    }
    printf("LGEP_DONE\n");
    return 0;
}
