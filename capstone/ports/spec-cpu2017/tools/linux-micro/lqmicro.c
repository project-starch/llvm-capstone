/* lqmicro: per-page and per-process cost microbenchmarks for the guest-Linux question
 * ("does a tag-aware guest Linux pay off"), built twice from this one source: with the
 * Capstone SDK (a virtual-profile domain) and with the plain rv64 compiler.
 *
 * Each mode does its setup, opens ONE capprof window (the plugin's start/stop markers, as
 * tools/capprof-validation/vt.c), runs the measured phase, closes the window and prints one
 * "LQ" result line with the unit counts, so attempts per unit = window attempts / units.
 * Everything in the window is counted system-wide (program, launcher, module, kernel).
 *
 *   touch MIB              map MIB MiB, store one word per 4 KiB page, unmap      unit: page
 *   heap ROUNDS BATCH      ROUNDS x (BATCH mallocs of a fixed gcc-like size mix, first and
 *                          last byte written, all freed)                          unit: allocation
 *   fork K ROUNDS          K pages mapped and touched before the window; per round a fork and
 *                          K stores. Capstone: sched_yield, which the experiment launcher
 *                          turns into a fork of a sleeping launcher child when
 *                          CAPSTONE_VM_TORTURE_FORK is set (a domain cannot fork), then the
 *                          domain stores into the K pages (copy-on-write in the parent).
 *                          Plain: the same shape (fork a child that sleeps, store K pages in
 *                          the parent); with LQ_FORK_CHILD=1 instead the child stores K pages
 *                          and exits and the parent waits                         unit: fork, page
 *   exec N PATH            N x (posix_spawn PATH noop, waitpid)                   unit: process
 *   compact CHUNKS DO      CHUNKS x 4 MiB mapped and touched before the window, every other
 *                          chunk unmapped (fragmentation); in the window: DO=1 writes 1 to
 *                          /proc/sys/vm/compact_memory, then one store per remaining page
 *                          (DO=0 is the same without the compaction, the control)
 *                          unit: migrated page (/proc/vmstat pgmigrate_success delta)
 *   noop                   exit 0 at once (the exec mode's child)
 *   close                  the stop marker alone: run-micro.sh runs it (plain build) after a
 *                          failed invocation, whose window would otherwise stay open, so the
 *                          next invocation's start does not continue it
 */
#include <fcntl.h>
#include <sched.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define PAGE 4096UL

extern char **environ;

#ifdef __CAPSTONE__
static const char *arm = "capstone";
#else
static const char *arm = "plain";
#endif

static void die(const char *what) { perror(what); exit(2); }

static char *map(size_t bytes) {
    char *p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap");
    return p;
}

__attribute__((noinline)) static void store_pages(char *p, size_t pages, long v) {
    for (size_t i = 0; i < pages; i++)
        *(volatile long *)(p + i * PAGE) = v + (long)i;
}

/* /proc/vmstat counters, read outside the window */
static long vmstat(const char *name) {
    FILE *f = fopen("/proc/vmstat", "r");
    if (!f) return -1;
    char key[128]; long v, r = -1;
    while (fscanf(f, "%127s %ld", key, &v) == 2)
        if (!strcmp(key, name)) { r = v; break; }
    fclose(f);
    return r;
}

static long param(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static int mode_touch(long mib) {
    size_t bytes = (size_t)mib << 20, pages = bytes / PAGE;
    START();
    char *p = map(bytes);
    store_pages(p, pages, 1);
    munmap(p, bytes);
    STOP();
    printf("LQ %s touch mib=%ld pages=%zu\n", arm, mib, pages);
    return 0;
}

/* A fixed gcc-like size mix from an LCG (the same sequence in both arms): mostly small
 * objects, a quarter 256 B-4 KiB, an eighth 4-32 KiB, a few 32-128 KiB, so mallocng's groups
 * for the larger classes are separate mappings (8-64 KiB and up) that empty and are released
 * every round. */
static unsigned long lcg = 12345;
static size_t next_size(void) {
    lcg = lcg * 6364136223846793005UL + 1442695040888963407UL;
    unsigned r = (unsigned)(lcg >> 33), pick = r % 100, span = r >> 8;
    if (pick < 60) return 16 + span % 240;
    if (pick < 85) return 256 + span % 3840;
    if (pick < 97) return 4096 + span % 28672;
    return 32768 + span % 98304;
}

static int mode_heap(long rounds, long batch) {
    char **obj = calloc((size_t)batch, sizeof *obj);
    size_t *len = calloc((size_t)batch, sizeof *len);
    if (!obj || !len) die("calloc");
    size_t bytes = 0;
    START();
    for (long r = 0; r < rounds; r++) {
        for (long i = 0; i < batch; i++) {
            len[i] = next_size();
            obj[i] = malloc(len[i]);
            if (!obj[i]) die("malloc");
            obj[i][0] = (char)i; obj[i][len[i] - 1] = (char)r;
            bytes += len[i];
        }
        for (long i = 0; i < batch; i++) free(obj[i]);
    }
    STOP();
    printf("LQ %s heap rounds=%ld batch=%ld allocations=%ld bytes=%zu\n", arm, rounds, batch,
           rounds * batch, bytes);
    return 0;
}

static int mode_fork(long k, long rounds) {
    size_t bytes = (size_t)(k > 0 ? k : 1) * PAGE;
    char *p = map(bytes);
    store_pages(p, (size_t)k, 7);
    int child_stores = getenv("LQ_FORK_CHILD") && !strcmp(getenv("LQ_FORK_CHILD"), "1");
    START();
    for (long r = 0; r < rounds; r++) {
#ifdef __CAPSTONE__
        sched_yield();                      /* the experiment launcher forks here */
        store_pages(p, (size_t)k, r);
#else
        pid_t pid = fork();
        if (pid < 0) die("fork");
        if (pid == 0) {
            if (child_stores) { store_pages(p, (size_t)k, r); _exit(0); }
            struct timespec t = {3, 0};
            nanosleep(&t, 0);
            _exit(0);
        }
        if (child_stores) {
            int st; waitpid(pid, &st, 0);
        } else {
            store_pages(p, (size_t)k, r);
        }
#endif
    }
    STOP();
    printf("LQ %s fork k=%ld rounds=%ld pages=%ld child_stores=%d\n", arm, k, rounds, k * rounds,
           child_stores);
    return 0;
}

static int mode_exec(long n, const char *path) {
    char *argv[] = {(char *)path, "noop", 0};
    long bad = 0;
    START();
    for (long i = 0; i < n; i++) {
        pid_t pid;
        if (posix_spawn(&pid, path, 0, 0, argv, environ)) die("posix_spawn");
        int st;
        if (waitpid(pid, &st, 0) != pid || !WIFEXITED(st) || WEXITSTATUS(st)) bad++;
    }
    STOP();
    printf("LQ %s exec n=%ld processes=%ld bad=%ld\n", arm, n, n, bad);
    return bad ? 1 : 0;
}

static int mode_compact(long chunks, long do_compact) {
    const size_t chunk = 4UL << 20, cpages = chunk / PAGE;
    char **c = calloc((size_t)chunks, sizeof *c);
    if (!c) die("calloc");
    for (long i = 0; i < chunks; i++) { c[i] = map(chunk); store_pages(c[i], cpages, i); }
    /* the helper that fragmented guest memory (hog) is told to exit now, so the holes it
     * leaves around the domain's pages are free when compaction runs */
    int fd = open("/tmp/lq-hog-exit", O_WRONLY | O_CREAT, 0644);
    if (fd >= 0) close(fd);
    struct timespec t = {2, 0};
    nanosleep(&t, 0);
    for (long i = 0; i < chunks; i += 2) munmap(c[i], chunk);
    long m0 = vmstat("pgmigrate_success"), s0 = vmstat("compact_migrate_scanned");
    long cp0 = param("/sys/module/captag/parameters/copies");
    START();
    if (do_compact) {
        int cf = open("/proc/sys/vm/compact_memory", O_WRONLY);
        if (cf < 0 || write(cf, "1\n", 2) != 2) die("compact_memory");
        close(cf);
    }
    for (long i = 1; i < chunks; i += 2) store_pages(c[i], cpages, 100 + i);
    STOP();
    long m1 = vmstat("pgmigrate_success"), s1 = vmstat("compact_migrate_scanned");
    long cp1 = param("/sys/module/captag/parameters/copies");
    printf("LQ %s compact chunks=%ld do=%ld retouched_pages=%zu migrated=%ld scanned=%ld captag_copies=%ld\n",
           arm, chunks, do_compact, (size_t)(chunks / 2) * cpages, m1 - m0, s1 - s0,
           cp0 >= 0 && cp1 >= 0 ? cp1 - cp0 : -1);
    return 0;
}

int main(int argc, char **argv) {
    const char *m = argc > 1 ? argv[1] : "";
    long a = argc > 2 ? strtol(argv[2], 0, 0) : 0, b = argc > 3 ? strtol(argv[3], 0, 0) : 0;
    if (!strcmp(m, "noop")) return 0;
    if (!strcmp(m, "close")) { STOP(); return 0; }
    if (!strcmp(m, "touch") && argc == 3) return mode_touch(a);
    if (!strcmp(m, "heap") && argc == 4) return mode_heap(a, b);
    if (!strcmp(m, "fork") && argc == 4) return mode_fork(a, b);
    if (!strcmp(m, "exec") && argc == 4) return mode_exec(a, argv[3]);
    if (!strcmp(m, "compact") && argc == 4) return mode_compact(a, b);
    fprintf(stderr, "usage: lqmicro touch MIB | heap ROUNDS BATCH | fork K ROUNDS | exec N PATH | "
                    "compact CHUNKS DO | noop\n");
    return 125;
}
