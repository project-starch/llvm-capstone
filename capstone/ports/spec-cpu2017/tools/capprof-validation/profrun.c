/* profrun PROGRAM ARGS...: start a capprof window, run the program as a child,
 * stop the window when it has exited, exit with its status. Linked away from
 * the usual static base so its instructions never share an address with the
 * program's. */
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* After the window: the child's page faults and peak resident size (wait4's rusage), one line
 * on stderr, so an arm's page count can be read beside its attempts. Written outside the window. */

/* PAGE-VOLUME CENSUS (PROFRUN_SMAPS=<file>): the child's /proc/<pid>/smaps at the moment it
 * exits, appended to that file. The child is traced (PTRACE_TRACEME before the exec) with
 * PTRACE_O_TRACEEXIT, so it stops inside exit_group with its address space still intact: what is
 * resident then is what the program held at its end, VMA by VMA, the same instant the Capstone
 * launcher's own dump (CAPSTONE_VM_SMAPS) takes. The ptrace stops add instructions inside the
 * window, so a run with the variable set is for pages, not attempts. Inert without the variable:
 * the fork/exec/wait4 path is unchanged. The error of this picture: pages faulted and unmapped
 * before the exit (a freed mmap, a munmap'd file) are counted by minflt but not here. */
static void put_all(int fd, const char *s, size_t n) {
    while (n > 0) { ssize_t w = write(fd, s, n); if (w <= 0) return; s += w; n -= (size_t)w; }
}
static void copy_file(int out, const char *name) {
    char buf[4096];
    int in = open(name, O_RDONLY | O_CLOEXEC);
    if (in < 0) return;
    ssize_t r;
    while ((r = read(in, buf, sizeof buf)) > 0) put_all(out, buf, (size_t)r);
    close(in);
}
/* For every VMA of the child (/proc/<pid>/maps), which of its pages are present in its page
 * table (/proc/<pid>/pagemap bit 63), as a hex bitmap: page i = bit i, least significant bit of
 * each nibble first. Ties a resident page to the section or symbol it holds. */
static void put_present_maps(int out, pid_t pid) {
    char name[64], line[512];
    static char text[1 << 16];
    static uint64_t entry[512];
    snprintf(name, sizeof name, "/proc/%d/maps", (int)pid);
    int in = open(name, O_RDONLY | O_CLOEXEC);
    if (in < 0) return;
    ssize_t got = 0, r;
    while (got < (ssize_t)sizeof text - 1 && (r = read(in, text + got, sizeof text - 1 - got)) > 0) got += r;
    close(in);
    text[got] = 0;
    snprintf(name, sizeof name, "/proc/%d/pagemap", (int)pid);
    int pm = open(name, O_RDONLY | O_CLOEXEC);
    if (pm < 0) return;
    for (char *p = text; *p;) {
        char *end = strchr(p, '\n');
        if (!end) break;
        *end = 0;
        unsigned long lo, hi;
        if (sscanf(p, "%lx-%lx", &lo, &hi) == 2) {
            size_t pages = (hi - lo) / 4096, page = lo / 4096;
            int n = snprintf(line, sizeof line, "PROFRUN_PRESENT base=0x%lx pages=%zu present=", lo, pages);
            put_all(out, line, (size_t)n);
            for (size_t at = 0; at < pages; at += 512) {
                size_t span = pages - at < 512 ? pages - at : 512;
                if (lseek(pm, (off_t)((page + at) * 8), SEEK_SET) < 0 ||
                    read(pm, entry, span * 8) != (ssize_t)(span * 8)) break;
                char hex[128 + 1]; size_t k = 0;
                for (size_t i = 0; i < span; i += 4) {
                    unsigned v = 0;
                    for (size_t j = 0; j < 4 && i + j < span; ++j) v |= (unsigned)(entry[i + j] >> 63) << j;
                    hex[k++] = "0123456789abcdef"[v];
                }
                put_all(out, hex, k);
            }
            put_all(out, "\n", 1);
        }
        p = end + 1;
    }
    close(pm);
}
static void dump_smaps(const char *path, pid_t pid) {
    char name[64], buf[256];
    int out = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (out < 0) return;
    int n = snprintf(buf, sizeof buf, "PROFRUN_SMAPS phase=exit pid=%d\n", (int)pid);
    put_all(out, buf, (size_t)n);
    /* The child's own minor-fault count so far, from its /proc stat (field 10 after the command). */
    snprintf(name, sizeof name, "/proc/%d/stat", (int)pid);
    put_all(out, "PROFRUN_STAT ", 13);
    copy_file(out, name);
    put_present_maps(out, pid);
    put_all(out, "PROFRUN_SMAPS_BEGIN\n", 20);
    snprintf(name, sizeof name, "/proc/%d/smaps", (int)pid);
    copy_file(out, name);
    put_all(out, "PROFRUN_SMAPS_END\n", 18);
    close(out);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: profrun PROGRAM ARGS...\n"); return 125; }
    const char *smaps = getenv("PROFRUN_SMAPS");
    if (smaps && !*smaps) smaps = NULL;
    int status;
    struct rusage ru;
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    pid_t pid = fork();
    if (pid == 0) {
        if (smaps && ptrace(PTRACE_TRACEME, 0, NULL, NULL)) _exit(126);
        execvp(argv[1], argv + 1); _exit(127);
    }
    if (smaps && pid > 0) {
        /* The exec stop first; then run the child to its exit stop, delivering its other signals. */
        int traced = 0;
        for (;;) {
            if (wait4(pid, &status, 0, &ru) < 0) continue;
            if (!WIFSTOPPED(status)) break;
            int sig = WSTOPSIG(status);
            if (!traced && sig == SIGTRAP) {
                traced = 1;
                ptrace(PTRACE_SETOPTIONS, pid, NULL, (void *)(long)PTRACE_O_TRACEEXIT);
                sig = 0;
            } else if (status >> 8 == (SIGTRAP | (PTRACE_EVENT_EXIT << 8))) {
                dump_smaps(smaps, pid);
                sig = 0;
            }
            ptrace(PTRACE_CONT, pid, NULL, (void *)(long)sig);
        }
    } else {
        while (pid > 0 && wait4(pid, &status, 0, &ru) < 0) {}
    }
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    if (pid < 0) return 126;
    fprintf(stderr, "PROFRUN_RUSAGE minflt=%ld majflt=%ld maxrss_kb=%ld nvcsw=%ld nivcsw=%ld\n",
            ru.ru_minflt, ru.ru_majflt, ru.ru_maxrss, ru.ru_nvcsw, ru.ru_nivcsw);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
