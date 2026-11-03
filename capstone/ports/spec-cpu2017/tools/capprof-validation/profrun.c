/* profrun PROGRAM ARGS...: start a capprof window, run the program as a child,
 * stop the window when it has exited, exit with its status. Linked away from
 * the usual static base so its instructions never share an address with the
 * program's. */
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* After the window: the child's page faults and peak resident size (wait4's rusage), one line
 * on stderr, so an arm's page count can be read beside its attempts. Written outside the window. */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: profrun PROGRAM ARGS...\n"); return 125; }
    int status;
    struct rusage ru;
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    pid_t pid = fork();
    if (pid == 0) { execvp(argv[1], argv + 1); _exit(127); }
    while (pid > 0 && wait4(pid, &status, 0, &ru) < 0) {}
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    if (pid < 0) return 126;
    fprintf(stderr, "PROFRUN_RUSAGE minflt=%ld majflt=%ld maxrss_kb=%ld nvcsw=%ld nivcsw=%ld\n",
            ru.ru_minflt, ru.ru_majflt, ru.ru_maxrss, ru.ru_nvcsw, ru.ru_nivcsw);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
