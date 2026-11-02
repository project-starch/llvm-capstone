/* profrun PROGRAM ARGS...: start a capprof window, run the program as a child,
 * stop the window when it has exited, exit with its status. Linked away from
 * the usual static base so its instructions never share an address with the
 * program's. */
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: profrun PROGRAM ARGS...\n"); return 125; }
    int status;
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    pid_t pid = fork();
    if (pid == 0) { execvp(argv[1], argv + 1); _exit(127); }
    while (pid > 0 && waitpid(pid, &status, 0) < 0) {}
    asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    if (pid < 0) return 126;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
