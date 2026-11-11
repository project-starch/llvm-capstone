/* SPDX-License-Identifier: MIT */
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char **argv)
{
    int p[2], status;
    if (argc != 2 || pipe(p) || fcntl(p[0], F_SETFL, O_NONBLOCK) || write(p[1], "ABC", 3) != 3) return 2;
    alarm(15);
    pid_t pid = fork();
    if (!pid) {
        if (dup2(p[0], 0) < 0) _exit(3);
        close(p[0]); close(p[1]);
        execl(argv[1], argv[1], "pipe", (char *)0); _exit(4);
    }
    close(p[0]); /* Keep writer open: empty pipe must yield EAGAIN, not EOF. */
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 5;
    close(p[1]);
    return !WIFEXITED(status) || WEXITSTATUS(status);
}
