/* memfd-launch MODE LAUNCHER IMAGE: run LAUNCHER on a memfd copy of IMAGE
 * ("child" argument), keeping the memfd's writable descriptor open meanwhile.
 *   open    no seals: anyone with that descriptor can still write the image
 *   write   F_SEAL_WRITE only: the image can still grow or shrink
 *   sealed  F_SEAL_WRITE, F_SEAL_GROW and F_SEAL_SHRINK: the image is fixed
 * The launcher must refuse the first two (ETXTBSY) and run the third. A native
 * helper for run-exec.py; exits with the launcher's status. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: memfd-launch open|write|sealed LAUNCHER IMAGE\n"); return 2; }
    int seals = !strcmp(argv[1], "open") ? 0 : !strcmp(argv[1], "write") ? F_SEAL_WRITE :
                !strcmp(argv[1], "sealed") ? F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK : -1;
    int src = open(argv[3], O_RDONLY), fd = memfd_create("image", MFD_ALLOW_SEALING);
    if (seals < 0 || src < 0 || fd < 0) { perror("memfd-launch"); return 2; }
    char buf[65536];
    ssize_t n;
    while ((n = read(src, buf, sizeof buf)) > 0)
        if (write(fd, buf, n) != n) { perror("memfd-launch: write"); return 2; }
    if (n < 0 || (seals && fcntl(fd, F_ADD_SEALS, seals))) { perror("memfd-launch: seal"); return 2; }
    char path[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    pid_t pid = fork();
    if (pid == 0) {
        execl(argv[2], argv[2], path, "child", (char *)NULL);
        _exit(127);
    }
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 2;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
