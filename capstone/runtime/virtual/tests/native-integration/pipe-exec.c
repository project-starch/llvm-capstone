/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv)
{
    static char buffer[65536];
    int fds[2];
    if (argc < 2 || pipe(fds) || fcntl(fds[1], F_SETPIPE_SZ, sizeof(buffer)) < sizeof(buffer)) return 2;
    if (write(fds[1], buffer, sizeof(buffer)) != sizeof(buffer)) return 3;
    close(fds[1]);
    if (dup2(fds[0], 0) < 0) return 4;
    close(fds[0]);
    execvp(argv[1], argv + 1);
    perror("exec");
    return 5;
}
