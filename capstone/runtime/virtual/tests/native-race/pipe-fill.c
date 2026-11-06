/* SPDX-License-Identifier: MIT */
/* Supplier for the race experiment: a pipe preloaded with 64 KiB of 'Q'
 * becomes stdin of the program that is exec'd. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv)
{
    static char buffer[65536];
    int fds[2];
    memset(buffer, 'Q', sizeof(buffer));
    if (argc < 2 || pipe(fds) || fcntl(fds[1], F_SETPIPE_SZ, sizeof(buffer)) < (int)sizeof(buffer)) return 2;
    if (write(fds[1], buffer, sizeof(buffer)) != sizeof(buffer)) return 3;
    close(fds[1]);
    if (dup2(fds[0], 0) < 0) return 4;
    close(fds[0]);
    execvp(argv[1], argv + 1);
    perror("exec");
    return 5;
}
