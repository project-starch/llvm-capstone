#include <errno.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

/* Scalar supervisor; the selected shell image itself enters native binfmt. */
int main(int argc, char **argv)
{
    int status;
    pid_t pid;
    if (argc != 4) {
        fprintf(stderr, "usage: driver SHELL_IMAGE SCRIPT CHILD_IMAGE\n");
        return 2;
    }
    pid = fork();
    if (pid < 0) return 3;
    if (!pid) {
        char *args[] = {"sh", argv[2], argv[3], NULL};
        char *env[] = {"PATH=.", "LC_ALL=C", NULL};
        execve(argv[1], args, env);
        perror("shell execve");
        _exit(127);
    }
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return 4;
    }
    printf("HUSH driver status=%d\n", status);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
