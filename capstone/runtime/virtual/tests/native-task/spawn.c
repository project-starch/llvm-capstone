/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "SPAWN_FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
static int child(void)
{
    char value[6] = {0};
    CHECK(getenv("NATIVE_CHILD") && !strcmp(getenv("NATIVE_CHILD"), "inherited"));
    CHECK(read(9, value, 5) == 5 && !strcmp(value, "hello"));
    CHECK(fcntl(9, F_GETFD) == 0);
    CHECK(chdir("..") == 0); /* inherited launch grant, not child cwd as root */
    CHECK(chdir("..") == -1 && errno == EACCES);
    CHECK(open("outside/marker", O_RDONLY) == -1);
    CHECK(write(1, "child-ok", 8) == 8);
    fprintf(stderr, "NATIVE_SPAWN_CHILD checks=%d failures=%d\n", checks, failures);
    return failures ? 99 : 37;
}
static void reap(pid_t pid, int code)
{
    int status = -1;
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == code);
    CHECK(waitpid(pid, &status, WNOHANG) == -1 && errno == ECHILD);
}
int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "child")) return child();
    if (argc > 1 && !strcmp(argv[1], "exit")) return 17;
    int input[2], output[2];
    pid_t pid = -1;
    posix_spawn_file_actions_t actions;
    char *args[] = {"spawn.dom", "child", NULL};
    char *env[] = {"NATIVE_CHILD=inherited", NULL};
    CHECK(pipe2(input, O_CLOEXEC) == 0 && pipe2(output, O_CLOEXEC) == 0);
    CHECK(write(input[1], "hello", 5) == 5);
    CHECK(posix_spawn_file_actions_init(&actions) == 0);
    CHECK(posix_spawn_file_actions_adddup2(&actions, input[0], 9) == 0);
    CHECK(posix_spawn_file_actions_adddup2(&actions, output[1], 1) == 0);
    CHECK(posix_spawn_file_actions_addclose(&actions, input[1]) == 0);
    CHECK(posix_spawn_file_actions_addchdir_np(&actions, "a") == 0);
    CHECK(posix_spawn(&pid, "../spawn.dom", &actions, NULL, args, env) == 0 && pid > 0);
    CHECK(posix_spawn_file_actions_destroy(&actions) == 0);
    CHECK(close(input[0]) == 0 && close(input[1]) == 0 && close(output[1]) == 0);
    char text[16] = {0};
    CHECK(read(output[0], text, sizeof text) == 8 && !strcmp(text, "child-ok"));
    CHECK(read(output[0], text, 1) == 0 && close(output[0]) == 0);
    if (pid > 0) reap(pid, 37);
    args[1] = "exit";
    for (int i = 0; i < 16; ++i) {
        CHECK(posix_spawn(&pid, "spawn.dom", NULL, NULL, args, environ) == 0);
        reap(pid, 17);
    }
    CHECK(posix_spawn(&pid, "missing.dom", NULL, NULL, args, environ) == ENOENT);
    CHECK(posix_spawn(&pid, "scalar", NULL, NULL, args, environ) == ENOEXEC);
    CHECK(posix_spawn(&pid, "script", NULL, NULL, args, environ) == ENOEXEC);
    CHECK(posix_spawn(&pid, "../outside.dom", NULL, NULL, args, environ) == EACCES);
    CHECK(posix_spawn(&pid, "escape", NULL, NULL, args, environ) != 0);
    CHECK(open("script-ran", O_RDONLY) == -1 && errno == ENOENT);
    CHECK(posix_spawn_file_actions_init(&actions) == 0);
    CHECK(posix_spawn_file_actions_adddup2(&actions, 999, 1) == 0);
    CHECK(posix_spawn(&pid, "spawn.dom", &actions, NULL, args, environ) == EBADF);
    CHECK(posix_spawn_file_actions_destroy(&actions) == 0);
    CHECK(posix_spawn(&pid, "spawn.dom", NULL, NULL, args, environ) == 0);
    reap(pid, 17);
    CHECK(waitpid(getpid(), NULL, WNOHANG) == -1 && errno == ECHILD);
    fprintf(stderr, "NATIVE_SPAWN checks=%d failures=%d\n", checks, failures);
    return failures != 0;
}
