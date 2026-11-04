/* Native guest observer: an invalid domain exchange must neither execute the
 * read nor consume bytes. The parent retains the pipe's read endpoint across
 * the refused launcher process and checks the whole payload afterwards. */
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    assert(argc == 4); /* launcher, domain, case */
    int p[2], status;
    char data[256], back[257];
    memset(data, 0x5c, sizeof data);
    assert(!pipe(p) && write(p[1], data, sizeof data) == sizeof data);
    close(p[1]);
    pid_t child = fork(); assert(child >= 0);
    if (!child) {
        assert(dup2(p[0], 0) == 0); close(p[0]);
        execl(argv[1], argv[1], argv[2], argv[3], "stdin", (char *)0);
        perror("exec launcher"); _exit(127);
    }
    assert(waitpid(child, &status, 0) == child);
    ssize_t n = read(p[0], back, sizeof back); close(p[0]);
    int pass = WIFEXITED(status) && WEXITSTATUS(status) == 125 &&
        n == sizeof data && !memcmp(data, back, sizeof data);
    printf("IOV1 observe case=%s status=%d pipe_bytes=%ld %s\n", argv[3], status,
           (long)n, pass ? "PASS" : "FAIL");
    return !pass;
}
