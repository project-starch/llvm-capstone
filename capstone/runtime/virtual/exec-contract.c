/* Exec in place and the image's write denial, from inside a virtual domain.
 *
 *   exec-contract.dom exec IMAGE     re-executes IMAGE with "child"; prints the child's line
 *   exec-contract.dom refused PATH   execve of a file that is not an image must fail with an
 *                                    errno and leave this process running
 *   exec-contract.dom hold           stays alive, until its stdin ends, while the gate tries
 *                                    to write its image
 *   exec-contract.dom child          what the re-executed image prints
 *
 * The launcher denies writers to a running image (ETXTBSY, as Linux does for a running
 * executable); the gate script checks that from outside, with this image holding. */
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "child")) {
        printf("EXEC_CONTRACT_CHILD pid=%d\n", (int)getpid());
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "exec")) {
        printf("EXEC_CONTRACT_PARENT pid=%d\n", (int)getpid());
        fflush(stdout);
        char *child[] = {argv[2], "child", NULL};
        execve(argv[2], child, environ);
        int error = errno;
        printf("EXEC_CONTRACT_FAIL exec errno=%d\n", error);
        return 1;
    }
    if (argc == 3 && !strcmp(argv[1], "refused")) {
        char *child[] = {argv[2], "child", NULL};
        int rc = execve(argv[2], child, environ);
        int error = errno;
        /* Still this process: the refusal came back as an errno. */
        printf("EXEC_CONTRACT_REFUSED rc=%d errno=%d pid=%d\n", rc, error, (int)getpid());
        return rc == -1 && error ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "hold")) {
        char c;
        ssize_t n;
        printf("EXEC_CONTRACT_HOLDING\n");
        fflush(stdout);
        /* The gate releases the hold by closing the other end of stdin. */
        while ((n = read(0, &c, 1)) > 0)
            ;
        if (n < 0) {
            printf("EXEC_CONTRACT_FAIL read errno=%d\n", errno);
            return 1;
        }
        printf("EXEC_CONTRACT_HELD\n");
        return 0;
    }
    printf("EXEC_CONTRACT_USAGE\n");
    return 2;
}
