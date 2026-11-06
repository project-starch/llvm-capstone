#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Ordinary executable used by shell tests; no workload-specific commands. */
int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    if (!strcmp(argv[1], "exit")) return argc == 3 ? atoi(argv[2]) : 2;
    if (!strcmp(argv[1], "args")) {
        for (int i = 2; i < argc; i++) printf("[%s]\n", argv[i]);
        return 0;
    }
    if (!strcmp(argv[1], "env")) {
        const char *value = argc == 3 ? getenv(argv[2]) : NULL;
        if (!value) return 1;
        puts(value);
        return 0;
    }
    if (!strcmp(argv[1], "fds")) {
        for (int i = 0; i < 64; i++)
            if (fcntl(i, F_GETFD) >= 0) printf("%d ", i);
        putchar('\n');
        return 0;
    }
    if (!strcmp(argv[1], "emit") && argc == 3) {
        char buffer[4096];
        unsigned long left = strtoul(argv[2], NULL, 10);
        memset(buffer, 'A', sizeof(buffer));
        while (left) {
            size_t want = left < sizeof(buffer) ? left : sizeof(buffer);
            ssize_t n = write(1, buffer, want);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return 3;
            left -= n;
        }
        return 0;
    }
    if (!strcmp(argv[1], "count")) {
        char buffer[4096];
        unsigned long count = 0;
        ssize_t n;
        while ((n = read(0, buffer, sizeof(buffer))) != 0) {
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) return 4;
            count += n;
        }
        printf("%lu\n", count);
        return 0;
    }
    return 2;
}
