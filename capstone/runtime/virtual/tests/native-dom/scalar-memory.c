#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Negative qualification: a domain must not read or write its scalar host
 * trampoline through Linux's process-memory file. Write back the same word
 * so the probe detects the authority without executing modified code. */
int main(void) {
    int maps = open("/proc/self/maps", O_RDONLY);
    char text[16384];
    unsigned long base = 0;
    long got = maps < 0 ? -1 : pread(maps, text, sizeof text - 1, 0);
    if (maps >= 0) close(maps);
    if (got > 0) {
        text[got] = 0;
        char *line = strtok(text, "\n");
        while (line) {
            if (strstr(line, "capstone-trampoline")) {
                sscanf(line, "%lx", &base);
                break;
            }
            line = strtok(NULL, "\n");
        }
    }
    int fd = open("/proc/self/mem", O_RDWR);
    int refused = fd < 0 && (errno == EACCES || errno == EPERM);
    uint32_t word = 0;
    long nr = -1, nw = -1;
    if (base && fd >= 0) {
        nr = pread(fd, &word, sizeof word, (off_t)base);
        if (nr == sizeof word) nw = pwrite(fd, &word, sizeof word, (off_t)base);
    }
    if (fd >= 0) close(fd);
    printf("SCALAR_MEMORY base_found=%d opened=%d read=%ld write=%ld\n",
           base != 0, fd >= 0, nr, nw);
    if (nr == sizeof word || nw == sizeof word) {
        puts("FAIL scalar host memory accessible from delegated I/O");
        return 1;
    }
    if (refused) {
        puts("PASS scalar memory open refused (other vectors require separate probes)");
        return 0;
    }
    puts("UNRESOLVED: access not reproduced; refusal alone is not a security proof");
    return 2;
}
