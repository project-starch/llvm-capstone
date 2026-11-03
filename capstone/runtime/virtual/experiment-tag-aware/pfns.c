/* EXPERIMENT (lane M): physical frames of a range of another process, from /proc/PID/pagemap (root).
 * usage: pfns PID ADDR LEN STATEFILE   (ADDR/LEN hex or decimal)
 * Prints "present=N moved=M new=K" against the PFNs saved in STATEFILE by the previous call,
 * then saves the current ones there. moved = present both times at a different frame. */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: pfns PID ADDR LEN STATEFILE\n"); return 2; }
    char path[64];
    snprintf(path, sizeof path, "/proc/%s/pagemap", argv[1]);
    uint64_t addr = strtoull(argv[2], 0, 0), len = strtoull(argv[3], 0, 0);
    size_t n = (len + 4095) / 4096;
    uint64_t *cur = calloc(n, 8), *old = calloc(n, 8);
    int fd = open(path, O_RDONLY);
    if (fd < 0 || !cur || !old) { perror("pagemap"); return 1; }
    for (size_t i = 0; i < n; i++) {
        uint64_t e = 0;
        if (pread(fd, &e, 8, (off_t)(((addr >> 12) + i) * 8)) != 8) { perror("pread"); return 1; }
        cur[i] = (e >> 63) & 1 ? (e & ((1ull << 55) - 1)) : 0;
    }
    close(fd);
    FILE *f = fopen(argv[4], "rb");
    int had = f && fread(old, 8, n, f) == n;
    if (f) fclose(f);
    long present = 0, moved = 0, fresh = 0;
    for (size_t i = 0; i < n; i++) {
        if (!cur[i]) continue;
        present++;
        if (had && old[i] && old[i] != cur[i]) moved++;
        if (had && !old[i]) fresh++;
    }
    f = fopen(argv[4], "wb");
    if (!f || fwrite(cur, 8, n, f) != n) { perror("state"); return 1; }
    fclose(f);
    printf("present=%ld moved=%ld new=%ld\n", present, moved, fresh);
    return 0;
}
