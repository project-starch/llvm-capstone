/* Per-call I/O attempts: one capprof window and one LQ line per invocation.
 * Run with the linux-micro run-micro.sh driver. Compare two round counts to
 * remove fixed setup costs. All I/O uses the unchanged libc transport. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __CAPSTONE__
static const char *arm = "capstone";
#else
static const char *arm = "plain";
#endif

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#ifndef F_SETPIPE_SZ
#define F_SETPIPE_SZ 1031
#endif

static void die(const char *what) { perror(what); exit(2); }


static long do_read(int fd, char *buf, size_t n)
{
    long r = read(fd, buf, n);
    return r < 0 ? -errno : r;
}
static long do_write(int fd, char *buf, size_t n)
{
    long r = write(fd, buf, n);
    return r < 0 ? -errno : r;
}

static void fill(int fd, const char *buf, size_t n, long rounds)
{
    for (long i = 0; i < rounds; i++)
        for (size_t done = 0; done < n;) {
            long w = write(fd, buf + done, n - done);
            if (w <= 0) die("fill");
            done += (size_t)w;
        }
}

int main(int argc, char **argv)
{
    const char *m = argc > 1 ? argv[1] : "";
    if (!strcmp(m, "noop")) return 0;
    if (!strcmp(m, "close")) { STOP(); return 0; }
    if (argc < 4) {
        fprintf(stderr, "usage: inline-io-micro pipe|file|pipew N R | noop | close\n");
        return 125;
    }
    size_t n = strtoul(argv[2], 0, 0);
    long rounds = strtol(argv[3], 0, 0);
    size_t total = n * (size_t)rounds;
    char *buf = malloc(n + 64), *src = malloc(n + 64);
    if (!buf || !src) die("malloc");
    for (size_t i = 0; i < n; i++) src[i] = (char)(i * 7 + 1);
    memset(buf, 0, n);   /* every page of the buffer admitted before the window */
    long bad = 0, got = 0;
    if (!strcmp(m, "pipe") || !strcmp(m, "pipew")) {
        int p[2];
        if (pipe(p)) die("pipe");
        size_t cap = 65536;
        while (cap < total) cap <<= 1;
        if (fcntl(p[1], F_SETPIPE_SZ, (int)cap) < 0) die("F_SETPIPE_SZ");
        if (!strcmp(m, "pipe")) {
            fill(p[1], src, n, rounds);
            START();
            for (long i = 0; i < rounds; i++) { long r = do_read(p[0], buf, n); got += r; if (r != (long)n) bad++; }
            STOP();
        } else {
            START();
            for (long i = 0; i < rounds; i++) { long r = do_write(p[1], src, n); got += r; if (r != (long)n) bad++; }
            STOP();
            for (long i = 0; i < rounds; i++) if (read(p[0], buf, n) != (long)n) bad++;
        }
        if (memcmp(buf, src, n)) bad++;
    } else if (!strcmp(m, "file")) {
        int fd = open("/tmp/inline-io-micro.dat", O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) die("open");
        fill(fd, src, n, rounds);
        if (lseek(fd, 0, SEEK_SET) != 0) die("lseek");
        START();
        for (long i = 0; i < rounds; i++) { long r = do_read(fd, buf, n); got += r; if (r != (long)n) bad++; }
        STOP();
        if (memcmp(buf, src, n)) bad++;
        close(fd);
        unlink("/tmp/inline-io-micro.dat");
    } else {
        fprintf(stderr, "unknown mode %s\n", m);
        return 125;
    }
    printf("LQ %s %s n=%zu rounds=%ld bytes=%ld bad=%ld\n", arm, m, n, rounds, got, bad);
    return bad ? 1 : 0;
}
