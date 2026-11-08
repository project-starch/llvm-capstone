/* SPDX-License-Identifier: MIT */
/* Identical mapping geometry; no malloc in the measured access loop. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#define PAGES 256
#define PAGE 4096
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const char *mode = argv[1];
    int read_first = !strcmp(mode, "read") || !strcmp(mode, "read-write") || !strcmp(mode, "tagged");
    int write_pages = !strcmp(mode, "write") || !strcmp(mode, "read-write");
    volatile unsigned char *p = mmap(0, PAGES*PAGE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 3;
    if (!strcmp(mode, "denied")) {
        if (mprotect((void *)p, PAGES*PAGE, PROT_READ)) return 4;
        if (write(1, "ARMED_DENIED\n", 13) != 13) return 5;
        p[0] = 1;
        return 91;
    }
    if (!strcmp(mode, "stale")) {
        if (munmap((void *)p, PAGES*PAGE)) return 4;
        if (write(1, "ARMED_STALE\n", 12) != 12) return 5;
        return p[0] ? 92 : 93;
    }
    if (!read_first && !write_pages) return 6;
    if (read_first)
        for (unsigned i=0; i<PAGES; ++i) if (p[i*PAGE] != 0) return 10;
    if (write_pages) {
        for (unsigned i=0; i<PAGES; ++i) p[i*PAGE] = (i%251)+1;
        for (unsigned i=PAGES; i--;) if (p[i*PAGE] != (i%251)+1) return 11;
    }
    if (!strcmp(mode, "tagged")) {
        for (unsigned i=0; i<PAGES; ++i) {
            volatile unsigned char * volatile *slot = (void *)(p+i*PAGE);
            p[i*PAGE+128] = (i%251)+1;
            *slot = p+i*PAGE+128;
        }
        for (unsigned i=PAGES; i--;) {
            volatile unsigned char * volatile *slot = (void *)(p+i*PAGE);
#ifdef CAPSTONE_PROBE
            if (!__builtin_capstone_cap_get_tag((void *)*slot)) return 12;
#endif
            if (**slot != (i%251)+1) return 13;
        }
    }
    if (munmap((void *)p, PAGES*PAGE)) return 14;
    printf("MEMORY_ACCESS_PASS:%s\n", mode);
    return 0;
}
