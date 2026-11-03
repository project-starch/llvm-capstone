/* capprof alias test (plain RISC-V Linux program): one physical code page,
 * mapped executable at two virtual addresses, run N1 times through the first
 * and N2 times through the second inside one window. The page holds
 *   addi a0, a0, 0x3e5   (0x3e550513)
 *   ret                  (0x00008067)
 * With CF_PCREL a block translated at one alias also runs at the other. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")

int main(void) {
    int fd = memfd_create("alias", 0);
    unsigned code[2] = { 0x3e550513u, 0x00008067u };
    if (fd < 0 || ftruncate(fd, 4096) || pwrite(fd, code, sizeof code, 0) != sizeof code) return 1;
    long (*f1)(long) = mmap(NULL, 4096, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
    long (*f2)(long) = mmap(NULL, 4096, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
    if (f1 == MAP_FAILED || f2 == MAP_FAILED || f1 == f2) return 1;
    __builtin___clear_cache((char *)f1, (char *)f1 + 8);
    long a = 0;
    START();
    for (int i = 0; i < 3000; i++) a = f1(a);
    for (int i = 0; i < 1000; i++) a = f2(a);
    STOP();
    printf("f1=%p f2=%p a=%ld\n", (void *)f1, (void *)f2, a);
    return 0;
}
