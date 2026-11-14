#include <sys/mman.h>
#include <unistd.h>
int main(int argc, char **argv, char **envp)
{
    if (write(1, "ok", 2) != 2) return 1;
    char *p = mmap(0, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 2;
    *(volatile char *)p = 42;
    if (*(volatile char *)p != 42) return 3;
    if (munmap(p, 4096)) return 4;
    return 7;
}
