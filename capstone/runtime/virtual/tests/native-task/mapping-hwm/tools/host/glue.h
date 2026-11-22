/* host shim for plain_reuse.c: only what the file uses from mallocng's glue.h */
#include <stddef.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#define __syscall(n, a, b, c) syscall(n, a, b, c)
void *__plain_map(size_t); void __plain_unmap(void *, size_t);
void __plain_remapped(size_t, size_t); void __plain_remap_fit(size_t, size_t);
