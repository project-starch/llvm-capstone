/* Static Linux process startup for the capability pointer ABI. */
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <capstone/linux-start.h>
#include "libc.h"
#include "syscall.h"

extern int main(int, char **, char **);
extern int __libc_start_main(int (*)(int, char **, char **), int, char **,
                            void (*)(), void (*)(), void (*)());
void __capstone_init_globals(void);
void *__capstone_aux_random;
static size_t scalar_auxv[2 * 38];

_Noreturn void __capstone_linux_start(void *stack)
{
    size_t argc = *(size_t *)stack;
    char **argv = (char **)((char *)stack + 16);
    __capstone_init_globals();
    if (argc > 0x7fffffff) {
        __syscall(SYS_exit_group, 127);
        __builtin_trap();
    }
    __libc_start_main(main, argc, argv, 0, 0, 0);
    __builtin_trap();
}

/* Called by musl's __init_libc. Keep a scalar auxv view for legacy queries,
 * and preserve reference-valued entries separately for runtime consumers. */
void __capstone_linux_init(char **envp, char *pn)
{
    char **end = envp;
    size_t aux[38] = {0};
    char *execfn = 0;
    __environ = envp;
    while (*end) ++end;
    const struct capstone_linux_auxv *v = (const void *)(end + 1);
    size_t used = 0;
    for (; v->type; ++v) {
        if (v->type >= 38) continue;
        void *value = v->value.pointer;
        size_t scalar = __builtin_capstone_cap_get_tag(value)
            ? __builtin_capstone_cap_get_cursor(value) : v->value.scalar;
        aux[v->type] = scalar;
        if (used + 2 < sizeof scalar_auxv / sizeof *scalar_auxv) {
            scalar_auxv[used++] = v->type;
            scalar_auxv[used++] = scalar;
        }
        if (v->type == AT_RANDOM) __capstone_aux_random = v->value.pointer;
        if (v->type == AT_EXECFN) execfn = v->value.pointer;
    }
    libc.auxv = scalar_auxv;
    __hwcap = aux[AT_HWCAP];
    libc.page_size = aux[AT_PAGESZ];
    if (libc.page_size != 4096) {
        __syscall(SYS_exit_group, 127);
        __builtin_trap();
    }
    if (!pn) pn = execfn;
    if (!pn) pn = "";
    __progname = __progname_full = pn;
    for (size_t i = 0; pn[i]; ++i)
        if (pn[i] == '/') __progname = pn + i + 1;
    __init_tls(aux);
    __init_ssp(__capstone_aux_random);
    if (aux[AT_UID] == aux[AT_EUID] && aux[AT_GID] == aux[AT_EGID]
        && !aux[AT_SECURE]) return;
    struct pollfd pfd[3] = {{.fd=0}, {.fd=1}, {.fd=2}};
    if (__syscall(SYS_ppoll, pfd, 3, &(struct timespec){0}, 0, 8) < 0)
        __builtin_trap();
    for (size_t i = 0; i < 3; ++i)
        if ((pfd[i].revents & POLLNVAL) &&
            __syscall(SYS_openat, AT_FDCWD, "/dev/null", O_RDWR, 0) < 0)
            __builtin_trap();
    libc.secure = 1;
}

/* The compiler emits untagged link addresses in the init/fini arrays.
 * Recover their image-relative address before calling through a reference. */
extern const unsigned char __init_array_start[], __init_array_end[];
extern const unsigned char __fini_array_start[], __fini_array_end[];
void __capstone_array_anchor(void) {}
extern const unsigned long __capstone_array_anchor_link;
__asm__(".section .rodata\n.balign 8\n"
        ".global __capstone_array_anchor_link\n"
        "__capstone_array_anchor_link: .quad __capstone_array_anchor\n.previous\n");
static void call_array(const unsigned char *p)
{
    void (*fn)(void) = *(void (*const *)(void))p;
    if (!__builtin_capstone_cap_get_tag(fn)) {
        unsigned long address = *(const unsigned long *)p;
        fn = (void (*)(void))((char *)__capstone_array_anchor +
            (long)(address - __capstone_array_anchor_link));
    }
    fn();
}
void __libc_start_init(void)
{
    for (const unsigned char *p = __init_array_start; p < __init_array_end;
         p += sizeof(void (*)(void))) call_array(p);
}
void __libc_exit_fini(void)
{
    for (const unsigned char *p = __fini_array_end; p > __fini_array_start;
         p -= sizeof(void (*)(void))) call_array(p - sizeof(void (*)(void)));
}
