#include <capstone/delegate.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

static struct capstone_delegate_entry *meta;
static char *exchange;
static size_t exchange_bytes;
static int failures;
extern void __real___capstone_vm_delegate_transport(void *, void *, size_t);
void __wrap___capstone_vm_delegate_transport(void *m, void *x, size_t n)
{
    meta = m; exchange = x; exchange_bytes = n;
    __real___capstone_vm_delegate_transport(m, x, n);
}
static long raw(unsigned nr, const uint64_t args[6])
{
    struct capstone_delegate_entry saved = *meta;
    struct capstone_delegate_entry e = {.version = 2, .count = 1, .nr = nr};
    memcpy(e.args, args, sizeof e.args);
    capstone_delegate_pack(&e, nr, args);
    *meta = e;
    __real___capstone_vm_delegate_transport(meta, exchange, exchange_bytes);
    long result = meta->result;
    *meta = saved;
    return result;
}
static void check(const char *name, int ok)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}
static void denied_path(const char *path)
{
    errno = 0;
    int fd = open(path, O_RDWR);
    int error = errno;
    check(path, fd < 0 && (error == EACCES || error == EPERM || error == ELOOP || error == ENOTDIR));
    if (fd >= 0) close(fd);
}
int main(int argc, char **argv)
{
    char b[32] = {0};
    struct stat st;
    (void)getpid();
    if (!meta || exchange_bytes < 4096) return 2;
    if (argc == 2 && !strcmp(argv[1], "stdio")) {
        check("unsafe inherited stdio", pread(0, b, 1, 0) == -1 && errno == EBADF);
        return failures != 0;
    }
    const char *paths[] = {
        "/proc/self/mem", "/proc/self/maps", "/proc/self/pagemap",
        "/proc/self/fd/9", "/proc/self/cwd", "/proc/self/root", "/proc/self/exe",
        "/proc/thread-self/mem", "/dev/mem", "/dev/kmem", "/sys/kernel/uevent_helper",
        "../resource-outside", "nested/../../resource-outside",
        "mem-link", "proc-link/self/mem", "fd-link", "root-link/proc/self/mem",
        "proc-mount/self/mem", "memory-device", "null-device"
    };
    for (unsigned i = 0; i < sizeof paths / sizeof *paths; ++i) denied_path(paths[i]);
    check("outside metadata hidden", stat("/proc/self/maps", &st) == -1 && errno == ENOENT);
    check("inherited proc fd pread", pread(9, b, 1, 0) == -1 && errno == EBADF);
    check("inherited proc fd pwrite", pwrite(9, b, 1, 0) == -1 && errno == EBADF);
    check("inherited directory fd", openat(8, "self/mem", O_RDONLY) == -1 && errno == EBADF);
    check("inherited proc fd stat", fstat(9, &st) == -1 && errno == EBADF);
    check("inherited proc fd read", read(9, b, 1) == -1 && errno == EBADF);
    check("inherited proc fd write", write(9, b, 1) == -1 && errno == EBADF);
    struct iovec v = {b, 1};
    check("inherited proc fd readv", readv(9, &v, 1) == -1 && errno == EBADF);
    check("inherited proc fd writev", writev(9, &v, 1) == -1 && errno == EBADF);
    uint64_t send_args[6] = {1, 9, 0, 1};
    check("inherited proc fd sendfile", raw(CAPSTONE_SYS_sendfile, send_args) == -EBADF);
    uint64_t copy_args[6] = {9, 0, 1, 0, 1, 0};
    check("inherited proc fd copy_file_range", raw(CAPSTONE_SYS_copy_file_range, copy_args) == -EBADF);
    struct { unsigned nr; const char *name; } calls[] = {
        {270,"process_vm_readv"}, {271,"process_vm_writev"}, {117,"ptrace"},
        {226,"raw mprotect"}, {222,"raw mmap fixed"}, {215,"raw munmap"},
        {216,"raw mremap"}, {214,"raw brk"}, {233,"madvise"}, {167,"prctl"},
        {92,"personality"}, {221,"execve"}, {281,"execveat"},
        {425,"io_uring_setup"}, {426,"io_uring_enter"}, {427,"io_uring_register"},
        {282,"userfaultfd"}, {241,"perf_event_open"}, {280,"bpf"},
        {129,"kill self"}, {130,"tkill self"}, {131,"tgkill self"},
        {220,"clone fork"}, {435,"clone3"}, {437,"raw openat2"},
        {198,"socket"}, {199,"socketpair"}, {212,"recvmsg rights"},
        {36,"symlinkat"}, {37,"linkat"}, {33,"mknodat"}, {49,"chdir"},
        {50,"fchdir"}, {276,"renameat2"}, {134,"raw sigaction"},
        {139,"rt_sigreturn"}, {132,"sigaltstack"}, {103,"setitimer"},
        {23,"dup"}, {24,"dup3"}, {59,"pipe2"}, {279,"memfd_create"},
        {265,"open_by_handle_at"}, {438,"pidfd_getfd"}, {40,"mount"},
        {97,"unshare"}, {268,"setns"}
    };
    for (unsigned i = 0; i < sizeof calls / sizeof *calls; ++i) {
        uint64_t a[6] = {0};
        long result = raw(calls[i].nr, a);
        if (result != -ENOSYS) printf("nr=%u result=%ld\n", calls[i].nr, result);
        check(calls[i].name, result == -ENOSYS);
    }
    uint64_t a[6] = {1, 0x5412, 128}; /* TIOCSTI */
    check("arbitrary ioctl", raw(CAPSTONE_SYS_ioctl, a) == -ENOSYS);
    check("buffer ioctl", raw(CAPSTONE_NR_IOCTL_BUF, a) == -ENOSYS);
    a[1] = 8; /* F_SETOWN */
    check("fcntl signal owner", raw(CAPSTONE_SYS_fcntl, a) == -ENOSYS);
    a[1] = 4; a[2] = 020000; /* F_SETFL O_ASYNC */
    check("fcntl asynchronous signal", raw(CAPSTONE_SYS_fcntl, a) == -ENOSYS);
    int fd = open("ordinary", O_CREAT | O_TRUNC | O_RDWR, 0600);
    check("ordinary file open", fd >= 0);
    if (fd >= 0) {
        check("ordinary file write", write(fd, "good", 4) == 4);
        check("ordinary file pread", pread(fd, b, 4, 0) == 4 && !memcmp(b, "good", 4));
        check("ordinary path stat", !stat("ordinary", &st) && st.st_size == 4);
        check("ordinary file close", !close(fd));
        check("closed fd revoked", pread(fd, b, 1, 0) == -1 && errno == EBADF);
        check("ordinary unlink", !unlink("ordinary"));
    }
    int dir = open("nested", O_RDONLY | O_DIRECTORY);
    check("directory grant", dir >= 0);
    if (dir >= 0) {
        fd = openat(dir, "child", O_CREAT | O_RDWR, 0600);
        check("directory relative create", fd >= 0);
        if (fd >= 0) close(fd);
        check("directory parent escape", openat(dir, "../ordinary", O_RDONLY) == -1 && errno == EACCES);
        close(dir);
    }
    extern long __capstone_vm_protect(uintptr_t, size_t, int);
    uintptr_t host = argc > 1 ? strtoull(argv[1], NULL, 0) : 0;
    check("grant protect host", host && __capstone_vm_protect(host, 4096, PROT_WRITE) == -EINVAL);
    extern long __capstone_vm_unmap(uintptr_t, size_t);
    check("grant unmap host", host && __capstone_vm_unmap(host, 4096) == -EINVAL);
    printf("resource failures=%d\n", failures);
    return failures != 0;
}
