/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <capstone/delegate.h>
#include <capstone/spawn.h>

extern char **environ;
extern void __capstone_vm_delegate_transport(void *, void *, size_t);
#ifdef SPAWN_DIRECT
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static void *scalar(unsigned long n)
{
    void *r;
    __asm__ volatile("movc %0, zero\n\tmv %0, %1" : "=&r"(r) : "r"(n));
    return r;
}
static void *dirty(unsigned long n)
{
    _Alignas(16) uint64_t words[2] = {n, 0xabadcafe};
    void *r;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(r) : "r"(words) : "memory");
    return r;
}
#endif

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "SPAWN_LIFECYCLE_FAIL line=%d errno=%d test=%s\n", __LINE__, errno, #x); } } while (0)
static void packet(char tag, pid_t pid)
{
    uint64_t p[2] = {(unsigned char)tag, (uint64_t)pid};
    CHECK(write(1, p, sizeof p) == sizeof p);
}
static void barrier(char tag)
{
    char c = 0;
    packet(tag, getpid());
    CHECK(read(0, &c, 1) == 1 && c == tag);
}
static pid_t start(const char *mode, const char *argument)
{
    pid_t pid = -1;
    char *args[] = {"probe.dom", (char *)mode, (char *)argument, NULL};
    int error = posix_spawn(&pid, "probe.dom", NULL, NULL, args, environ);
    CHECK(error == 0 && pid > 0);
    return error ? -1 : pid;
}
static long wire_spawn(void *meta, void *data, size_t bytes)
{
    struct capstone_delegate_entry *e = meta;
    memset(e, 0, sizeof *e);
    e->version = CAPSTONE_DELEGATE_VERSION;
    e->count = 1;
    e->nr = CAPSTONE_NR_SPAWN;
    e->args[0] = 16;
    e->args[1] = bytes;
    e->flags = 1;
    __capstone_vm_delegate_transport(meta, data, 8192);
    return e->result;
}
static long request(void *meta, void *data, size_t bytes)
{
#ifdef SPAWN_DIRECT
    (void)meta;
    return __capstone_native_syscall(CAPSTONE_NR_SPAWN, (char *)data + 16,
                                    scalar(bytes), 0, 0, 0, 0);
#else
    return wire_spawn(meta, data, bytes);
#endif
}
static size_t pack(void *block, const char *path, const char *mode)
{
    char *args[] = {"probe.dom", (char *)mode, NULL};
    char *env[] = {NULL};
    size_t bytes = 0;
    CHECK(capstone_spawn_pack(block, 4096, 0, 0, path, args, env, NULL, 0, NULL, &bytes) == 0);
    return bytes;
}
static void negative(void)
{
    char *meta = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *data = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(meta != MAP_FAILED && data != MAP_FAILED);
    if (meta == MAP_FAILED || data == MAP_FAILED) return;
    memset(meta, 0, 8192); memset(data, 0, 8192);
    char *block = data + 16;
    char saved[4096];
    size_t bytes = pack(block, "probe.dom", "exit");
    memcpy(saved, block, bytes);
    barrier('B'); /* supervisor samples the live task before all failures */
    /* Establish that the raw exchange request is valid before the fatal
     * metadata variant uses the same wire. Also qualify direct admission. */
    long child = wire_spawn(meta, data, bytes);
    int status = 0;
    CHECK(child > 0);
    if (child > 0) CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 17);
#ifdef SPAWN_DIRECT
    child = request(meta, data, bytes);
    CHECK(child > 0);
    if (child > 0) CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 17);
#endif
    for (unsigned kind = 0; kind < 8; ++kind) {
        struct capstone_spawn_header h;
        memcpy(block, saved, bytes);
        memcpy(&h, block, sizeof h);
        switch (kind) {
        case 0: h.magic[0] ^= 1; break;
        case 1: ++h.version; break;
        case 2: h.flags = 0x80000000u; break;
        case 3: h.bytes = CAPSTONE_SPAWN_BYTES + 1; break;
        case 4: h.path = 0; break;
        case 5: h.argc = CAPSTONE_SPAWN_STRINGS + 1; break;
        case 6: h.actions = CAPSTONE_SPAWN_ACTIONS + 1; break;
        case 7: h.argc = 0; break;
        }
        memcpy(block, &h, sizeof h);
        long rc = request(meta, data, bytes);
        CHECK(rc == -EINVAL);
        if (rc > 0) CHECK(waitpid(rc, NULL, 0) == rc);
    }
    memcpy(block, saved, bytes);
    uint32_t bad_offset = bytes;
    memcpy(block + sizeof(struct capstone_spawn_header), &bad_offset, sizeof bad_offset);
    CHECK(request(meta, data, bytes) == -EINVAL);
    memcpy(block, saved, bytes);
    block[bytes - 1] = 'x'; /* remove the final argument's terminator */
    CHECK(request(meta, data, bytes) == -EINVAL);
    memcpy(block, saved, bytes);
    unsigned long base = __builtin_capstone_cap_get_cursor(data);
    void *short_data = __builtin_capstone_cap_shrink(data, base, base + 16 + bytes - 1);
    void *write_only = __builtin_capstone_cap_tighten(data, 2);
    CHECK(request(meta, short_data, bytes) == -EFAULT);
    CHECK(request(meta, write_only, bytes) == -EFAULT);
#ifdef SPAWN_DIRECT
    CHECK(__capstone_native_syscall(CAPSTONE_NR_SPAWN, scalar(base + 16), scalar(bytes), 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(CAPSTONE_NR_SPAWN, block, dirty(bytes), 0, 0, 0, 0) == -EINVAL);
    CHECK(__capstone_native_syscall(CAPSTONE_NR_SPAWN, block, block, 0, 0, 0, 0) == -EINVAL);
#endif
    /* These requests create a real bootstrap child and fail its pinned exec.
     * Every failure must finish reaping before the service returns. */
    char *args[] = {"missing.dom", NULL};
    for (unsigned i = 0; i < 128; ++i) {
        pid_t pid = -777;
        CHECK(posix_spawn(&pid, "missing.dom", NULL, NULL, args, environ) == ENOENT);
        CHECK(pid == -777);
        CHECK(waitpid(-1, NULL, WNOHANG) == -1 && errno == ECHILD);
    }
    barrier('F');
    CHECK(munmap(meta, 8192) == 0 && munmap(data, 8192) == 0);
}
static void wait_any(void)
{
    /* The supervisor arranges a real, unregistered scalar child before exec.
     * It remains a Linux child, but this frontend must never reap it. */
    CHECK(waitpid(-1, NULL, WNOHANG) == -1 && errno == ECHILD);
    pid_t a = start("exit", "31"), b = start("exit", "32");
    unsigned seen = 0;
    for (unsigned i = 0; i < 2; ++i) {
        int status = 0;
        pid_t p = waitpid(-1, &status, 0);
        CHECK(WIFEXITED(status));
        if (p == a) { CHECK(!(seen & 1) && WEXITSTATUS(status) == 31); seen |= 1; }
        else if (p == b) { CHECK(!(seen & 2) && WEXITSTATUS(status) == 32); seen |= 2; }
        else CHECK(0);
    }
    CHECK(seen == 3);
    CHECK(waitpid(-1, NULL, WNOHANG) == -1 && errno == ECHILD);
}
static void autoreap(void)
{
    struct sigaction ignore = {.sa_handler = SIG_IGN}, saved;
    CHECK(sigemptyset(&ignore.sa_mask) == 0 && sigaction(SIGCHLD, &ignore, &saved) == 0);
    for (unsigned i = 0; i < 64; ++i) {
        pid_t pid = start("exit", "17");
        if (pid > 0) CHECK(waitpid(pid, NULL, 0) == -1 && errno == ECHILD);
    }
    CHECK(waitpid(-1, NULL, WNOHANG) == -1 && errno == ECHILD);
    /* More than the frontend's 64 bookkeeping slots, with no intervening
     * wait calls: detached ignored children must not consume slots forever. */
    for (unsigned i = 0; i < 100; ++i) {
        start("exit", "17");
        CHECK(sched_yield() == 0);
    }
    CHECK(waitpid(-1, NULL, 0) == -1 && errno == ECHILD);
    CHECK(sigaction(SIGCHLD, &saved, NULL) == 0);
    pid_t pid = start("exit", "29");
    int status = 0;
    CHECK(pid > 0 && waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 29);
}
static void bad_metadata(void)
{
    char *meta = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *data = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (meta == MAP_FAILED || data == MAP_FAILED) _exit(90);
    memset(meta, 0, 8192); memset(data, 0, 8192);
    size_t bytes = pack(data + 16, "probe.dom", "hold");
    barrier('M');
    if (mprotect(meta + 4096, 4096, PROT_READ)) _exit(91);
    /* Spawn succeeds, but signal-count publication on metadata page one
     * fails. The spawned child blocks on inherited stdin if it runs at all.
     * The supervisor must find no surviving/adopted child after our death.
     * This intentionally exercises the exchange route in BOTH SDK builds. */
    wire_spawn(meta, data, bytes);
    _exit(92);
}
int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    if (!strcmp(argv[1], "exit")) return argc > 2 ? atoi(argv[2]) : 17;
    if (!strcmp(argv[1], "hold")) {
        char c = 0;
        packet('H', getpid());
        return read(0, &c, 1) == 1 && c == 'E' ? 23 : 97;
    }
    if (!strcmp(argv[1], "negative")) negative();
    else if (!strcmp(argv[1], "wait-any")) wait_any();
    else if (!strcmp(argv[1], "autoreap")) autoreap();
    else if (!strcmp(argv[1], "bad-meta")) bad_metadata();
    else if (!strcmp(argv[1], "wait-parent")) {
        pid_t pid = start("hold", NULL);
        packet('P', pid);
        if (pid > 0) waitpid(pid, NULL, 0);
        CHECK(0); /* supervisor kills us after observing the blocked child */
    } else return 2;
    fprintf(stderr, "NATIVE_SPAWN_LIFECYCLE mode=%s checks=%d failures=%d\n", argv[1], checks, failures);
    return failures != 0;
}
