/* Native guest observer: an invalid domain exchange must neither execute the
 * read nor consume bytes. The parent retains the pipe's read endpoint across
 * the refused launcher process and checks the whole payload afterwards.
 * Also drives the deterministic remap-kill contract against /dev/capstone-vm. */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../wire.h"
#include "../../include/capstone/delegate.h"

static long pin_balance(void)
{
    int rfd = open("/proc/sys/vm/stat_refresh", O_WRONLY);
    if (rfd >= 0) { (void)!write(rfd, "1\n", 2); close(rfd); }
    FILE *f = fopen("/proc/vmstat", "r");
    assert(f);
    char key[64];
    unsigned long val = 0, acq = 0, rel = 0;
    while (fscanf(f, "%63s %lu", key, &val) == 2) {
        if (!strcmp(key, "nr_foll_pin_acquired")) acq = val;
        else if (!strcmp(key, "nr_foll_pin_released")) rel = val;
    }
    fclose(f);
    return (long)(acq - rel);
}

static void *map_anon(size_t bytes, int prot)
{
    void *p = mmap(NULL, bytes, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    (void)madvise(p, bytes, MADV_NOHUGEPAGE);
    return p;
}

struct step_ctx {
    int dev, rc, err;
    struct cv_step step;
    atomic_int done;
};

static void *step_runner(void *arg)
{
    struct step_ctx *c = arg;
    c->step = (struct cv_step){.thread = 0};
    c->rc = ioctl(c->dev, CV_STEP, &c->step);
    c->err = errno;
    atomic_store(&c->done, 1);
    return NULL;
}

static void remap_kill_child(int mode, int ready_fd)
{
    int dev = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    assert(dev >= 0);
    int p[2];
    assert(!pipe(p));
    void *code = map_anon(4096, PROT_READ | PROT_WRITE | PROT_EXEC);
    void *stack = map_anon(4096, PROT_READ | PROT_WRITE);
    void *meta = map_anon(CAPSTONE_DELEGATE_META_BYTES, PROT_READ | PROT_WRITE);
    void *exch = map_anon(4096, PROT_READ | PROT_WRITE);
    void *remap_buf = map_anon(8192, PROT_READ | PROT_WRITE);
    uint32_t *insns = code;
    insns[0] = 0x00001637u; /* lui a2, 1 (4096) */
    insns[1] = 0x00a00893u; /* li a7, 10 (CV_SERVICE_THREAD_DELEGATE) */
    insns[2] = 0x00000073u; /* ecall */
    insns[3] = 0x00100073u; /* ebreak */
    __builtin___clear_cache((char *)code, (char *)code + 16);
    memset(stack, 0, 4096);
    memset(meta, 0, CAPSTONE_DELEGATE_META_BYTES);
    ((volatile char *)remap_buf)[0] = 1;
    /* Leave exch untouched before CV_ADD so io_admit must pin the cold page. */
    struct capstone_delegate_entry *e = meta;
    e->version = CAPSTONE_DELEGATE_VERSION;
    e->flags = 2;
    e->count = 1;
    e->nr = CAPSTONE_SYS_read;
    e->args[0] = (uint64_t)p[0];
    e->args[1] = 0;
    e->args[2] = 1;
    e->result = -ENOSYS;

    struct cv_map cm = {.address = (uintptr_t)code, .bytes = 4096, .permissions = 7,
                        .reg = 3, .cursor = (uintptr_t)code};
    struct cv_map sm = {.address = (uintptr_t)stack, .bytes = 4096, .permissions = 6,
                        .reg = 2, .cursor = (uintptr_t)stack + 4096};
    struct cv_map mm = {.address = (uintptr_t)meta, .bytes = CAPSTONE_DELEGATE_META_BYTES,
                        .permissions = 6, .reg = 10, .cursor = (uintptr_t)meta};
    struct cv_map xm = {.address = (uintptr_t)exch, .bytes = 4096, .permissions = 6,
                        .reg = 11, .cursor = (uintptr_t)exch, .linear = 1};
    struct cv_map rm = {.address = (uintptr_t)remap_buf, .bytes = 4096, .permissions = 6,
                        .reg = 13, .cursor = (uintptr_t)remap_buf, .linear = 1};
    assert(!ioctl(dev, CV_ADD, &cm));
    assert(!ioctl(dev, CV_ADD, &sm));
    assert(!ioctl(dev, CV_ADD, &mm));
    assert(!ioctl(dev, CV_ADD, &xm));
    assert(!ioctl(dev, CV_ADD, &rm));

    int has_inline = access("/sys/module/capstone_vm/parameters/inline_served", F_OK) == 0;
    if (has_inline) {
        struct cv_io_private priv = {0};
        struct cv_io_transport tr = {
            .thread = 0, .meta = (uintptr_t)meta, .exchange = (uintptr_t)exch, .bytes = 4096
        };
        assert(!ioctl(dev, CV_IO_PRIVATE, &priv));
        assert(!ioctl(dev, CV_IO_TRANSPORT, &tr));
        struct cv_stats st0, st1;
        assert(!ioctl(dev, CV_STATS, &st0));
        struct step_ctx ctx = {.dev = dev};
        pthread_t th;
        assert(!pthread_create(&th, NULL, step_runner, &ctx));
        int held = 0;
        for (int i = 0; i < 500; ++i) {
            struct cv_remap probe = {.thread = 0, .id = xm.id, .bytes = 8192};
            if (ioctl(dev, CV_REMAP_BEGIN, &probe) == -1 && errno == EBUSY) {
                held = 1;
                break;
            }
            usleep(2000);
        }
        assert(held && !atomic_load(&ctx.done));
        assert(!ioctl(dev, CV_STATS, &st1));
        assert(st1.pinned_pages == st0.pinned_pages + 1);
        struct cv_remap remap = {.thread = 0, .id = rm.id, .bytes = 8192};
        assert(!ioctl(dev, CV_REMAP_BEGIN, &remap));
        if (mode == 0) {
            assert(write(p[1], "Q", 1) == 1);
            usleep(30000);
            /* The domain must never resume while c->remapping is open. */
            assert(!atomic_load(&ctx.done));
            remap.address = 0;
            assert(!ioctl(dev, CV_REMAP_END, &remap));
            assert(!pthread_join(th, NULL));
            assert(!ctx.rc && ctx.step.kind == 2 && e->result == 1 && ((char *)exch)[0] == 'Q');
            _exit(0);
        }
        if (mode == 2) {
            assert(write(p[1], "Q", 1) == 1);
            usleep(30000);
            assert(!atomic_load(&ctx.done));
        }
        assert(write(ready_fd, "R", 1) == 1);
        for (;;) pause();
    } else {
        struct cv_step step = {.thread = 0};
        assert(!ioctl(dev, CV_STEP, &step) && step.kind == 3);
        struct cv_remap remap = {.thread = 0, .id = rm.id, .bytes = 8192};
        assert(!ioctl(dev, CV_REMAP_BEGIN, &remap));
        if (mode == 0) {
            remap.address = 0;
            assert(!ioctl(dev, CV_REMAP_END, &remap));
            _exit(0);
        }
        assert(write(ready_fd, "R", 1) == 1);
        for (;;) pause();
    }
}

static int remap_kill(void)
{
    long pins0 = pin_balance();
    int status = 0, live_ok = 0, reaped = 0;
    pid_t c0 = fork();
    assert(c0 >= 0);
    if (!c0) remap_kill_child(0, -1);
    for (int i = 0; i < 300; ++i) {
        if (waitpid(c0, &status, WNOHANG) == c0) {
            live_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
            break;
        }
        usleep(10000);
    }
    if (live_ok) {
        for (int mode = 1; mode <= 2; ++mode) {
            int ready[2];
            char ch = 0;
            assert(!pipe(ready));
            pid_t child = fork();
            assert(child >= 0);
            if (!child) {
                close(ready[0]);
                remap_kill_child(mode, ready[1]);
            }
            close(ready[1]);
            assert(read(ready[0], &ch, 1) == 1 && ch == 'R');
            close(ready[0]);
            assert(!kill(child, SIGKILL));
            int got = 0;
            for (int i = 0; i < 200; ++i) {
                if (waitpid(child, &status, WNOHANG) == child) {
                    got = WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
                    break;
                }
                usleep(10000);
            }
            if (!got) break;
            ++reaped;
        }
    }
    int dev = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    assert(dev >= 0);
    struct cv_global g = {0};
    assert(!ioctl(dev, CV_GLOBAL, &g));
    close(dev);
    long pins1 = pin_balance();
    long pin_delta = pins1 - pins0;
    int pass = live_ok && reaped == 2 && g.contexts == 0 && g.arenas == 0 &&
               g.pinned_pages == 0 && pin_delta == 0;
    printf("IOV1 observe case=remap-kill live=%d reaped=%d contexts=%llu arenas=%llu "
           "pinned_pages=%llu pin_delta=%ld %s\n",
           live_ok, reaped, (unsigned long long)g.contexts, (unsigned long long)g.arenas,
           (unsigned long long)g.pinned_pages, pin_delta, pass ? "PASS" : "FAIL");
    return !pass;
}

int main(int argc, char **argv)
{
    assert(argc == 4); /* launcher, domain, case */
    if (!strcmp(argv[3], "remap-kill"))
        return remap_kill();
    int p[2], status;
    char data[256], back[257];
    memset(data, 0x5c, sizeof data);
    assert(!pipe(p) && write(p[1], data, sizeof data) == sizeof data);
    close(p[1]);
    pid_t child = fork(); assert(child >= 0);
    if (!child) {
        assert(dup2(p[0], 0) == 0); close(p[0]);
        execl(argv[1], argv[1], argv[2], argv[3], "stdin", (char *)0);
        perror("exec launcher"); _exit(127);
    }
    assert(waitpid(child, &status, 0) == child);
    ssize_t n = read(p[0], back, sizeof back); close(p[0]);
    int pass = WIFEXITED(status) && WEXITSTATUS(status) == 125 &&
        n == sizeof data && !memcmp(data, back, sizeof data);
    printf("IOV1 observe case=%s status=%d pipe_bytes=%ld %s\n", argv[3], status,
           (long)n, pass ? "PASS" : "FAIL");
    return !pass;
}
