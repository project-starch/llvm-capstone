#define _GNU_SOURCE
#include "wire.h"
#include "../linux/application-image.h"
#include "../linux/delegate-service.h"
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

extern char **environ;
struct mapping { void *address; size_t bytes; uint64_t id; int heap; };
static struct mapping maps[CV_MAX_ARENAS];
static void *reuse_address;
static size_t reuse_bytes;
static int device = -1;
static struct capstone_delegate_host host;
static unsigned long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000 + t.tv_nsec;
}

static size_t rounded(size_t n)
{
    size_t size = 4096;
    if (!n || n > (64u << 20)) return 0;
    while (size < n) size <<= 1;
    return size;
}
static void *reserve(size_t bytes)
{
    /* Let Linux reuse the most recently retired range if it is still free.
     * A fresh grant always creates new identities, even at this same VA. */
    if (reuse_address && bytes == reuse_bytes) {
        void *p = mmap(reuse_address, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        reuse_address = NULL;
        if (p != MAP_FAILED) return p;
    }
    char *raw = mmap(NULL, bytes * 2, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + bytes - 1) & ~(bytes - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes * 2 - base - bytes;
    if (tail) munmap((void *)(base + bytes), tail);
    return (void *)base;
}
static int grant(void *base, size_t bytes, unsigned reg, unsigned perms,
                 uintptr_t cursor, int linear)
{
    unsigned i;
    for (i = 0; i < CV_MAX_ARENAS && maps[i].address; ++i) {}
    if (i == CV_MAX_ARENAS) { errno = ENOSPC; return -1; }
    struct cv_map r = {(uintptr_t)base, bytes, perms, reg, cursor, linear, 0};
    if (ioctl(device, CV_ADD, &r)) return -1;
    maps[i] = (struct mapping){base, bytes, r.id, linear};
    return 0;
}
static long unmap_arena(uintptr_t base, size_t bytes)
{
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        if ((uintptr_t)maps[i].address != base || maps[i].bytes != bytes || !maps[i].heap)
            continue;
        if (ioctl(device, CV_RETIRE, &maps[i].id)) return -errno;
        if (munmap(maps[i].address, maps[i].bytes)) return -errno;
        reuse_address = maps[i].address; reuse_bytes = maps[i].bytes;
        memset(&maps[i], 0, sizeof(maps[i]));
        return 0;
    }
    return -EINVAL;
}
static void cleanup(void)
{
    if (device >= 0) { close(device); device = -1; }
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (maps[i].address) munmap(maps[i].address, maps[i].bytes);
    capstone_delegate_host_free(&host);
}
static int die(const char *what)
{
    fprintf(stderr, "capstone-vexec: %s: %s\n", what, strerror(errno));
    cleanup(); return 125;
}
static int load(int fd, void **image, size_t *bytes, uintptr_t *entry, size_t *stack)
{
    struct stat st;
    if (fstat(fd, &st)) return -1;
    unsigned char *raw = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) return -1;
    /* The shared image inspector has validated every header and file span. */
    const Elf64_Ehdr *h = (void *)raw;
    const Elf64_Phdr *ph = (void *)(raw + h->e_phoff);
    const Elf64_Shdr *sh = (void *)(raw + h->e_shoff);
    const char *names = (void *)(raw + sh[h->e_shstrndx].sh_offset);
    uint64_t low = UINT64_MAX, high = 0;
    int virtual = 0;
    *stack = 1u << 20;
    for (unsigned i = 0; i < h->e_phnum; ++i) if (ph[i].p_type == PT_LOAD) {
        if (ph[i].p_vaddr < low) low = ph[i].p_vaddr;
        if (ph[i].p_vaddr + ph[i].p_memsz > high) high = ph[i].p_vaddr + ph[i].p_memsz;
    }
    for (unsigned i = 0; i < h->e_shnum; ++i) {
        uint64_t v[3];
        if (sh[i].sh_type != SHT_PROGBITS) continue;
        if (!strcmp(names + sh[i].sh_name, ".capstone_virtual") && sh[i].sh_size == 8) {
            memcpy(v, raw + sh[i].sh_offset, 8);
            virtual = v[0] == UINT64_C(0x314d56564e4f5043);
        }
        if (!strcmp(names + sh[i].sh_name, ".capstone_domreq") && sh[i].sh_size == 24) {
            memcpy(v, raw + sh[i].sh_offset, 24);
            *stack = rounded(v[2]);
        }
    }
    if (!virtual || !*stack || !(*bytes = rounded(high - low))) {
        munmap(raw, st.st_size); errno = ENOEXEC; return -1;
    }
    *image = reserve(*bytes);
    if (!*image) { munmap(raw, st.st_size); return -1; }
    for (unsigned i = 0; i < h->e_phnum; ++i) if (ph[i].p_type == PT_LOAD)
        memcpy((char *)*image + ph[i].p_vaddr - low, raw + ph[i].p_offset, ph[i].p_filesz);
    *entry = (uintptr_t)*image + h->e_entry - low;
    munmap(raw, st.st_size);
    if (mprotect(*image, *bytes, PROT_READ | PROT_WRITE | PROT_EXEC)) {
        munmap(*image, *bytes); return -1;
    }
    __builtin___clear_cache(*image, (char *)*image + *bytes);
    return 0;
}
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: capstone-vexec PROGRAM.dom [ARG...]\n"); return 2; }
    unsigned long long started = now_ns(), launch_ns;
    int trace = getenv("CAPSTONE_VM_TRACE") != NULL;
    struct capstone_application_descriptor_v2 desc;
    int fd = capstone_application_image(argv[1], &desc);
    if (fd < 0) { perror("capstone-vexec: image"); return 126; }
    void *image; size_t image_bytes, stack_bytes; uintptr_t entry;
    if (load(fd, &image, &image_bytes, &entry, &stack_bytes)) { close(fd); perror("capstone-vexec: virtual ABI"); return 126; }
    close(fd);
    device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    if (device < 0) { munmap(image, image_bytes); return die("open"); }
    if (grant(image, image_bytes, 3, 7, entry, 0)) { munmap(image, image_bytes); return die("image grant"); }
    void *stack = reserve(stack_bytes), *meta = reserve(CAPSTONE_DELEGATE_META_BYTES);
    size_t exchange_bytes = rounded(desc.exchange_bytes);
    void *exchange = exchange_bytes ? reserve(exchange_bytes) : NULL;
    void *startup = reserve(CAPSTONE_LAUNCH_BYTES);
    if (!stack || !meta || !exchange || !startup) return die("allocate startup");
    char *cwd = getcwd(NULL, 0);
    struct timespec rt, mt;
    clock_gettime(CLOCK_REALTIME, &rt); clock_gettime(CLOCK_MONOTONIC, &mt);
    struct capstone_launch_task task = { .pid = getpid(), .ppid = getppid(),
        .uid = getuid(), .euid = geteuid(), .gid = getgid(), .egid = getegid(),
        .realtime_ns = (uint64_t)rt.tv_sec * 1000000000 + rt.tv_nsec,
        .monotonic_ns = (uint64_t)mt.tv_sec * 1000000000 + mt.tv_nsec };
    int error = cwd ? capstone_launch_pack(startup, CAPSTONE_LAUNCH_BYTES,
        argc - 1, argv + 1, environ, cwd, 7, &task) : ENOMEM;
    free(cwd);
    if (error) { errno = error; return die("startup"); }
    if (grant(stack, stack_bytes, 2, 6, (uintptr_t)stack + stack_bytes, 0) ||
        grant(meta, CAPSTONE_DELEGATE_META_BYTES, 10, 6, (uintptr_t)meta, 0) ||
        grant(exchange, exchange_bytes, 11, 6, (uintptr_t)exchange, 0) ||
        grant(startup, CAPSTONE_LAUNCH_BYTES, 12, 4, (uintptr_t)startup, 0)) return die("grant");
    host.exchange = exchange; host.exchange_bytes = exchange_bytes;
    host.private_fds[host.private_count++] = device;
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    if ((error = capstone_delegate_seccomp())) { errno = error; return die("seccomp"); }
    launch_ns = now_ns() - started;
    struct cv_step step = {0};
    for (;;) {
        while (ioctl(device, CV_STEP, &step))
            if (errno != EINTR) return die("step");
        step.reply = 0;
        if (step.kind == 1) continue;
        if (step.kind == 4) {
            if (!ioctl(device, CV_RESOLVE)) continue;
        }
        if (step.kind != 3) {
            fprintf(stderr, "CAPSTONE_VM_FAULT cause=%llu pc=%llx address=%llx\n",
                    step.cause, step.pc, step.address);
            cleanup(); return 128 + SIGSEGV;
        }
        step.reply = 1; step.result = 0;
        if (step.args[7] == CV_SERVICE_DELEGATE) {
            struct capstone_delegate_entry *request = meta;
            if (request->version != CAPSTONE_DELEGATE_VERSION) { errno = EPROTO; return die("service"); }
            capstone_delegate_serve(&host, request);
            if (trace) fprintf(stderr, "CAPSTONE_VM_SERVICE nr=%llu result=%lld\n",
                               (unsigned long long)request->nr, (long long)request->result);
            if (host.exec_requested) { host.exec_requested = 0; request->result = -ENOSYS; }
            if (host.exiting) {
                struct cv_stats stats;
                if (!ioctl(device, CV_STATS, &stats))
                    fprintf(stderr, "CAPSTONE_VM_STATS arenas=%llu pages=%llu peak=%llu nodes=%llu steps=%llu faults=%llu rounds=%llu bytes_in=%llu bytes_out=%llu node_bytes=%lu launch_ns=%llu elapsed_ns=%llu\n",
                        stats.arenas, stats.pinned_pages, stats.peak_pages, stats.nodes, stats.steps,
                        stats.faults, (unsigned long long)host.rounds,
                        (unsigned long long)host.bytes_in, (unsigned long long)host.bytes_out,
                        CV_NODE_BYTES, launch_ns, now_ns() - started);
                int result = host.exit_status; cleanup(); return result;
            }
        } else if (step.args[7] == CV_SERVICE_MAP) {
            size_t bytes = rounded(step.args[0]);
            void *p = bytes ? reserve(bytes) : NULL;
            if (p && grant(p, bytes, 10, 6, (uintptr_t)p, 1)) { munmap(p, bytes); p = NULL; }
            /* ADD moved a tagged reply into a0's frame slot. Otherwise zero
             * is the explicit allocation failure; a scalar address is never minted by libc. */
            step.result = 0;
        } else if (step.args[7] == CV_SERVICE_UNMAP) {
            step.result = unmap_arena(step.args[0], step.args[1]);
        } else step.result = -ENOSYS;
    }
}
