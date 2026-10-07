#define _GNU_SOURCE
#include "wire.h"
#include "capstone/linux-domain-fault.h"
#include "capstone/spawn.h"
#include <linux/binfmts.h>
#include <sys/auxv.h>
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
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

extern char **environ;
struct mapping {
    void *address;
    size_t bytes, visible;
    uint64_t id;
    int heap, kind;
};
static struct mapping maps[CV_MAX_ARENAS];
static void *reuse_address;
static size_t reuse_bytes;
static int device = -1, image_fd = -1;
static const char *image_path;
static struct capstone_spawner spawner = {.socket = -1};
static struct capstone_delegate_host host;
#define CV_MAX_THREADS 32
static uint64_t thread_ids[CV_MAX_THREADS] = {0};
static unsigned thread_done[CV_MAX_THREADS];
static unsigned thread_count = 1;
static pthread_mutex_t service_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t thread_state_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t thread_state_cond = PTHREAD_COND_INITIALIZER;
static unsigned long long launch_ns, start_ns;
static int trace;
static void *meta;
static void *virtual_thread_worker(void *opaque);

static int thread_add(uint64_t id)
{
    int result = 0;
    if (!id) return -EINVAL;
    pthread_mutex_lock(&thread_state_lock);
    if (thread_count >= CV_MAX_THREADS) result = -EAGAIN;
    else {
        for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
            if (!thread_ids[i]) {
                thread_ids[i] = id;
                thread_done[i] = 0;
                ++thread_count;
                break;
            }
    }
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
static void thread_remove(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        thread_ids[i] = 0;
        thread_done[i] = 0;
        --thread_count;
        pthread_mutex_unlock(&thread_state_lock);
        return;
    }
    pthread_mutex_unlock(&thread_state_lock);
}
static void thread_mark_done(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
        if (thread_ids[i] == id) {
            thread_done[i] = 1;
            pthread_cond_broadcast(&thread_state_cond);
            break;
        }
    pthread_mutex_unlock(&thread_state_lock);
}
static int thread_join(uint64_t id)
{
    int result = 0;
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        while (!thread_done[i])
            pthread_cond_wait(&thread_state_cond, &thread_state_lock);
        thread_ids[i] = 0;
        thread_done[i] = 0;
        --thread_count;
        pthread_mutex_unlock(&thread_state_lock);
        return 0;
    }
    result = -ESRCH;
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
static unsigned long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000 + t.tv_nsec;
}

static size_t rounded(size_t n)
{
    size_t size = 4096;
    if (!n || n > CV_MAX_REGION_BYTES) return 0;
    while (size < n) size <<= 1;
    return size;
}
static void *private_pages(void *p, size_t bytes)
{
    /* Demand resolution accounts for one base page at a time. The module
     * requires this VMA policy on kernels that can instantiate huge pages. */
    if (madvise(p, bytes, MADV_NOHUGEPAGE) && errno != EINVAL) {
        int error = errno; munmap(p, bytes); errno = error; return NULL;
    }
    return p;
}
static void *reserve(size_t bytes)
{
    /* Let Linux reuse the most recently retired range if it is still free.
     * A fresh grant always creates new identities, even at this same VA. */
    if (reuse_address && bytes == reuse_bytes) {
        void *p = mmap(reuse_address, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        reuse_address = NULL;
        if (p != MAP_FAILED) return private_pages(p, bytes);
    }
    char *raw = mmap(NULL, bytes * 2, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + bytes - 1) & ~(bytes - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes * 2 - base - bytes;
    if (tail) munmap((void *)(base + bytes), tail);
    return private_pages((void *)base, bytes);
}
static int grant(void *base, size_t bytes, unsigned reg, unsigned perms,
                 uintptr_t cursor, int linear, uint64_t thread)
{
    unsigned i;
    for (i = 0; i < CV_MAX_ARENAS && maps[i].address; ++i) {}
    if (i == CV_MAX_ARENAS) { errno = ENOSPC; return -1; }
    struct cv_map r = {thread, (uintptr_t)base, bytes, perms, reg, cursor, linear, 0};
    if (ioctl(device, CV_ADD, &r)) return -1;
    maps[i] = (struct mapping){.address = base, .bytes = bytes, .visible = bytes,
                               .id = r.id, .heap = linear,
                               .kind = linear ? CV_MAP_HEAP : -1};
    return 0;
}
static long unmap_arena(uintptr_t base, size_t bytes)
{
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        if ((uintptr_t)maps[i].address != base || maps[i].visible != bytes || !maps[i].heap || maps[i].kind == CV_MAP_METADATA)
            continue;
        if (ioctl(device, CV_RETIRE, &maps[i].id)) return -errno;
        if (munmap(maps[i].address, maps[i].bytes)) return -errno;
        reuse_address = maps[i].address; reuse_bytes = maps[i].bytes;
        memset(&maps[i], 0, sizeof(maps[i]));
        return 0;
    }
    return -EINVAL;
}
static long protect_range(uintptr_t base, size_t bytes, int prot)
{
    if (!bytes || (base & 4095) || (bytes & 4095) ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))) return -EINVAL;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        struct mapping *m = &maps[i];
        uintptr_t at = (uintptr_t)m->address;
        if (!m->address || !m->heap || m->kind == CV_MAP_METADATA || base < at ||
            base - at > m->visible || bytes > m->visible - (base - at)) continue;
        return mprotect((void *)base, bytes, prot) ? -errno : 0;
    }
    return -EINVAL;
}
static long acquire_mapping(uint64_t tid, struct cv_step *step)
{
    size_t visible = step->args[0], alignment = step->args[1];
    unsigned prot = step->args[2], rights = step->args[3], kind = step->args[4];
    size_t bytes = rounded(visible);
    if (!bytes || (visible & 4095) || alignment < 4096 ||
        (alignment & (alignment - 1)) || alignment > CV_MAX_REGION_BYTES ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) || rights > 7 ||
        kind > CV_MAP_METADATA ||
        (kind != CV_MAP_APPLICATION && (prot != (PROT_READ | PROT_WRITE) || rights != 6)) ||
        (kind == CV_MAP_APPLICATION && rights != 7)) return -EINVAL;
    if (bytes < alignment) bytes = alignment;
    void *p = reserve(bytes);
    if (!p) return -errno;
    /* Padding remains inaccessible, including after later mprotect calls.
     * No capability reply exists until registration has succeeded. */
    if ((visible < bytes && mprotect((char *)p + visible, bytes - visible, PROT_NONE)) ||
        mprotect(p, visible, prot) || grant(p, bytes, 10, rights, (uintptr_t)p, 1, tid)) {
        int error = errno;
        munmap(p, bytes);
        return -error;
    }
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (maps[i].address == p) {
            maps[i].visible = visible; maps[i].kind = kind;
            break;
        }
    return 0;
}
static void cleanup(void)
{
    if (device >= 0) { close(device); device = -1; }
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (maps[i].address) munmap(maps[i].address, maps[i].bytes);
    if (image_fd >= 0) { close(image_fd); image_fd = -1; }
    capstone_spawner_stop(&spawner);
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
            virtual = v[0] == CV_IMAGE_MAGIC;
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
struct exec_state {
  uint64_t magic;
  struct capstone_spawner spawner;
  unsigned child_count;
  int image;
  pid_t children[CAPSTONE_DELEGATE_CHILDREN];
};
#define EXEC_STATE_MAGIC UINT64_C(0x4350455845433031)

static int above_stdio(int fd) {
  if (fd < 0 || fd >= 3) return fd;
  int parked = fcntl(fd, F_DUPFD_CLOEXEC, 3);
  int error = errno;
  close(fd);
  errno = error;
  return parked;
}

static long exec_in_place(void) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 7], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  struct capstone_application_descriptor_v2 descriptor;
  host.exec_requested = 0;
  if (capstone_spawn_unpack(host.exec_block, host.exec_bytes, argv + 6,
                            CAPSTONE_SPAWN_STRINGS + 1, envp, CAPSTONE_SPAWN_STRINGS + 1, paths,
                            CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  int checked = above_stdio(capstone_application_image(view.path, &descriptor));
  if (checked < 0) return -errno;
  if (!view.argc) { close(checked); return -EINVAL; }
  struct exec_state state = {.magic = EXEC_STATE_MAGIC, .spawner = spawner,
                            .child_count = host.child_count, .image = checked};
  memcpy(state.children, host.children, sizeof state.children);
  int fd = above_stdio(memfd_create("capstone-exec-state", MFD_CLOEXEC));
  if (fd < 0) { int error = errno; close(checked); return -error; }
  int error = 0;
  if (write(fd, &state, sizeof state) != sizeof state || lseek(fd, 0, SEEK_SET) < 0 ||
      fcntl(fd, F_SETFD, 0) < 0 || fcntl(checked, F_SETFD, 0) < 0 ||
      (spawner.socket >= 0 && fcntl(spawner.socket, F_SETFD, 0) < 0)) {
    error = errno ? errno : EIO;
  } else {
    char fd_string[32];
    snprintf(fd_string, sizeof fd_string, "%d", fd);
    argv[0] = spawner.self;
    argv[1] = "--resume";
    argv[2] = fd_string;
    argv[3] = "--application-argv";
    argv[4] = (char *)view.path;
    argv[5] = "--";
    execve(spawner.self, argv, envp);
    error = errno;
  }
  close(fd);
  close(checked);
  if (spawner.socket >= 0) fcntl(spawner.socket, F_SETFD, FD_CLOEXEC);
  return -error;
}

static void cleanup_fault(void *unused) { (void)unused; cleanup(); }
static void fault(const struct cv_step *step)
{
    if (!host.image_sha256[0] && image_fd >= 0)
        capstone_application_hash(image_fd, host.image_sha256);
    const char *record = getenv("CAPSTONE_FAULT_RECORD");
    int fd = record && *record ? open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600) : -1;
    if (fd >= 0) {
        capstone_delegate_fault_record(fd, &host, image_path, step->cause, step->pc, step->address);
        close(fd);
    }
    if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))
        capstone_delegate_fault_record(2, &host, image_path, step->cause, step->pc, step->address);
    capstone_domain_exit_on_fault(CAPSTONE_DOMAIN_FAULT_RETVAL, cleanup_fault, NULL);
}
static int reserve_stdio(unsigned *mask) {
  *mask = 0;
  for (int i = 0; i < 3; ++i) {
    if (fcntl(i, F_GETFD) >= 0) {
      *mask |= 1u << i;
      continue;
    }
    if (errno != EBADF)
      return -1;
    int fd = open("/dev/null", O_RDWR);
    if (fd < 0)
      return -1;
    if (fd != i) {
      int rc = dup2(fd, i);
      close(fd);
      if (rc < 0)
        return -1;
    }
  }
  return 0;
}

static int virtual_service(uint64_t tid, struct cv_step *step)
{
    step->reply = 1;
    step->result = 0;
    if (step->args[7] == CV_SERVICE_THREAD_CREATE) {
        struct cv_thread_create request = {.frame = step->args[0]};
        if (ioctl(device, CV_THREAD_CREATE, &request)) {
            step->result = -errno;
        } else if (thread_add(request.thread)) {
            struct cv_thread_control rollback = {.thread = request.thread};
            ioctl(device, CV_THREAD_EXIT, &rollback);
            step->result = -EAGAIN;
        } else {
            uint64_t *child = malloc(sizeof(*child));
            pthread_t worker;
            if (!child || pthread_create(&worker, NULL, virtual_thread_worker,
                                         (child ? (*child = request.thread), child : NULL))) {
                free(child);
                thread_remove(request.thread);
                struct cv_thread_control rollback = {.thread = request.thread};
                ioctl(device, CV_THREAD_EXIT, &rollback);
                step->result = -EAGAIN;
            } else {
                pthread_detach(worker);
                step->result = request.thread;
            }
        }
    } else if (step->args[7] == CV_SERVICE_THREAD_EXIT) {
        struct cv_thread_control request = {.thread = tid};
        if (tid == 0 || ioctl(device, CV_THREAD_EXIT, &request))
            return -1;
        if (request.frame) {
            long cleanup_rc = unmap_arena(request.frame, 4096);
            if (cleanup_rc && cleanup_rc != -EINVAL) return -1;
        }
        thread_mark_done(tid);
        step->reply = 0;
        return 1;
    } else if (step->args[7] == CV_SERVICE_THREAD_JOIN) {
        /* The child must acquire the same transport lock to publish its
         * exit.  Do not wait while holding it, or join would deadlock. */
        pthread_mutex_unlock(&service_lock);
        step->result = thread_join(step->args[0]);
        pthread_mutex_lock(&service_lock);
    } else if (step->args[7] == CV_SERVICE_DELEGATE) {
        struct capstone_delegate_entry *request = meta;
        if (request->version != CAPSTONE_DELEGATE_VERSION) return -1;
        capstone_delegate_serve(&host, request);
        if (trace) fprintf(stderr, "CAPSTONE_VM_SERVICE tid=%llu nr=%llu result=%lld\n",
                           (unsigned long long)tid, (unsigned long long)request->nr,
                           (long long)request->result);
        if (host.exec_requested) request->result = exec_in_place();
        if (host.exiting) {
            struct cv_stats stats;
            if (tid == 0 && (getenv("CAPSTONE_VM_STATS") || getenv("CAPSTONE_DELEGATE_STATS")) &&
                !ioctl(device, CV_STATS, &stats))
                fprintf(stderr, "CAPSTONE_VM_STATS arenas=%llu pages=%llu peak=%llu nodes=%llu steps=%llu faults=%llu collections=%llu reclaimed=%llu rounds=%llu bytes_in=%llu bytes_out=%llu node_bytes=%lu launch_ns=%llu elapsed_ns=%llu\n",
                    stats.arenas, stats.pinned_pages, stats.peak_pages, stats.nodes, stats.steps,
                    stats.faults, stats.collections, stats.reclaimed, (unsigned long long)host.rounds,
                    (unsigned long long)host.bytes_in, (unsigned long long)host.bytes_out,
                    CV_NODE_BYTES, launch_ns, now_ns() - start_ns);
            if (tid == 0) {
                int result = host.exit_status;
                cleanup();
                return 2 | (result << 8);
            }
            _exit(host.exit_status);
        }
    } else if (step->args[7] == CV_SERVICE_MAP) {
        step->result = acquire_mapping(tid, step);
    } else if (step->args[7] == CV_SERVICE_UNMAP) {
        step->result = unmap_arena(step->args[0], step->args[1]);
    } else if (step->args[7] == CV_SERVICE_PROTECT) {
        step->result = protect_range(step->args[0], step->args[1], step->args[2]);
    } else if (step->args[7] == CV_SERVICE_WAIT) {
        /* The heap's contended atomic mutex must let its owner run, including
         * the owner's mapping service. Never yield with the wire lock held. */
        pthread_mutex_unlock(&service_lock);
        sched_yield();
        pthread_mutex_lock(&service_lock);
    } else {
        step->result = -ENOSYS;
    }
    return 0;
}

static int virtual_service_loop(uint64_t tid)
{
    struct cv_step step = {.thread = tid};
    for (;;) {
        while (ioctl(device, CV_STEP, &step))
            if (errno != EINTR) return die("step");
        step.reply = 0;
        if (step.kind == 1) {
            /* The supervisor quantum is the preemption point. Yielding here
             * lets Linux schedule another launcher thread on this one hart. */
            sched_yield();
            continue;
        }
        if (step.kind == 4) {
            struct cv_thread_control control = {.thread = step.thread};
            if (!ioctl(device, CV_RESOLVE, &control)) continue;
        }
        if (step.kind != 3) {
            pthread_mutex_lock(&service_lock);
            host.preparing_nr = ((struct capstone_delegate_entry *)meta)->nr;
            fault(&step);
            pthread_mutex_unlock(&service_lock);
            return 125;
        }
        pthread_mutex_lock(&service_lock);
        int result = virtual_service(tid, &step);
        pthread_mutex_unlock(&service_lock);
        if (result == 1) return 0;
        if (result < 0) return die("service");
        if (result & 2) return result >> 8;
    }
}

static void *virtual_thread_worker(void *opaque)
{
    uint64_t tid = *(uint64_t *)opaque;
    free(opaque);
    int result = virtual_service_loop(tid);
    /* A virtual thread is hosted by a Linux worker, so normal virtual-thread
     * exit must return from this worker rather than terminate the process. */
    if (result) _exit(result);
    return NULL;
}

int main(int argc, char **argv)
{
    int binfmt = (getauxval(AT_FLAGS) & AT_FLAGS_PRESERVE_ARGV0) != 0;
    if (binfmt && argc < 3) return 125;
    struct exec_state resumed = {0};
    if (!binfmt && argc > 3 && !strcmp(argv[1], "--resume")) {
        char *end; long fd = strtol(argv[2], &end, 10);
        if (*end || fd < 3 || fd > 65535 || read(fd, &resumed, sizeof resumed) != sizeof resumed ||
            resumed.magic != EXEC_STATE_MAGIC || resumed.child_count > CAPSTONE_DELEGATE_CHILDREN)
            return 125;
        close(fd);
        if (resumed.spawner.socket >= 0 && fcntl(resumed.spawner.socket, F_SETFD, FD_CLOEXEC) < 0)
            return 125;
        spawner = resumed.spawner; host.child_count = resumed.child_count;
        memcpy(host.children, resumed.children, sizeof resumed.children);
        argc -= 2; argv += 2;
    }
    int app_args = !binfmt && argc >= 5 && !strcmp(argv[1], "--application-argv") && !strcmp(argv[3], "--");
    int literal = binfmt || (argc > 1 && !strcmp(argv[1], "--"));
    if (literal && !binfmt) { --argc; ++argv; }
    if (argc < 2) { fprintf(stderr, "usage: capstone-vexec [--] PROGRAM.dom [ARG...]\n"); return 2; }
    if (!literal && argc == 2 && !strcmp(argv[1], "--stats")) {
        int fd = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
        struct cv_global stats;
        if (fd < 0 || ioctl(fd, CV_GLOBAL, &stats)) return 125;
        close(fd);
        printf("{\"version\":1,\"live_domains\":%llu,\"live_regions\":%llu,\"live_bytes\":%llu,\"cached_bytes\":0,\"poisoned_blocks\":0,\"nodes_high_water\":%llu,\"nodes_live\":%llu,\"nodes_retired\":%llu,\"nodes_allocated_total\":%llu,\"tag_pages\":%llu,\"node_capacity\":%lu,\"collections\":%llu,\"nodes_reclaimed\":%llu}\n",
            stats.contexts, stats.arenas, stats.pinned_pages * 4096,
            stats.nodes_high_water, stats.nodes_live, stats.nodes_retired, stats.nodes_allocated,
            stats.pinned_pages, CV_NODE_BYTES / 16, stats.collections, stats.reclaimed);
        return 0;
    }
    image_path = app_args ? argv[2] : argv[1];
    unsigned stdio_mask;
    if (reserve_stdio(&stdio_mask)) return 125;
    start_ns = now_ns();
    trace = getenv("CAPSTONE_VM_TRACE") != NULL;
    struct capstone_application_descriptor_v2 desc;
    char resumed_path[64];
    if (resumed.magic) snprintf(resumed_path, sizeof resumed_path, "/proc/self/fd/%d", resumed.image);
    int fd = capstone_application_image(resumed.magic ? resumed_path : image_path, &desc);
    int image_error = errno;
    if (resumed.magic) close(resumed.image);
    errno = image_error;
    if (fd < 0) { perror("capstone-vexec: image"); return errno == ENOENT ? 127 : 126; }
    image_fd = fd;
    if (spawner.socket < 0) { int error = capstone_spawner_start(&spawner);
        if (error) { errno = error; return die("spawner"); } }
    host.spawner = &spawner;
    void *image; size_t image_bytes, stack_bytes; uintptr_t entry;
    if (load(fd, &image, &image_bytes, &entry, &stack_bytes)) { close(fd); perror("capstone-vexec: virtual ABI"); return 126; }
    host.private_fds[host.private_count++] = fd;
    device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    if (device < 0) { munmap(image, image_bytes); return die("open"); }
    if (grant(image, image_bytes, 3, 7, entry, 0, 0)) { munmap(image, image_bytes); return die("image grant"); }
    void *stack = reserve(stack_bytes);
    meta = reserve(CAPSTONE_DELEGATE_META_BYTES);
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
        argc - (app_args ? 4 : binfmt ? 2 : 1), argv + (app_args ? 4 : binfmt ? 2 : 1), environ, cwd, stdio_mask, &task) : ENOMEM;
    free(cwd);
    if (error) { errno = error; return die("startup"); }
    if (grant(stack, stack_bytes, 2, 6, (uintptr_t)stack + stack_bytes, 0, 0) ||
        grant(meta, CAPSTONE_DELEGATE_META_BYTES, 10, 6, (uintptr_t)meta, 0, 0) ||
        grant(exchange, exchange_bytes, 11, 6, (uintptr_t)exchange, 0, 0) ||
        grant(startup, CAPSTONE_LAUNCH_BYTES, 12, 4, (uintptr_t)startup, 0, 0)) return die("grant");
    if (desc.v1.heap_bytes) {
        size_t bytes = rounded(desc.v1.heap_bytes);
        void *region = bytes ? reserve(bytes) : NULL;
        if (!region || grant(region, bytes, 13, 6, (uintptr_t)region, 1, 0))
            return die("nested allocator grant");
    }
    host.exchange = exchange; host.exchange_bytes = exchange_bytes;
    host.private_fds[host.private_count++] = device;
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    if ((error = capstone_delegate_seccomp())) { errno = error; return die("seccomp"); }
    for (int fd = 0; fd < 3; ++fd) if (!(stdio_mask & (1u << fd))) close(fd);
    launch_ns = now_ns() - start_ns;
    return virtual_service_loop(0);
}
