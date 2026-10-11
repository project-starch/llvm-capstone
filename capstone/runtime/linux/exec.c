#define _GNU_SOURCE
#include "application-image.h"
#include "capstone/context.h"
#include "capstone/delegate.h"
#include "capstone/linux-domain-fault.h"
#include "capstone/spawn.h"
#include "delegate-service.h"
#include "libcapstone.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <signal.h>
#include <linux/binfmts.h>
#include <sys/auxv.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>

extern char **environ;

/* Delegation entry block, exchange buffer, and immutable startup data. */
enum { REGION_META, REGION_DATA, REGION_STARTUP, REGIONS };

struct execution {
  struct capstone_delegate_host delegate;   /* the first context's, and the process's */
  size_t slice_bytes;                       /* one transport's exchange region */
  struct capstone_spawner spawner;
  void *maps[REGIONS];
  size_t sizes[REGIONS];
  int image, device_open;
  const char *path;
  unsigned long ticks_start;
};

static unsigned long ticks(void) {
#if defined(__riscv)
  unsigned long t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
#else
  return 0;
#endif
}

static void cleanup(void *context) {
  struct execution *e = context;
  for (unsigned i = 0; i < REGIONS; ++i)
    if (e->maps[i] && e->maps[i] != MAP_FAILED)
      munmap(e->maps[i], e->sizes[i]);
  if (e->device_open)
    capstone_cleanup();
  if (e->image >= 0)
    close(e->image);
  capstone_spawner_stop(&e->spawner);
  capstone_delegate_host_free(&e->delegate);
}

/* The RISC-V timebase from the device tree, big-endian; 0 when unreadable, and
 * the libc then delegates every clock_gettime. */
static uint64_t timebase_frequency(void) {
  unsigned char raw[8];
  int fd = open("/proc/device-tree/cpus/timebase-frequency", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return 0;
  ssize_t n = read(fd, raw, sizeof raw);
  close(fd);
  uint64_t value = 0;
  if (n != 4 && n != 8)
    return 0;
  for (ssize_t i = 0; i < n; ++i)
    value = value << 8 | raw[i];
  return value;
}

/* A CPU whose domains cannot read the time counter. The Capstone FPGA core (device-tree compatible "eth, ariane")
 * has no `time` CSR: csr_regfile.sv has no CSR_TIME read case, so `rdtime` is an illegal instruction there. Linux
 * user code never notices, because OpenSBI emulates it, but a domain's rdtime is not emulated -- it is a fault, or an
 * event under supervision. On that CPU the launcher reports no timebase, and the libc then delegates every clock
 * read (launch.h). The match is an exact entry of the NUL-separated compatible list. */
static int cpu_without_time_csr(void) {
  static const char ariane[] = "eth, ariane";
  char list[256];
  int fd = open("/proc/device-tree/cpus/cpu@0/compatible", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return 0;
  ssize_t n = read(fd, list, sizeof list);
  close(fd);
  for (ssize_t at = 0; at < n;) {
    size_t len = strnlen(list + at, (size_t)(n - at));
    if (len == sizeof ariane - 1 && memcmp(list + at, ariane, len) == 0)
      return 1;
    at += (ssize_t)len + 1;
  }
  return 0;
}

/* What the domain may answer itself: identity, and the clocks paired with the
 * counter it can read. The three reads sit together so the pairing is tight. */
static struct capstone_launch_task task_record(void) {
  struct capstone_launch_task t = {
      .pid = (uint32_t)getpid(), .ppid = (uint32_t)getppid(),
      .uid = getuid(), .euid = geteuid(), .gid = getgid(), .egid = getegid(),
      .ticks_per_second = cpu_without_time_csr() ? 0 : timebase_frequency()};
  struct timespec realtime, monotonic;
  clock_gettime(CLOCK_MONOTONIC, &monotonic);
  clock_gettime(CLOCK_REALTIME, &realtime);
  t.ticks = ticks();
  t.realtime_ns = (uint64_t)realtime.tv_sec * 1000000000u + (uint64_t)realtime.tv_nsec;
  t.monotonic_ns = (uint64_t)monotonic.tv_sec * 1000000000u + (uint64_t)monotonic.tv_nsec;
  return t;
}

/* Where a launch spends its time before the program's first instruction:
 * rdtime at each stage boundary, printed with the counters. */
enum { LAUNCH_START, LAUNCH_IMAGE, LAUNCH_HASH, LAUNCH_SPAWNER, LAUNCH_DEVICE,
       LAUNCH_DOMAIN, LAUNCH_REGIONS, LAUNCH_SECCOMP, LAUNCH_STAGES };
static unsigned long launch_stamp[LAUNCH_STAGES];
static void launch_mark(int stage) { launch_stamp[stage] = ticks(); }

/* Counters on request, to stderr, so a run can be costed without a tool. */
static void report_stats(const struct execution *e) {
  if (!getenv("CAPSTONE_DELEGATE_STATS"))
    return;
  if (launch_stamp[LAUNCH_SECCOMP])
    fprintf(stderr, "capstone-exec: launch ticks image=%lu hash=%lu spawner=%lu device=%lu "
            "domain=%lu regions=%lu seccomp=%lu total=%lu\n",
            launch_stamp[LAUNCH_IMAGE] - launch_stamp[LAUNCH_START],
            launch_stamp[LAUNCH_HASH] - launch_stamp[LAUNCH_IMAGE],
            launch_stamp[LAUNCH_SPAWNER] - launch_stamp[LAUNCH_HASH],
            launch_stamp[LAUNCH_DEVICE] - launch_stamp[LAUNCH_SPAWNER],
            launch_stamp[LAUNCH_DOMAIN] - launch_stamp[LAUNCH_DEVICE],
            launch_stamp[LAUNCH_REGIONS] - launch_stamp[LAUNCH_DOMAIN],
            launch_stamp[LAUNCH_SECCOMP] - launch_stamp[LAUNCH_REGIONS],
            launch_stamp[LAUNCH_SECCOMP] - launch_stamp[LAUNCH_START]);
  fprintf(stderr, "capstone-exec: delegate rounds=%llu syscalls=%llu refused=%llu "
          "bytes_in=%llu bytes_out=%llu ticks=%lu\n",
          (unsigned long long)e->delegate.rounds, (unsigned long long)e->delegate.syscalls,
          (unsigned long long)e->delegate.refused, (unsigned long long)e->delegate.bytes_in,
          (unsigned long long)e->delegate.bytes_out, ticks() - e->ticks_start);
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

static int print_stats(void) {
  struct ioctl_process_stats stats;
  if (capstone_process_init()) { perror("capstone-exec: device"); return 125; }
  if (capstone_process_stats(&stats)) {
    perror("capstone-exec: stats");
    capstone_cleanup();
    return 125;
  }
  printf("{\"version\":%lu,\"live_domains\":%lu,\"live_regions\":%lu,"
         "\"live_bytes\":%lu,\"cached_bytes\":%lu,\"poisoned_blocks\":%lu,"
         "\"nodes_high_water\":%lu,\"nodes_live\":%lu,\"nodes_retired\":%lu,"
         "\"nodes_allocated_total\":%lu,\"tag_pages\":%lu,\"node_capacity\":%lu}\n",
         stats.version, stats.live_domains, stats.live_regions, stats.live_bytes,
         stats.cached_bytes, stats.poisoned_blocks, stats.nodes_high_water,
         stats.nodes_live, stats.nodes_retired, stats.nodes_allocated_total,
         stats.tag_pages, stats.node_capacity);
  capstone_cleanup();
  return 0;
}

/* Held by the thread that ends or replaces the process: by exit, by fault or
   by exec in place. Any other thread that gets to one of them meanwhile waits:
   an end that has begun takes it with the process, and an exec that failed
   gives it back. */
static pthread_mutex_t ending = PTHREAD_MUTEX_INITIALIZER;
static void end_alone(void) {
  pthread_mutex_lock(&ending);
}

/* A fault in any context ends the process with SIGSEGV; the record goes out
 * first, without blocking, so a full pipe cannot swallow the diagnosis. It
 * names the faulting context's last request. `host` is NULL for a fault before
 * the first context ran. Nothing is unmapped or closed first: another context
 * thread may still be serving its transport. */
static void fault(struct execution *e, struct capstone_delegate_host *host,
                  const struct capstone_delegate_entry *entry,
                  const struct ioctl_dom_step_args *step) {
  end_alone();
  if (!host)
    host = &e->delegate;
  if (entry)
    host->preparing_nr = entry->nr;
  /* The record names the image by its SHA-256. Hashing 20 MB byte by byte
     costs a launch two seconds in the guest, so it happens here, on the one
     path that prints it; the image descriptor is still open. */
  if (!e->delegate.image_sha256[0] && e->image >= 0 &&
      capstone_application_hash(e->image, e->delegate.image_sha256))
    e->delegate.image_sha256[0] = 0;
  /* Application streams carry application bytes only. The record goes to the
     file CAPSTONE_FAULT_RECORD names, which the host CLI reads back, and to
     stderr only when that is a terminal or diagnostics were asked for. */
  const char *record = getenv("CAPSTONE_FAULT_RECORD");
  int fd = -1;
  if (record && *record)
    fd = open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (fd >= 0) {
    capstone_delegate_fault_record(fd, host, e->path,
                                   step ? step->cause : 0, step ? step->pc : 0,
                                   step ? step->address : 0);
    close(fd);
  }
  if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))
    capstone_delegate_fault_record(2, host, e->path,
                                   step ? step->cause : 0, step ? step->pc : 0,
                                   step ? step->address : 0);
  report_stats(e);
  capstone_spawner_stop(&e->spawner);
  capstone_domain_exit_on_fault(CAPSTONE_DOMAIN_FAULT_RETVAL, NULL, NULL);
}

/* The way out by exit or exit_group, from any context: report and exit with
   `status`. Nothing is unmapped or closed first, for the reason above;
   exit_group ends every thread before the kernel releases the mappings and
   the device. */
static void process_end(struct execution *e, int status) {
  end_alone();
  report_stats(e);
  capstone_spawner_stop(&e->spawner);
  _exit(status);
}

/* Preserve the unfiltered helper and owned children across exec. Forking a
 * replacement helper after exec would inherit seccomp and break native execs.
 * This private memfd never enters the domain's descriptor table. */
struct exec_state {
  uint64_t magic;
  struct capstone_spawner spawner;
  unsigned child_count;
  int image;
  pid_t children[CAPSTONE_DELEGATE_CHILDREN];
  uint64_t mask;   /* the calling context's mask, the new image's */
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

/* Exec in place from any context, as Linux's execve from any thread: the new
 * image replaces the whole process. The request is the serving host's; the
 * children and the spawner are the process's, under its lock, which a
 * successful execve never releases. */
static long exec_locked(struct execution *e, struct capstone_delegate_host *host);
static long exec_in_place(struct execution *e, struct capstone_delegate_host *host) {
  host->exec_requested = 0;
  pthread_mutex_lock(&ending);
  pthread_mutex_lock(&e->delegate.lock);
  long r = exec_locked(e, host);
  pthread_mutex_unlock(&e->delegate.lock);
  pthread_mutex_unlock(&ending);
  return r;
}

static long exec_locked(struct execution *e, struct capstone_delegate_host *host) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 7], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  struct capstone_application_descriptor_v2 descriptor;
  if (capstone_spawn_unpack(host->exec_block, host->exec_bytes, argv + 6,
                            CAPSTONE_SPAWN_STRINGS + 1, envp, CAPSTONE_SPAWN_STRINGS + 1, paths,
                            CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  int checked = above_stdio(capstone_application_image(view.path, &descriptor));
  if (checked < 0) return -errno;
  if (!view.argc) { close(checked); return -EINVAL; }
  struct exec_state state = {.magic = EXEC_STATE_MAGIC, .spawner = e->spawner,
                            .child_count = e->delegate.child_count, .image = checked,
                            .mask = host->signals.logical};
  memcpy(state.children, e->delegate.children, sizeof state.children);
  int fd = above_stdio(memfd_create("capstone-exec-state", MFD_CLOEXEC));
  if (fd < 0) { int error = errno; close(checked); return -error; }
  int error = 0;
  if (write(fd, &state, sizeof state) != sizeof state || lseek(fd, 0, SEEK_SET) < 0 ||
      fcntl(fd, F_SETFD, 0) < 0 || fcntl(checked, F_SETFD, 0) < 0 ||
      (e->spawner.socket >= 0 && fcntl(e->spawner.socket, F_SETFD, 0) < 0)) {
    error = errno ? errno : EIO;
  } else {
    char fd_string[32];
    snprintf(fd_string, sizeof fd_string, "%d", fd);
    argv[0] = e->spawner.self;
    argv[1] = "--resume";
    argv[2] = fd_string;
    argv[3] = "--application-argv";
    argv[4] = (char *)view.path;
    argv[5] = "--";
    report_stats(e);
    /* execve keeps the calling thread's mask and the pending signals. Every
       signal stays blocked across it, so that one arriving now stays pending
       instead of reaching this image's trampoline; the new launcher sets the
       calling context's mask (the state's) before anything else. Taken back
       if the call fails. */
    uint64_t kept = capstone_signals_set_kernel_mask(~UINT64_C(0));
    execve(e->spawner.self, argv, envp);
    error = errno;
    capstone_signals_set_kernel_mask(kept);
  }
  close(fd);
  close(checked);
  if (e->spawner.socket >= 0) fcntl(e->spawner.socket, F_SETFD, FD_CLOEXEC);
  return -error;
}

/* Contexts the application minted (docs/plans/delegation-threads.md). The
 * launcher registers an offered seal with ADOPT and, in thread mode, steps it
 * from a Linux thread of its own until it ends; Linux schedules that thread
 * like any other. The thread serves the context's delegated calls through the
 * transport the application reserved for it before the request, so a call
 * that blocks in Linux blocks only that context. The first context stays on
 * the main thread. Signals are per context (B8): each thread's kernel mask is
 * its context's, its trampoline records into its context's ring, and tkill
 * reaches the thread that serves the context named. */
enum { TRANSPORT_FREE, TRANSPORT_RESERVED, TRANSPORT_LIVE, TRANSPORT_EXITING };
struct context_service {
  pthread_mutex_t lock;
  pthread_cond_t released;   /* a transport became free */
  struct execution *e;
  dom_id_t first;
  unsigned transports;   /* 1 + the descriptor's contexts; transport 0 is the first's */
  unsigned char state[1 + CAPSTONE_DELEGATE_CONTEXTS_MAX];
  dom_id_t live[1 + CAPSTONE_DELEGATE_CONTEXTS_MAX];
  uint64_t wake[1 + CAPSTONE_DELEGATE_CONTEXTS_MAX];   /* CONTEXT_EXITING's key */
  struct context_thread *threads[1 + CAPSTONE_DELEGATE_CONTEXTS_MAX];
  pthread_cond_t attached;   /* a new context thread serves its signals */
};

struct context_thread {
  struct context_service *service;
  dom_id_t id;
  unsigned transport;
  long tid;          /* the context's thread identity, the runtime's */
  pid_t linux_tid;   /* the thread that serves it, once attached */
  int *attached;     /* the creator's flag, set once and dropped at attach */
  struct capstone_delegate_host host;
};

/* capstone_delegate_host.tkill: to the Linux thread serving the context whose
   thread identity is tid, the first context's being the pid. Sent under the
   service lock, which a context thread takes to leave the table before it
   ends: the Linux tid named is never one Linux has given to another thread. */
static long context_tkill(struct capstone_delegate_host *host, long tid, int sig) {
  struct context_service *service = host->context_state;
  long r = -ESRCH;
  if (tid == getpid())
    return syscall(SYS_tgkill, getpid(), getpid(), sig) ? -errno : 0;
  pthread_mutex_lock(&service->lock);
  for (unsigned i = 1; i < service->transports; ++i) {
    struct context_thread *t = service->threads[i];
    if (t && t->tid == tid && t->linux_tid)
      r = syscall(SYS_tgkill, getpid(), t->linux_tid, sig) ? -errno : 0;
  }
  pthread_mutex_unlock(&service->lock);
  return r;
}


static struct capstone_delegate_entry *transport_entry(struct execution *e, unsigned transport) {
  return (struct capstone_delegate_entry *)((char *)e->maps[REGION_META] +
                                            (size_t)transport * CAPSTONE_DELEGATE_META_BYTES);
}

/* Returns the key CONTEXT_EXITING named, 0 for none. */
static uint64_t transport_release(struct context_service *service, unsigned transport) {
  pthread_mutex_lock(&service->lock);
  uint64_t key = service->wake[transport];
  service->state[transport] = TRANSPORT_FREE;
  service->live[transport] = 0;
  service->wake[transport] = 0;
  service->threads[transport] = NULL;
  pthread_cond_broadcast(&service->released);
  pthread_mutex_unlock(&service->lock);
  return key;
}

static void *context_thread(void *arg) {
  struct context_thread *t = arg;
  struct execution *e = t->service->e;
  struct ioctl_dom_step_args step;
  struct capstone_delegate_entry *entry = transport_entry(e, t->transport);
  /* Serve the context's signals before its first step, and let the creator
     go on: a tkill for it from now on reaches this thread. */
  capstone_signals_attach(&t->host.signals);
  pthread_mutex_lock(&t->service->lock);
  t->linux_tid = (pid_t)syscall(SYS_gettid);
  *t->attached = 1;   /* the creator's own frame: it may not read t again */
  t->attached = NULL;
  pthread_cond_broadcast(&t->service->attached);
  pthread_mutex_unlock(&t->service->lock);
  for (;;) {
    if (capstone_step(t->id, &step)) {
      if (errno == EINTR) continue;
      /* as for the first context: the driver or the monitor refused a step
         that should have run, and the context would never run again */
      fprintf(stderr, "capstone-exec: context %#lx: step: %s\n", (unsigned long)t->id,
              strerror(errno));
      process_end(e, 125);
    }
    if (step.event == CAPSTONE_STEP_PREEMPTED)
      continue;
    if (step.event == CAPSTONE_STEP_FAULT)
      fault(e, &t->host, entry, &step);   /* a fault in any context ends the process */
    if (step.event != CAPSTONE_STEP_RETURNED)
      break;                       /* dead, stale or refused: it can never run */
    if (step.result == CAPSTONE_CONTEXT_EXITED)
      break;
    if (entry->version != CAPSTONE_DELEGATE_VERSION) {
      fprintf(stderr, "capstone-exec: context %#lx returned without a request\n",
              (unsigned long)t->id);
      process_end(e, 125);
    }
    capstone_delegate_serve(&t->host, entry);
    if (t->host.exec_requested)
      entry->result = exec_in_place(e, &t->host);
    if (t->host.exiting)
      process_end(e, t->host.exit_status);
  }
  /* The context can never run again: its transport is free for the next
     one, then its clear word's waiters are woken (a joiner already saw the
     word and went on: the wake is harmless). Free first, so a joiner that
     makes a thread at once finds the transport. */
  /* This thread serves no context any more: a signal that still reaches it
     is kept blocked, as Linux drops a thread's own pending signals when it
     ends; the state goes with the thread record. */
  capstone_signals_detach();
  capstone_forget(t->id);
  capstone_delegate_host_free(&t->host);
  uint64_t key = transport_release(t->service, t->transport);
  if (key)
    capstone_park_wake(e->delegate.park, key, 1);
  free(t);
  return NULL;
}

/* Start a THREAD context on the transport it reserved. The creating thread
   blocks every signal across pthread_create, so the new thread starts with
   them blocked until it attached to its context's signal state. A failed start
   takes the registration back: no context runs after a reported failure.
   CAPSTONE_CONTEXT_TEST_THREAD_FAILS makes the start fail, for the rollback
   probe. */
static long context_start(struct context_service *service, struct capstone_delegate_host *creator,
                          dom_id_t child, unsigned transport, long tid) {
  struct execution *e = service->e;
  struct context_thread *t = calloc(1, sizeof *t);
  if (!t) return -ENOMEM;
  t->service = service;
  t->id = child;
  t->transport = transport;
  t->tid = tid;
  t->host.owner = creator->owner ? creator->owner : creator;
  t->host.exchange = (char *)e->maps[REGION_DATA] + (size_t)transport * e->slice_bytes;
  t->host.exchange_bytes = e->slice_bytes;
  t->host.context = t->host.owner->context;
  t->host.context_state = service;
  t->host.context_id = child;
  t->host.tkill = context_tkill;
  /* the creator's mask, as a new thread inherits it; the process's dispositions */
  capstone_signals_init_context(&t->host.signals,
      (struct capstone_signal_block *)((char *)transport_entry(e, transport) + CAPSTONE_SIGNAL_OFFSET),
      &creator->signals);
  pthread_mutex_lock(&service->lock);
  service->live[transport] = child;
  service->threads[transport] = t;
  pthread_mutex_unlock(&service->lock);
  /* The thread starts with every signal blocked but the C library's own,
     which it unblocks in every new thread; one that arrives there before
     the thread attached is kept (signals.c). The creator waits until it
     attached, on a flag in this frame: the thread may run its context to the
     end and free its record before this reads anything. */
  pthread_t thread;
  int attached = 0;
  t->attached = &attached;
  uint64_t prev = capstone_signals_set_kernel_mask(~UINT64_C(0));
  int failed = getenv("CAPSTONE_CONTEXT_TEST_THREAD_FAILS") ||
               pthread_create(&thread, NULL, context_thread, t);
  capstone_signals_set_kernel_mask(prev);
  if (failed) {
    pthread_mutex_lock(&service->lock);
    service->threads[transport] = NULL;
    pthread_mutex_unlock(&service->lock);
    free(t);
    return -EAGAIN;
  }
  pthread_detach(thread);
  pthread_mutex_lock(&service->lock);
  while (!attached)
    pthread_cond_wait(&service->attached, &service->lock);
  pthread_mutex_unlock(&service->lock);
  return 0;
}

static long context_request(struct capstone_delegate_host *host,
                            const struct capstone_delegate_entry *request) {
  struct context_service *service = host->context_state;
  if (request->nr == CAPSTONE_NR_CONTEXT_RESERVE) {
    /* A context that announced its end frees its transport within its last
       steps, without waiting on anything: wait for it rather than answer
       EAGAIN for a thread that has already finished. */
    long found = service->transports > 1 ? -EAGAIN : -ENOSYS;
    pthread_mutex_lock(&service->lock);
    for (;;) {
      int exiting = 0;
      for (unsigned i = 1; i < service->transports && found < 0; ++i) {
        exiting |= service->state[i] == TRANSPORT_EXITING;
        if (service->state[i] == TRANSPORT_FREE) {
          service->state[i] = TRANSPORT_RESERVED;
          found = i;
        }
      }
      if (found > 0 || !exiting)
        break;
      pthread_cond_wait(&service->released, &service->lock);
    }
    pthread_mutex_unlock(&service->lock);
    return found;
  }
  if (request->nr == CAPSTONE_NR_CONTEXT_EXITING) {
    long r = -EINVAL;   /* the first context, or a REGISTER one: no transport of its own */
    pthread_mutex_lock(&service->lock);
    for (unsigned i = 1; i < service->transports; ++i)
      if (service->state[i] == TRANSPORT_LIVE && service->live[i] == (dom_id_t)host->context_id) {
        service->state[i] = TRANSPORT_EXITING;
        service->wake[i] = request->args[0];
        r = 0;
      }
    pthread_mutex_unlock(&service->lock);
    return r;
  }
  if (request->nr == CAPSTONE_NR_CONTEXT_CREATE) {
    uint64_t mode = request->args[1], transport = request->args[2];
    int thread = mode == CAPSTONE_CONTEXT_THREAD;
    long r = 0;
    dom_id_t child;
    /* The request consumes a reservation whatever its outcome: a THREAD
       request claims it at once, so a second request naming the same
       transport finds it no longer reserved. */
    pthread_mutex_lock(&service->lock);
    int reserved = transport && transport < service->transports &&
                   service->state[transport] == TRANSPORT_RESERVED;
    if (reserved)
      service->state[transport] = thread ? TRANSPORT_LIVE : TRANSPORT_FREE;
    pthread_mutex_unlock(&service->lock);
    if ((thread && !reserved) || (!thread && mode != CAPSTONE_CONTEXT_REGISTER) ||
        (!thread && transport))
      r = -EINVAL;
    else if (capstone_adopt((dom_id_t)host->context_id, request->args[0], &child))
      r = -errno;
    else if (thread && (r = context_start(service, host, child, (unsigned)transport,
                                          (long)request->args[3])))
      capstone_forget(child);
    if (thread && r && reserved)
      transport_release(service, (unsigned)transport);
    return r ? r : (long)child;
  }
  /* STEP and FORGET name a REGISTER context: not the requester itself, and
     not one a context thread steps. */
  dom_id_t id = (dom_id_t)request->args[0];
  int threaded = id == (dom_id_t)host->context_id;
  pthread_mutex_lock(&service->lock);
  for (unsigned i = 0; i < service->transports; ++i)
    threaded |= service->live[i] == id;
  pthread_mutex_unlock(&service->lock);
  if (threaded) return -EINVAL;
  if (request->nr == CAPSTONE_NR_CONTEXT_STEP) {
    struct ioctl_dom_step_args step;
    struct capstone_context_event event = {0};
    while (capstone_step(id, &step))
      if (errno != EINTR) return -errno;
    event.kind = step.event;
    event.result = step.result;
    event.cause = step.cause;
    event.pc = step.pc;
    event.address = step.address;
    if (request->args[2])
      memcpy(host->exchange + request->args[2], &event, sizeof event);
    return 0;
  }
  if (request->nr == CAPSTONE_NR_CONTEXT_FORGET)
    return capstone_forget(id) ? -errno : 0;
  return -ENOSYS;
}

static int fail(struct execution *e, const char *what, int use_errno) {
  if (use_errno)
    perror(what);
  else
    fprintf(stderr, "%s\n", what);
  cleanup(e);
  return 125;
}

int main(int argc, char **argv) {
  /* With binfmt_misc's P flag Linux passes interpreter, image, original argv.
     AT_FLAGS distinguishes this from the explicit launcher command line. */
  int binfmt = (getauxval(AT_FLAGS) & AT_FLAGS_PRESERVE_ARGV0) != 0;
  if (binfmt && argc < 3) return 125;
  struct exec_state resumed = {0};
  if (!binfmt && argc > 3 && !strcmp(argv[1], "--resume")) {
    char *end;
    long fd = strtol(argv[2], &end, 10);
    if (*end || fd < 3 || fd > 65535 || read((int)fd, &resumed, sizeof resumed) != sizeof resumed ||
        resumed.magic != EXEC_STATE_MAGIC || resumed.child_count > CAPSTONE_DELEGATE_CHILDREN)
      return 125;
    close((int)fd);
    if (resumed.spawner.socket >= 0 && fcntl(resumed.spawner.socket, F_SETFD, FD_CLOEXEC) < 0)
      return 125;
    argc -= 2;
    argv += 2;
  }
  int app_args = !binfmt && argc >= 5 && !strcmp(argv[1], "--application-argv") && !strcmp(argv[3], "--");
  int literal = binfmt || (argc > 1 && !strcmp(argv[1], "--"));
  if (literal && !binfmt) {
    --argc;
    ++argv;
  }
  if (argc == 2 && !literal && !strcmp(argv[1], "--stats"))
    return print_stats();
  if (argc < 2 || (!literal && !strcmp(argv[1], "--help"))) {
    fprintf(argc < 2 ? stderr : stdout, "usage: capstone-exec [--] PROGRAM [ARG...]\n       capstone-exec --stats\n");
    return argc < 2 ? 2 : 0;
  }
  capstone_signals_settle_libc();
  if (resumed.magic)
    capstone_signals_set_kernel_mask(resumed.mask);   /* exec in place: the caller's mask */
  struct execution e = {.image = -1, .path = app_args ? argv[2] : argv[1],
                        .spawner = {.socket = -1}};
  if (resumed.magic) {
    e.spawner = resumed.spawner;
    e.delegate.child_count = resumed.child_count;
    memcpy(e.delegate.children, resumed.children, sizeof resumed.children);
  }
  unsigned stdio_mask;
  launch_mark(LAUNCH_START);
  if (reserve_stdio(&stdio_mask))
    return 125;
  struct capstone_application_descriptor_v2 descriptor;
  char resumed_path[64];
  if (resumed.magic) snprintf(resumed_path, sizeof resumed_path, "/proc/self/fd/%d", resumed.image);
  e.image = capstone_application_image(resumed.magic ? resumed_path : e.path, &descriptor);
  int image_error = errno;
  if (resumed.magic) close(resumed.image);
  errno = image_error;
  if (e.image < 0) {
    int error = errno;
    fprintf(stderr, "capstone-exec: %s: %s (requires an image built against this runtime's application descriptor)\n",
            e.path, strerror(error));
    return error == ENOENT ? 127 : 126;
  }
  launch_mark(LAUNCH_IMAGE);
  launch_mark(LAUNCH_HASH); /* the hash is computed only for a fault record */
  char *cwd = getcwd(NULL, 0);
  void *startup = calloc(1, CAPSTONE_LAUNCH_BYTES);
  int first_arg = app_args ? 4 : binfmt ? 2 : 1;
  struct capstone_launch_task task = task_record();
  int error = !cwd || !startup ? ENOMEM : capstone_launch_pack(startup,
      CAPSTONE_LAUNCH_BYTES, argc - first_arg, argv + first_arg, environ, cwd, stdio_mask,
      &task);
  free(cwd);
  if (error) {
    fprintf(stderr, "capstone-exec: startup: %s\n", strerror(error));
    free(startup);
    return fail(&e, "capstone-exec: startup", 0);
  }
  /* Fork before device setup: the helper inherits no domain mappings. */
  int spawn_error = e.spawner.socket >= 0 ? 0 : capstone_spawner_start(&e.spawner);
  if (spawn_error) {
    free(startup);
    errno = spawn_error;
    return fail(&e, "capstone-exec: spawner", 1);
  }
  e.delegate.spawner = &e.spawner;
  launch_mark(LAUNCH_SPAWNER);
  capstone_set_verbose(0);
  /* Find descriptors opened by libcapstone without exposing its private fd
     through the application's direct Linux descriptor namespace. */
  unsigned char before[65536] = {0};
  DIR *fd_dir = opendir("/proc/self/fd");
  if (!fd_dir) { free(startup); return fail(&e, "capstone-exec: descriptors", 1); }
  struct dirent *fd_entry;
  while ((fd_entry = readdir(fd_dir))) {
    char *end;
    long fd = strtol(fd_entry->d_name, &end, 10);
    if (end != fd_entry->d_name && !*end && fd >= 0 && fd < 65536 && fd != dirfd(fd_dir))
      before[fd] = 1;
  }
  closedir(fd_dir);
  if (capstone_process_init()) {
    free(startup);
    return fail(&e, "capstone-exec: device", 1);
  }
  e.device_open = 1;
  launch_mark(LAUNCH_DEVICE);
  e.delegate.private_fds[e.delegate.private_count++] = e.image;
  fd_dir = opendir("/proc/self/fd");
  if (!fd_dir) { free(startup); return fail(&e, "capstone-exec: descriptors", 1); }
  while ((fd_entry = readdir(fd_dir))) {
    char *end;
    long fd = strtol(fd_entry->d_name, &end, 10);
    if (end == fd_entry->d_name || *end || fd < 0 || fd == dirfd(fd_dir)) continue;
    if (fd >= 65536 || (!before[fd] && e.delegate.private_count == 8)) {
      closedir(fd_dir); free(startup);
      return fail(&e, "capstone-exec: private descriptor capacity", 0);
    }
    if (!before[fd]) e.delegate.private_fds[e.delegate.private_count++] = (int)fd;
  }
  closedir(fd_dir);
  char image_path[64];
  snprintf(image_path, sizeof image_path, "/proc/self/fd/%d", e.image);
  dom_id_t domain = create_dom(image_path, NULL);
  if ((long)domain < 0) {
    free(startup);
    return fail(&e, "capstone-exec: cannot create domain", 1);
  }
  if (getenv("CAPSTONE_DELEGATE_STATS"))
    fprintf(stderr, "capstone-exec: domain id=%#lx\n", (unsigned long)domain);
  launch_mark(LAUNCH_DOMAIN);
  /* One transport per context that may run at once: the first context's,
     and the descriptor's contexts, each an entry block and an exchange
     slice at the same index of the two regions. The park table follows the
     last entry block. */
  size_t transports = 1 + (size_t)descriptor.contexts;
  e.slice_bytes = (size_t)descriptor.exchange_bytes;
  e.sizes[REGION_META] = transports * CAPSTONE_DELEGATE_META_BYTES + CAPSTONE_PARK_BYTES;
  e.sizes[REGION_DATA] = transports * e.slice_bytes;
  e.sizes[REGION_STARTUP] = CAPSTONE_LAUNCH_BYTES;
  for (unsigned i = 0; i < REGIONS; ++i) {
    region_id_t region = create_region(e.sizes[i]);
    if ((long)region < 0 ||
        !(e.maps[i] = map_region(region, e.sizes[i])) || e.maps[i] == MAP_FAILED) {
      free(startup);
      return fail(&e, "capstone-exec: cannot allocate launch regions", 0);
    }
    memset(e.maps[i], 0, e.sizes[i]);
    if (i == REGION_STARTUP)
      memcpy(e.maps[i], startup, CAPSTONE_LAUNCH_BYTES);
    if (capstone_share(domain, region, i == REGION_STARTUP ? 0 : 1, 2)) {
      if (errno == EFAULT)
        fault(&e, NULL, NULL, NULL);
      free(startup);
      return fail(&e, "capstone-exec: share launch region", 1);
    }
  }
  free(startup);
  if (descriptor.v1.heap_bytes) {
    region_id_t heap = create_region((unsigned long)descriptor.v1.heap_bytes);
    if ((long)heap < 0)
      return fail(&e, "capstone-exec: cannot allocate application heap", 1);
    if (capstone_share(domain, heap, 1, 3)) {
      if (errno == EFAULT)
        fault(&e, NULL, NULL, NULL);
      return fail(&e, "capstone-exec: share heap", 1);
    }
  }
  e.delegate.exchange = e.maps[REGION_DATA];
  e.delegate.exchange_bytes = e.slice_bytes;
  pthread_mutex_init(&e.delegate.lock, NULL);
  struct capstone_park park;
  if (capstone_park_init(&park, (_Atomic uint64_t *)((char *)e.maps[REGION_META] +
                                                     transports * CAPSTONE_DELEGATE_META_BYTES),
                         CAPSTONE_PARK_BUCKETS))
    return fail(&e, "capstone-exec: park table", 1);
  e.delegate.park = &park;
  e.delegate.context_id = domain;
  struct context_service contexts = {.e = &e, .first = domain, .transports = (unsigned)transports};
  pthread_mutex_init(&contexts.lock, NULL);
  pthread_cond_init(&contexts.released, NULL);
  pthread_cond_init(&contexts.attached, NULL);
  contexts.state[0] = TRANSPORT_LIVE;
  contexts.live[0] = domain;
  e.delegate.context = context_request;
  e.delegate.tkill = context_tkill;
  e.delegate.context_state = &contexts;
  capstone_signals_init(&e.delegate.signals,
      (struct capstone_signal_block *)((char *)e.maps[REGION_META] + CAPSTONE_SIGNAL_OFFSET));
  launch_mark(LAUNCH_REGIONS);
  if (!getenv("CAPSTONE_EXEC_NO_SECCOMP")) {
    int rc = capstone_delegate_seccomp();
    if (rc) {
      errno = rc;
      return fail(&e, "capstone-exec: seccomp filter", 1);
    }
  }
  for (int fd = 0; fd < 3; ++fd)
    if (!(stdio_mask & (1u << fd))) close(fd);
  launch_mark(LAUNCH_SECCOMP);
  e.ticks_start = ticks();
  for (;;) {
    struct ioctl_dom_step_args step;
    /* EINTR from the driver's interruptible lock: the domain was not entered
       and the signal is in the ring. */
    /* From here on other context threads may be running: every way out
       goes through process_end or fault, never through cleanup(). */
    struct capstone_delegate_entry *entry = transport_entry(&e, 0);
    while (capstone_step(domain, &step))
      if (errno != EINTR) {
        perror("capstone-exec: enter domain");
        process_end(&e, 125);
      }
    if (step.event == CAPSTONE_STEP_PREEMPTED)
      continue;
    if (step.event == CAPSTONE_STEP_FAULT)
      fault(&e, &e.delegate, entry, &step);
    if (step.event == CAPSTONE_STEP_DEAD || step.event == CAPSTONE_STEP_STALE ||
        step.event == CAPSTONE_STEP_REFUSED) {
      fprintf(stderr, "capstone-exec: the application's first context is gone\n");
      process_end(&e, 125);
    }
    /* A return with no request means the domain left its entry instead of
       yielding: re-entering would restart it from the top, forever. */
    if (entry->version != CAPSTONE_DELEGATE_VERSION) {
      fprintf(stderr, "capstone-exec: domain returned without a request (entry version %u, "
              "rounds so far %llu)\n", entry->version, (unsigned long long)e.delegate.rounds);
      fprintf(stderr, "capstone-exec: invalid runtime state\n");
      process_end(&e, 125);
    }
    capstone_delegate_serve(&e.delegate, entry);
    if (e.delegate.exec_requested)
      entry->result = exec_in_place(&e, &e.delegate);
    if (e.delegate.exiting)
      process_end(&e, e.delegate.exit_status);
  }
}
