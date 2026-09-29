#define _GNU_SOURCE
#include "application-image.h"
#include "capstone/application-service.h"
#include "capstone/delegate.h"
#include "capstone/linux-domain-fault.h"
#include "capstone/spawn.h"
#include "delegate-service.h"
#include "host-service.h"
#include "libcapstone.h"
#include <fcntl.h>
#include <dirent.h>
#include <signal.h>
#include <linux/binfmts.h>
#include <sys/auxv.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

extern char **environ;

/* Region roles. Both ABIs share the first three slots: v1 reads them as
 * metadata, payload and startup; v2 as entry block, exchange and startup. */
enum { REGION_META, REGION_DATA, REGION_STARTUP, REGIONS };

struct execution {
  struct hc_host host;
  struct capstone_delegate_host delegate;
  struct capstone_spawner spawner;
  void *maps[REGIONS];
  size_t sizes[REGIONS];
  int image, device_open, delegated;
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
  if (!e->delegated)
    hostcall_cleanup_open_handles(e->host.slots, HC_FILE_SERVICE_MAX_HANDLES);
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

/* Counters on request, to stderr, so a run can be costed without a tool. */
static void report_stats(const struct execution *e) {
  if (!e->delegated || !getenv("CAPSTONE_DELEGATE_STATS"))
    return;
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

/* A fault ends the process with SIGSEGV after cleanup; the record goes out
 * first, without blocking, so a full pipe cannot swallow the diagnosis. */
static void fault(struct execution *e, const struct ioctl_dom_step_args *step) {
  if (e->delegated && e->maps[REGION_META])
    e->delegate.preparing_nr = ((struct capstone_delegate_entry *)e->maps[REGION_META])->nr;
  /* Application streams carry application bytes only. The record goes to the
     file CAPSTONE_FAULT_RECORD names, which the host CLI reads back, and to
     stderr only when that is a terminal or diagnostics were asked for. */
  const char *record = getenv("CAPSTONE_FAULT_RECORD");
  int fd = -1;
  if (record && *record)
    fd = open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (fd >= 0) {
    capstone_delegate_fault_record(fd, e->delegated ? &e->delegate : NULL, e->path,
                                   step ? step->cause : 0, step ? step->pc : 0,
                                   step ? step->address : 0);
    close(fd);
  }
  if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))
    capstone_delegate_fault_record(2, e->delegated ? &e->delegate : NULL, e->path,
                                   step ? step->cause : 0, step ? step->pc : 0,
                                   step ? step->address : 0);
  report_stats(e);
  capstone_domain_exit_on_fault(CAPSTONE_DOMAIN_FAULT_RETVAL, cleanup, e);
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

static long exec_in_place(struct execution *e) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 7], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  struct capstone_application_descriptor_v2 descriptor;
  e->delegate.exec_requested = 0;
  if (capstone_spawn_unpack(e->delegate.exec_block, e->delegate.exec_bytes, argv + 6,
                            CAPSTONE_SPAWN_STRINGS + 1, envp, CAPSTONE_SPAWN_STRINGS + 1, paths,
                            CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  int checked = above_stdio(capstone_application_image(view.path, &descriptor));
  if (checked < 0) return -errno;
  if (!view.argc) { close(checked); return -EINVAL; }
  struct exec_state state = {.magic = EXEC_STATE_MAGIC, .spawner = e->spawner,
                            .child_count = e->delegate.child_count, .image = checked};
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
    execve(e->spawner.self, argv, envp);
    error = errno;
  }
  close(fd);
  close(checked);
  if (e->spawner.socket >= 0) fcntl(e->spawner.socket, F_SETFD, FD_CLOEXEC);
  return -error;
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
  struct execution e = {.image = -1, .path = app_args ? argv[2] : argv[1],
                        .spawner = {.socket = -1}};
  if (resumed.magic) {
    e.spawner = resumed.spawner;
    e.delegate.child_count = resumed.child_count;
    memcpy(e.delegate.children, resumed.children, sizeof resumed.children);
  }
  unsigned stdio_mask;
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
    fprintf(stderr, "capstone-exec: %s: %s (requires application ABI v1 or v2)\n",
            e.path, strerror(error));
    return error == ENOENT ? 127 : 126;
  }
  e.delegated = (descriptor.v1.flags & CAPSTONE_APPLICATION_DELEGATE) != 0;
  if (e.delegated && capstone_application_hash(e.image, e.delegate.image_sha256))
    return fail(&e, "capstone-exec: image hash", 1);
  char *cwd = getcwd(NULL, 0);
  void *startup = calloc(1, CAPSTONE_LAUNCH_BYTES);
  int first_arg = app_args ? 4 : binfmt ? 2 : 1;
  int error = !cwd || !startup ? ENOMEM : capstone_launch_pack(startup,
      CAPSTONE_LAUNCH_BYTES, argc - first_arg, argv + first_arg, environ, cwd, stdio_mask);
  free(cwd);
  if (error) {
    fprintf(stderr, "capstone-exec: startup: %s\n", strerror(error));
    free(startup);
    return fail(&e, "capstone-exec: startup", 0);
  }
  if (e.delegated) {
    /* Fork before device setup: the helper inherits no domain mappings. */
    int spawn_error = e.spawner.socket >= 0 ? 0 : capstone_spawner_start(&e.spawner);
    if (spawn_error) {
      errno = spawn_error;
      return fail(&e, "capstone-exec: spawner", 1);
    }
    e.delegate.spawner = &e.spawner;
  }
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
    return fail(&e, "capstone-exec: cannot create domain", 0);
  }
  e.sizes[REGION_META] = HC_V0_REGION_SIZE;
  e.sizes[REGION_DATA] = e.delegated ? (size_t)descriptor.exchange_bytes : HC_V0_REGION_SIZE;
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
        fault(&e, NULL);
      free(startup);
      return fail(&e, "capstone-exec: share launch region", 1);
    }
  }
  free(startup);
  if (descriptor.v1.heap_bytes) {
    region_id_t heap = create_region((unsigned long)descriptor.v1.heap_bytes);
    if ((long)heap < 0)
      return fail(&e, "capstone-exec: cannot allocate application heap", 0);
    if (capstone_share(domain, heap, 1, 3)) {
      if (errno == EFAULT)
        fault(&e, NULL);
      return fail(&e, "capstone-exec: share heap", 1);
    }
  }
  if (e.delegated) {
    e.delegate.exchange = e.maps[REGION_DATA];
    e.delegate.exchange_bytes = e.sizes[REGION_DATA];
    if (!getenv("CAPSTONE_EXEC_NO_SECCOMP")) {
      int rc = capstone_delegate_seccomp();
      if (rc) {
        errno = rc;
        return fail(&e, "capstone-exec: seccomp filter", 1);
      }
    }
  }
  if (e.delegated)
    for (int fd = 0; fd < 3; ++fd)
      if (!(stdio_mask & (1u << fd))) close(fd);
  e.ticks_start = ticks();
  struct hostcall_v0 *metadata = e.maps[REGION_META];
  char *payload = e.maps[REGION_DATA];
  for (;;) {
    struct ioctl_dom_step_args step;
    if (capstone_step(domain, &step))
      return fail(&e, "capstone-exec: enter domain", 1);
    if (step.event == CAPSTONE_STEP_PREEMPTED)
      continue;
    if (step.event == CAPSTONE_STEP_FAULT)
      fault(&e, &step);
    if (e.delegated) {
      struct capstone_delegate_entry *entry = e.maps[REGION_META];
      /* A return with no request means the domain left its entry instead of
         yielding: re-entering would restart it from the top, forever. */
      if (entry->version != CAPSTONE_DELEGATE_VERSION) {
        fprintf(stderr, "capstone-exec: domain returned without a request (entry version %u, "
                "rounds so far %llu)\n", entry->version, (unsigned long long)e.delegate.rounds);
        report_stats(&e);
        return fail(&e, "capstone-exec: invalid runtime state", 0);
      }
      capstone_delegate_serve(&e.delegate, entry);
      if (e.delegate.exec_requested)
        entry->result = exec_in_place(&e);
      if (e.delegate.exiting) {
        int status = e.delegate.exit_status;
        report_stats(&e);
        cleanup(&e);
        return status;
      }
      continue;
    }
    struct hostcall_v0 request;
    hostcall_snapshot_request(&request, metadata);
    if (request.phase == HC_V0_PHASE_DONE) {
      int status = (unsigned char)request.result;
      cleanup(&e);
      return status;
    }
    if (request.phase != HC_V0_PHASE_REQ || !hostcall_payload_range_valid(&request)) {
      fprintf(stderr, "capstone-exec: invalid runtime state (phase=%llu, result=%lld)\n",
              request.phase, request.result);
      return fail(&e, "capstone-exec: invalid runtime state", 0);
    }
    if (capstone_application_service(&stdio_mask, &request, metadata, payload) < 0 &&
        hc_host_service(&e.host, &request, metadata, payload) < 0)
      hc_host_error(metadata, ENOSYS);
    metadata->phase = HC_V0_PHASE_RESP;
  }
}
