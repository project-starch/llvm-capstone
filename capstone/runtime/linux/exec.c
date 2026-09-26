#define _GNU_SOURCE
#include "application-image.h"
#include "capstone/application-service.h"
#include "capstone/linux-domain-fault.h"
#include "host-service.h"
#include "libcapstone.h"
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>

extern char **environ;

struct execution {
  struct hc_host host;
  void *maps[3];
  size_t sizes[3];
  int image, device_open;
};

static void cleanup(void *context) {
  struct execution *e = context;
  hostcall_cleanup_open_handles(e->host.slots, HC_FILE_SERVICE_MAX_HANDLES);
  for (unsigned i = 0; i < 3; ++i)
    if (e->maps[i] && e->maps[i] != MAP_FAILED)
      munmap(e->maps[i], e->sizes[i]);
  if (e->device_open)
    capstone_cleanup();
  if (e->image >= 0)
    close(e->image);
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

int main(int argc, char **argv) {
  int literal = argc > 1 && !strcmp(argv[1], "--");
  if (literal) {
    --argc;
    ++argv;
  }
  if (argc < 2 || (!literal && !strcmp(argv[1], "--help"))) {
    fprintf(argc < 2 ? stderr : stdout, "usage: capstone-exec [--] PROGRAM [ARG...]\n");
    return argc < 2 ? 2 : 0;
  }
  struct execution e = {.image = -1};
  unsigned stdio_mask;
  if (reserve_stdio(&stdio_mask))
    return 125;
  struct capstone_application_descriptor descriptor;
  e.image = capstone_application_image(argv[1], &descriptor);
  if (e.image < 0) {
    int error = errno;
    fprintf(stderr, "capstone-exec: %s: %s (requires application ABI v1)\n",
            argv[1], strerror(error));
    return error == ENOENT ? 127 : 126;
  }
  char *cwd = getcwd(NULL, 0);
  void *startup = calloc(1, CAPSTONE_LAUNCH_BYTES);
  int error = !cwd || !startup ? ENOMEM : capstone_launch_pack(startup,
      CAPSTONE_LAUNCH_BYTES, argc - 1, argv + 1, environ, cwd, stdio_mask);
  free(cwd);
  if (error) {
    fprintf(stderr, "capstone-exec: startup: %s\n", strerror(error));
    free(startup);
    cleanup(&e);
    return 125;
  }
  capstone_set_verbose(0);
  if (capstone_init()) {
    perror("capstone-exec: device");
    free(startup);
    cleanup(&e);
    return 125;
  }
  e.device_open = 1;
  char image_path[64];
  snprintf(image_path, sizeof image_path, "/proc/self/fd/%d", e.image);
  dom_id_t domain = create_dom(image_path, NULL);
  if ((long)domain < 0) {
    fputs("capstone-exec: cannot create domain\n", stderr);
    free(startup);
    cleanup(&e);
    return 125;
  }
  e.sizes[0] = e.sizes[1] = HC_V0_REGION_SIZE;
  e.sizes[2] = CAPSTONE_LAUNCH_BYTES;
  for (unsigned i = 0; i < 3; ++i) {
    region_id_t region = create_region(e.sizes[i]);
    if ((long)region < 0 ||
        !(e.maps[i] = map_region(region, e.sizes[i])) || e.maps[i] == MAP_FAILED) {
      fputs("capstone-exec: cannot allocate launch regions\n", stderr);
      free(startup);
      cleanup(&e);
      return 125;
    }
    memset(e.maps[i], 0, e.sizes[i]);
    if (i == 2)
      memcpy(e.maps[i], startup, CAPSTONE_LAUNCH_BYTES);
    if (capstone_share(domain, region, i == 2 ? 0 : 1, 2)) {
      perror("capstone-exec: share launch region");
      free(startup);
      cleanup(&e);
      return 125;
    }
  }
  free(startup);
  if (descriptor.heap_bytes) {
    region_id_t heap = create_region((unsigned long)descriptor.heap_bytes);
    if ((long)heap < 0) {
      fputs("capstone-exec: cannot allocate application heap\n", stderr);
      cleanup(&e);
      return 125;
    }
    if (capstone_share(domain, heap, 1, 3)) {
      perror("capstone-exec: share heap");
      cleanup(&e);
      return 125;
    }
  }
  struct hostcall_v0 *metadata = e.maps[0];
  char *payload = e.maps[1];
  for (;;) {
    unsigned long result;
    if (capstone_call(domain, &result)) {
      perror("capstone-exec: enter domain");
      cleanup(&e);
      return 125;
    }
    capstone_domain_exit_on_fault(result, cleanup, &e);
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
      cleanup(&e);
      return 125;
    }
    if (capstone_application_service(&stdio_mask, &request, metadata, payload) < 0 &&
        hc_host_service(&e.host, &request, metadata, payload) < 0)
      hc_host_error(metadata, ENOSYS);
    metadata->phase = HC_V0_PHASE_RESP;
  }
}
