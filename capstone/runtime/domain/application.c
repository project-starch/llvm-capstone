#include "capstone/launch.h"
#include <errno.h>
#include <string.h>
#include <unistd.h>

#ifndef CAPSTONE_APPLICATION_HEAP_BYTES
#define CAPSTONE_APPLICATION_HEAP_BYTES 0
#endif

__attribute__((used, section(".capstone_application")))
static const struct capstone_application_descriptor descriptor = {
    CAPSTONE_APPLICATION_MAGIC, CAPSTONE_LAUNCH_VERSION,
    CAPSTONE_APPLICATION_RECOVERY, CAPSTONE_LAUNCH_BYTES,
    CAPSTONE_APPLICATION_HEAP_BYTES};

static char storage[CAPSTONE_LAUNCH_BYTES] __attribute__((aligned(16)));
static char *arguments[CAPSTONE_LAUNCH_STRINGS + 1];
static char *environment[CAPSTONE_LAUNCH_STRINGS + 1];
static struct capstone_launch_view launch;
static int prepared;

int __capstone_application_prepare(const void *region, size_t bytes) {
  if (prepared || !region || bytes < sizeof(struct capstone_launch_header))
    return EINVAL;
  struct capstone_launch_header h;
  memcpy(&h, region, sizeof h);
  if (h.bytes > bytes || h.bytes > sizeof storage || h.bytes < sizeof h)
    return EINVAL;
  memcpy(storage, region, h.bytes);
  int error = capstone_launch_unpack(storage, h.bytes, arguments,
      CAPSTONE_LAUNCH_STRINGS + 1, environment, CAPSTONE_LAUNCH_STRINGS + 1, &launch);
  if (error)
    return error;
  if (chdir(launch.cwd))
    return errno;
  prepared = 1;
  return 0;
}

unsigned __capstone_application_stdio(void) { return launch.stdio_mask; }
char **__capstone_domain_environ(void) { return environment; }

extern int main(int, char **);
int capstone_main(void) { return main((int)launch.argc, arguments); }
