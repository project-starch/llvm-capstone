#include "capstone/launch.h"
#include <errno.h>
#include <string.h>

#ifndef CAPSTONE_APPLICATION_HEAP_BYTES
#define CAPSTONE_APPLICATION_HEAP_BYTES 0
#endif

#include "capstone/delegate.h"
#ifndef CAPSTONE_APPLICATION_EXCHANGE_BYTES
#define CAPSTONE_APPLICATION_EXCHANGE_BYTES CAPSTONE_DELEGATE_DEFAULT_EXCHANGE
#endif
/* Delegated applications always declare the exchange region (ABI v2). */
__attribute__((used, section(".capstone_application")))
static const struct capstone_application_descriptor_v2 descriptor = {
    {CAPSTONE_APPLICATION_MAGIC, CAPSTONE_LAUNCH_VERSION,
     CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE,
     CAPSTONE_LAUNCH_BYTES, CAPSTONE_APPLICATION_HEAP_BYTES},
    CAPSTONE_APPLICATION_EXCHANGE_BYTES};

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
  /* No chdir: the domain is the user half of the task that packed this block,
     and that task already runs in launch.cwd. */
  prepared = 1;
  return 0;
}

const struct capstone_launch_task *__capstone_launch_task(void) {
  return prepared ? &launch.task : NULL;
}

char **__capstone_domain_environ(void) { return environment; }

extern int main(int, char **);
int capstone_main(void) { return main((int)launch.argc, arguments); }
