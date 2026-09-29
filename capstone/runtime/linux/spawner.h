/* The launcher's spawner: a child forked before the seccomp filter, so the
 * programs it starts are not filtered. It forks each child as the launcher's
 * own child (CLONE_PARENT), applies the spawn block's file actions and execs.
 * The launcher passes its inheritable descriptors with every request, so
 * the child sees the table posix_spawn promises. */
#ifndef CAPSTONE_LINUX_SPAWNER_H
#define CAPSTONE_LINUX_SPAWNER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define CAPSTONE_SPAWNER_FDS 64

struct capstone_spawner {
  int socket;      /* the launcher's end; -1 when not running */
  pid_t pid;
  char self[256];  /* the launcher binary, for Capstone images */
};

/* Fork the spawner. Call before installing seccomp. Returns 0 or errno. */
int capstone_spawner_start(struct capstone_spawner *s);
void capstone_spawner_stop(struct capstone_spawner *s);

/* Start a child from a spawn block. `fds` and `numbers` name the launcher's
 * descriptors and the numbers the child must see them at; bit i of `cloexec`
 * says descriptor i closes on exec, so a file action may still use it before
 * then. Returns the child's pid, or a negative errno; a child that failed to
 * exec has already been reaped. */
long capstone_spawner_spawn(struct capstone_spawner *s, const void *block, size_t bytes,
                            const int *fds, const int *numbers, uint64_t cloexec,
                            unsigned count);

/* The launcher's open descriptors except `skip`, with their close-on-exec
 * flags in `cloexec`. Returns the count or negative errno (including EMFILE on overflow);
 * numbers[i] == fds[i]. */
int capstone_spawner_descriptors(int *fds, int *numbers, uint64_t *cloexec,
                                      unsigned capacity, int skip);

/* True when the file is a Capstone domain image (ELF machine 259). */
int capstone_spawner_is_image(const char *path);

#endif
