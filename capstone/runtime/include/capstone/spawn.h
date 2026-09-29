/* The spawn request of the delegated runtime: posix_spawn crossing to the task.
 *
 * One block in the exchange region, little-endian, offsets never pointers:
 * a header, then the file actions, then the strings. The launcher hands it
 * to its unfiltered spawner, which forks the child as the launcher's child,
 * applies the actions and execs; a Capstone image is started through the
 * launcher itself. The request number is CAPSTONE_NR_SPAWN with
 * args[0] the block's exchange offset and args[1] its size.
 */
#ifndef CAPSTONE_SPAWN_H
#define CAPSTONE_SPAWN_H

#include <stddef.h>
#include <stdint.h>

#define CAPSTONE_NR_SPAWN UINT64_C(0xC0DE0002)
#define CAPSTONE_SPAWN_VERSION 1u
#define CAPSTONE_SPAWN_BYTES 65536u
#define CAPSTONE_SPAWN_STRINGS 1024u
#define CAPSTONE_SPAWN_ACTIONS 64u

/* header flags */
#define CAPSTONE_SPAWN_SEARCH_PATH 1u   /* posix_spawnp: resolve through PATH */
#define CAPSTONE_SPAWN_SETPGROUP 2u     /* setpgid(0, pgroup) in the child */
#define CAPSTONE_SPAWN_SETSID 4u
#define CAPSTONE_SPAWN_EXEC 8u          /* replace the caller: execve, not spawn */

/* file action commands, musl's numbering */
#define CAPSTONE_SPAWN_CLOSE 1u
#define CAPSTONE_SPAWN_DUP2 2u
#define CAPSTONE_SPAWN_OPEN 3u
#define CAPSTONE_SPAWN_CHDIR 4u
#define CAPSTONE_SPAWN_FCHDIR 5u

struct capstone_spawn_header {
  unsigned char magic[8];   /* "CPSPAWN1" */
  uint32_t version, bytes, flags, pgroup;
  uint32_t path, argc, envc, actions;  /* path: string offset; counts */
  /* then: uint32_t argv[argc], envp[envc]; struct capstone_spawn_action[actions]; strings */
};

struct capstone_spawn_action {
  uint32_t cmd, fd, srcfd, oflag, mode, path; /* path: string offset or 0 */
};

struct capstone_spawn_view {
  uint32_t flags, pgroup, argc, envc, actions;
  const char *path;
  const struct capstone_spawn_action *action; /* actions entries */
};

/* Errno values, zero on success; no allocation or global state. */
int capstone_spawn_pack(void *buffer, size_t capacity, uint32_t flags, uint32_t pgroup,
                        const char *path, char *const argv[], char *const envp[],
                        const struct capstone_spawn_action *actions, unsigned action_count,
                        const char *const action_paths[], size_t *bytes);
int capstone_spawn_unpack(const void *buffer, size_t bytes, char **argv, size_t argv_slots,
                          char **envp, size_t env_slots, const char **action_paths,
                          size_t action_slots, struct capstone_spawn_view *view);

#endif
