/* The launcher side of the delegated syscall ABI v2. One entry in, the real
 * Linux syscall run by this task, the result written back. Policy is the
 * seccomp filter installed from the same shape table. */
#ifndef CAPSTONE_LINUX_DELEGATE_SERVICE_H
#define CAPSTONE_LINUX_DELEGATE_SERVICE_H

#include "capstone/delegate.h"
#include "signals.h"
#include "spawner.h"
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define CAPSTONE_DELEGATE_CHILDREN 256

struct capstone_delegate_host {
  char *exchange;          /* the launcher's mapping of the exchange region */
  size_t exchange_bytes;
  /* Ordinary memory mirroring the exchange region. The region is a
     page-frame mapping the 9p transport cannot pin, so a large read from the
     share fails with EFAULT when the kernel is handed the mapping itself;
     buffer arguments go through here instead, one copy each way. Allocated
     on first use; allocation failure returns ENOMEM. */
  char *bounce;
  struct capstone_spawner *spawner;   /* NULL: spawn answers ENOSYS */
  pid_t children[CAPSTONE_DELEGATE_CHILDREN];
  unsigned child_count;
  int private_fds[8];       /* launcher resources, never application descriptors */
  unsigned private_count;
  /* set by an exec request: the launcher replaces itself with this image */
  int exec_requested;
  char exec_block[65536];
  size_t exec_bytes;
  /* from HELLO, for fault records */
  uint64_t entry_address, code_base, code_end;
  int hello_seen;
  char image_sha256[65];
  /* counters, and the requests around a fault */
  uint64_t rounds, syscalls, refused, bytes_in, bytes_out;
  uint64_t last_nr, preparing_nr;
  /* set by exit or exit_group: the process must end with this status */
  int exiting;
  int exit_status;
  /* signals: the ring, the classes, the masks; initialized by the launcher */
  struct capstone_signal_state signals;
  /* contexts (docs/plans/delegation-threads.md): the launcher answers the
     CONTEXT requests through this hook; NULL answers ENOSYS. It writes a step
     event, when asked for one, into the exchange region at the offset in
     args[2]. */
  long (*context)(struct capstone_delegate_host *host, const struct capstone_delegate_entry *request);
  void *context_state;
  /* A further context's host (docs/plans/delegation-threads.md) serves that
     context's own transport, counters and exec request, and shares the rest
     with the first context's host, `owner`: the spawner, the children, the
     private descriptors, HELLO's code range and the signal dispositions. NULL
     for the first context. `lock` is the owner's, for the children and the
     spawner. A further context's signal requests answer ENOSYS: signals stay
     with the first context until they are per context. */
  struct capstone_delegate_host *owner;
  pthread_mutex_t lock;
  uint64_t context_id;      /* the context this host serves */
};

/* Service one entry in place: validate, run, write result. Never returns an
 * error to the caller; every failure is a negative errno in entry->result.
 * After it returns with host->exiting set, the caller ends the process. */
void capstone_delegate_serve(struct capstone_delegate_host *host,
                             struct capstone_delegate_entry *entry);

/* Release what the host allocated: the bounce mirror. */
void capstone_delegate_host_free(struct capstone_delegate_host *host);

/* Install the seccomp allowlist: every delegated shape plus what the launcher
 * needs for itself. Returns 0, or errno when the kernel refuses; the caller
 * must fail launch unless filtering was explicitly disabled. */
int capstone_delegate_seccomp(void);

/* The fault record line, written without blocking. `image` may be NULL. */
void capstone_delegate_fault_record(int fd, const struct capstone_delegate_host *host,
                                    const char *image, uint64_t cause, uint64_t pc,
                                    uint64_t address);

#endif
