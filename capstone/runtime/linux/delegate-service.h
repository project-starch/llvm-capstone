/* The launcher side of the delegated syscall ABI v2. One entry in, the real
 * Linux syscall run by this task, the result written back. Policy is the
 * seccomp filter installed from the same shape table. */
#ifndef CAPSTONE_LINUX_DELEGATE_SERVICE_H
#define CAPSTONE_LINUX_DELEGATE_SERVICE_H

#include "capstone/delegate.h"
#include <stddef.h>
#include <stdint.h>

struct capstone_delegate_host {
  char *exchange;          /* the launcher's mapping of the exchange region */
  size_t exchange_bytes;
  /* from HELLO, for fault records */
  uint64_t entry_address, code_base, code_end;
  int hello_seen;
  /* counters */
  uint64_t rounds, syscalls, refused, bytes_in, bytes_out;
  /* set by exit or exit_group: the process must end with this status */
  int exiting;
  int exit_status;
};

/* Service one entry in place: validate, run, write result. Never returns an
 * error to the caller; every failure is a negative errno in entry->result.
 * After it returns with host->exiting set, the caller ends the process. */
void capstone_delegate_serve(struct capstone_delegate_host *host,
                             struct capstone_delegate_entry *entry);

/* Install the seccomp allowlist: every delegated shape plus what the launcher
 * needs for itself. Returns 0, or errno when the kernel refuses; the caller
 * decides whether to continue without a filter. */
int capstone_delegate_seccomp(void);

/* The fault record line, written without blocking. `image` may be NULL. */
void capstone_delegate_fault_record(int fd, const struct capstone_delegate_host *host,
                                    const char *image, uint64_t cause, uint64_t pc,
                                    uint64_t address);

#endif
