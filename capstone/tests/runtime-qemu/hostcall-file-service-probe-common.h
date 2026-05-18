#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_HOSTCALL_FILE_SERVICE_PROBE_COMMON_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_HOSTCALL_FILE_SERVICE_PROBE_COMMON_H

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "hostcall-stdout-probe/hostcall_stdout_probe.h"

#define HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES 8ULL

struct hostcall_file_service_handle_slot {
  int in_use;
  int fd;
};

static inline void hostcall_snapshot_request(struct hostcall_v0 *snapshot,
                                             const struct hostcall_v0 *metadata) {
  *snapshot = *metadata;
}

static inline int
hostcall_payload_range_valid(const struct hostcall_v0 *request) {
  if (request->offset > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
    return 0;
  if (request->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE - request->offset)
    return 0;
  return 1;
}

static inline void hostcall_cleanup_open_handles(
    struct hostcall_file_service_handle_slot *slots, hostcall_u64_t slot_count) {
  hostcall_u64_t slot_i;

  for (slot_i = 0; slot_i < slot_count; ++slot_i) {
    if (slots[slot_i].in_use) {
      close(slots[slot_i].fd);
      slots[slot_i].fd = -1;
      slots[slot_i].in_use = 0;
    }
  }
}

static inline hostcall_u64_t hostcall_allocate_handle_token(
    struct hostcall_file_service_handle_slot *slots, hostcall_u64_t slot_count,
    int fd) {
  hostcall_u64_t slot_i;

  for (slot_i = 0; slot_i < slot_count; ++slot_i) {
    if (!slots[slot_i].in_use) {
      slots[slot_i].in_use = 1;
      slots[slot_i].fd = fd;
      return slot_i + 1;
    }
  }

  errno = EMFILE;
  return 0;
}

static inline int hostcall_lookup_handle_fd(
    struct hostcall_file_service_handle_slot *slots, hostcall_u64_t slot_count,
    hostcall_u64_t token) {
  if (token == 0 || token > slot_count) {
    errno = EBADF;
    return -1;
  }
  if (!slots[token - 1].in_use) {
    errno = EBADF;
    return -1;
  }
  return slots[token - 1].fd;
}

static inline int hostcall_close_handle_token(
    struct hostcall_file_service_handle_slot *slots, hostcall_u64_t slot_count,
    hostcall_u64_t token) {
  struct hostcall_file_service_handle_slot *slot;
  int fd;

  if (token == 0 || token > slot_count) {
    errno = EBADF;
    return -1;
  }

  slot = &slots[token - 1];
  if (!slot->in_use) {
    errno = EBADF;
    return -1;
  }

  fd = slot->fd;
  slot->fd = -1;
  slot->in_use = 0;
  if (close(fd) < 0)
    return -1;
  return 0;
}

static inline ssize_t hostcall_write_full_at_offset(int fd, const char *buffer,
                                                    size_t len,
                                                    off_t file_offset) {
  size_t written = 0;

  if (lseek(fd, file_offset, SEEK_SET) < 0)
    return -1;

  while (written < len) {
    ssize_t rc = write(fd, buffer + written, len - written);
    if (rc < 0)
      return -1;
    if (rc == 0) {
      errno = EIO;
      return -1;
    }
    written += (size_t)rc;
  }

  return (ssize_t)written;
}

static inline ssize_t hostcall_read_into_buffer_at_offset(int fd, char *buffer,
                                                          size_t len,
                                                          off_t file_offset) {
  if (lseek(fd, file_offset, SEEK_SET) < 0)
    return -1;
  return read(fd, buffer, len);
}

#endif

