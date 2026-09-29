/* Native tests for the delegated syscall wire block: shape table, packing,
 * bounds validation against an exchange region, and the exception groups. */
#include "capstone/delegate.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define EXCHANGE 4096

static struct capstone_delegate_entry pack(uint64_t nr, uint64_t a, uint64_t b,
                                           uint64_t c, uint64_t d, uint64_t e,
                                           uint64_t f) {
  struct capstone_delegate_entry entry;
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&entry, nr, args));
  return entry;
}

static void wire_layout(void) {
  struct capstone_delegate_entry e;
  assert(sizeof e == 88);
  assert(offsetof(struct capstone_delegate_entry, nr) == 8);
  assert(offsetof(struct capstone_delegate_entry, args) == 16);
  assert(offsetof(struct capstone_delegate_entry, flags) == 64);
  assert(offsetof(struct capstone_delegate_entry, result) == 72);
  assert(offsetof(struct capstone_delegate_entry, pending) == 80);
}

static void groups(void) {
  assert(capstone_delegate_group_of(CAPSTONE_SYS_read) == CAPSTONE_GROUP_DELEGATED);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_mmap) == CAPSTONE_GROUP_MEMORY);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_fork) == CAPSTONE_GROUP_PROCESS);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_clone) == CAPSTONE_GROUP_PROCESS);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_rt_sigaction) == CAPSTONE_GROUP_SIGNAL);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_socket) == CAPSTONE_GROUP_UNKNOWN);
  assert(capstone_delegate_group_of(999999) == CAPSTONE_GROUP_UNKNOWN);
  /* exit_group and wait4 are task-model calls that pass through */
  assert(capstone_delegate_group_of(CAPSTONE_SYS_exit_group) == CAPSTONE_GROUP_DELEGATED);
  assert(capstone_delegate_group_of(CAPSTONE_SYS_wait4) == CAPSTONE_GROUP_DELEGATED);
  /* an excepted or unknown number cannot be packed */
  struct capstone_delegate_entry e;
  uint64_t z[CAPSTONE_DELEGATE_ARGS] = {0};
  assert(capstone_delegate_pack(&e, CAPSTONE_SYS_mmap, z) == EINVAL);
  assert(capstone_delegate_pack(&e, CAPSTONE_SYS_socket, z) == EINVAL);
}

static void flags_follow_the_shape(void) {
  struct capstone_delegate_entry e = pack(CAPSTONE_SYS_write, 1, 100, 7, 0, 0, 0);
  assert(e.version == CAPSTONE_DELEGATE_VERSION && e.count == 1);
  assert(e.flags == 0x2); /* only the buffer */
  e = pack(CAPSTONE_SYS_openat, (uint64_t)-100, 64, 0, 0, 0, 0);
  assert(e.flags == 0x2); /* the path string */
  e = pack(CAPSTONE_SYS_renameat2, 1, 0, 1, 32, 0, 0);
  assert(e.flags == 0xa); /* both paths; offset 0 is a valid string offset */
  e = pack(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  assert(e.flags == 0);
  /* optional buffers: NULL is not flagged, non-NULL is */
  e = pack(CAPSTONE_SYS_nanosleep, 0, 0, 0, 0, 0, 0);
  assert(e.flags == 0x1);
  e = pack(CAPSTONE_SYS_nanosleep, 0, 16, 0, 0, 0, 0);
  assert(e.flags == 0x3);
}

static void bounds(void) {
  struct capstone_delegate_entry e;
  /* a 7-byte write at offset 100 fits; at the last byte it does not */
  e = pack(CAPSTONE_SYS_write, 1, 100, 7, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_write, 1, EXCHANGE - 7, 7, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_write, 1, EXCHANGE - 6, 7, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  e = pack(CAPSTONE_SYS_write, 1, EXCHANGE, 0, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  /* a length that wraps */
  e = pack(CAPSTONE_SYS_read, 0, 8, (uint64_t)-1, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  /* fixed-size out buffers */
  e = pack(CAPSTONE_SYS_fstat, 3, EXCHANGE - 128, 0, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_fstat, 3, EXCHANGE - 127, 0, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  /* scaled: 3 iovecs of 16 bytes */
  e = pack(CAPSTONE_SYS_writev, 1, EXCHANGE - 48, 3, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_writev, 1, EXCHANGE - 47, 3, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  e = pack(CAPSTONE_SYS_writev, 1, 0, (uint64_t)-1, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
  /* an optional NULL passes without a flag; a required NULL is EINVAL */
  e = pack(CAPSTONE_SYS_nanosleep, 0, 0, 0, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_read, 0, 0, 16, 0, 0, 0);
  e.flags = 0;
  assert(capstone_delegate_validate(&e, EXCHANGE) == EINVAL);
  /* a flag on an integer argument is malformed, not a fault */
  e = pack(CAPSTONE_SYS_close, 3, 0, 0, 0, 0, 0);
  e.flags = 0x1;
  assert(capstone_delegate_validate(&e, EXCHANGE) == EINVAL);
  /* version, count, high flag bits */
  e = pack(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  e.version = 1;
  assert(capstone_delegate_validate(&e, EXCHANGE) == EINVAL);
  e = pack(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  e.count = 2;
  assert(capstone_delegate_validate(&e, EXCHANGE) == EINVAL);
  e = pack(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  e.flags = UINT64_C(1) << 40;
  assert(capstone_delegate_validate(&e, EXCHANGE) == EINVAL);
  /* an excepted number arriving on the wire is ENOSYS, not a crash */
  memset(&e, 0, sizeof e);
  e.version = CAPSTONE_DELEGATE_VERSION;
  e.count = 1;
  e.nr = CAPSTONE_SYS_clone;
  assert(capstone_delegate_validate(&e, EXCHANGE) == ENOSYS);
  e.nr = CAPSTONE_SYS_socket;
  assert(capstone_delegate_validate(&e, EXCHANGE) == ENOSYS);
}

static void strings(void) {
  char exchange[EXCHANGE];
  memset(exchange, 'x', sizeof exchange);
  exchange[10] = 0;
  assert(capstone_delegate_string_ok(exchange, EXCHANGE, 0));
  assert(capstone_delegate_string_ok(exchange, EXCHANGE, 10));
  assert(!capstone_delegate_string_ok(exchange, EXCHANGE, 11)); /* no NUL after */
  assert(!capstone_delegate_string_ok(exchange, EXCHANGE, EXCHANGE));
  assert(!capstone_delegate_string_ok(NULL, EXCHANGE, 0));
  /* a string offset is validated by the launcher, not by the shape: the
     validator only checks the offset is inside the region */
  struct capstone_delegate_entry e = pack(CAPSTONE_SYS_chdir, EXCHANGE - 1, 0, 0, 0, 0, 0);
  assert(!capstone_delegate_validate(&e, EXCHANGE));
  e = pack(CAPSTONE_SYS_chdir, EXCHANGE, 0, 0, 0, 0, 0);
  assert(capstone_delegate_validate(&e, EXCHANGE) == EFAULT);
}

static void table_is_consistent(void) {
  /* every delegated shape's buffer lengths refer to an existing argument */
  static const uint16_t numbers[] = {
    CAPSTONE_SYS_getcwd, CAPSTONE_SYS_read, CAPSTONE_SYS_write, CAPSTONE_SYS_readv,
    CAPSTONE_SYS_writev, CAPSTONE_SYS_pread64, CAPSTONE_SYS_pwrite64,
    CAPSTONE_SYS_getdents64, CAPSTONE_SYS_readlinkat, CAPSTONE_SYS_ppoll,
    CAPSTONE_SYS_getrandom};
  for (size_t n = 0; n < sizeof numbers / sizeof numbers[0]; ++n) {
    const struct capstone_delegate_shape *s = capstone_delegate_shape(numbers[n]);
    assert(s && s->name);
    for (unsigned i = 0; i < s->argc; ++i) {
      const struct capstone_delegate_arg *a = &s->args[i];
      if (a->length == CAPSTONE_LEN_ARG || a->length == CAPSTONE_LEN_ARG_SCALED) {
        assert(a->size < s->argc);
        assert(s->args[a->size].kind == CAPSTONE_ARG_INT);
      }
      if (a->length == CAPSTONE_LEN_ARG_SCALED)
        assert(a->scale);
    }
  }
}

int main(void) {
  wire_layout();
  groups();
  flags_follow_the_shape();
  bounds();
  strings();
  table_is_consistent();
  puts("delegate-test: ok");
  return 0;
}
