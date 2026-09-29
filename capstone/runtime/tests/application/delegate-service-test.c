/* Native test of the launcher's dispatcher: entries in, real syscalls out,
 * through an exchange buffer, with the failures the wire ABI promises. */
#include "../../linux/delegate-service.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define EXCHANGE 4096
static char exchange[EXCHANGE];
static struct capstone_delegate_host host = {.exchange = exchange, .exchange_bytes = EXCHANGE};

static struct capstone_delegate_entry entry(uint64_t nr, uint64_t a, uint64_t b, uint64_t c,
                                            uint64_t d, uint64_t e, uint64_t f) {
  struct capstone_delegate_entry x;
  uint64_t args[CAPSTONE_DELEGATE_ARGS] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&x, nr, args));
  return x;
}

static long serve(struct capstone_delegate_entry *x) {
  capstone_delegate_serve(&host, x);
  return (long)x->result;
}

int main(void) {
  struct capstone_delegate_entry x;
  /* an integer-only call */
  x = entry(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  assert(serve(&x) == getpid());
  /* HELLO is answered by the launcher itself */
  x = entry(CAPSTONE_NR_HELLO, 0x80001234, 0x80000000, 0x80010000, 0, 0, 0);
  assert(serve(&x) == 0 && host.hello_seen && host.entry_address == 0x80001234);
  /* write through the exchange region to a pipe, then read it back */
  int fds[2];
  assert(!pipe(fds));
  memcpy(exchange + 64, "hello\n", 6);
  x = entry(CAPSTONE_SYS_write, (uint64_t)fds[1], 64, 6, 0, 0, 0);
  assert(serve(&x) == 6);
  x = entry(CAPSTONE_SYS_read, (uint64_t)fds[0], 128, 32, 0, 0, 0);
  assert(serve(&x) == 6 && !memcmp(exchange + 128, "hello\n", 6));
  assert(host.bytes_in == 6 && host.bytes_out == 32);
  /* a string argument: openat of a path in the exchange region */
  char path[] = "/tmp/capstone-delegate-XXXXXX";
  int tmp = mkstemp(path);
  assert(tmp >= 0);
  strcpy(exchange + 256, path);
  x = entry(CAPSTONE_SYS_openat, (uint64_t)-100, 256, O_RDONLY, 0, 0, 0);
  long fd = serve(&x);
  assert(fd >= 0);
  x = entry(CAPSTONE_SYS_fstat, (uint64_t)fd, 512, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  {
    struct stat st;
    memcpy(&st, exchange + 512, sizeof st < 128 ? sizeof st : 128);
    (void)st;
  }
  x = entry(CAPSTONE_SYS_close, (uint64_t)fd, 0, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  /* an unterminated string is EFAULT before any syscall runs */
  memset(exchange + EXCHANGE - 8, 'x', 8);
  x = entry(CAPSTONE_SYS_chdir, EXCHANGE - 8, 0, 0, 0, 0, 0);
  assert(serve(&x) == -EFAULT);
  /* an offset outside the region, a wrapped length, a bad version */
  x = entry(CAPSTONE_SYS_write, 1, EXCHANGE, 1, 0, 0, 0);
  assert(serve(&x) == -EFAULT);
  x = entry(CAPSTONE_SYS_read, (uint64_t)fds[0], 0, (uint64_t)-1, 0, 0, 0);
  assert(serve(&x) == -EFAULT);
  x = entry(CAPSTONE_SYS_getpid, 0, 0, 0, 0, 0, 0);
  x.version = 7;
  assert(serve(&x) == -EINVAL);
  /* excepted and unknown numbers never reach the kernel */
  memset(&x, 0, sizeof x);
  x.version = CAPSTONE_DELEGATE_VERSION;
  x.count = 1;
  x.nr = CAPSTONE_SYS_execve;
  assert(serve(&x) == -ENOSYS);
  x.nr = 999999;
  assert(serve(&x) == -ENOSYS);
  /* a signal to another process is refused here */
  x = entry(CAPSTONE_SYS_kill, 1, 0, 0, 0, 0, 0);
  assert(serve(&x) == -EPERM);
  x = entry(CAPSTONE_SYS_kill, (uint64_t)getpid(), 0, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  /* exit_group ends the run, and does not run */
  x = entry(CAPSTONE_SYS_exit_group, 42, 0, 0, 0, 0, 0);
  assert(serve(&x) == 0 && host.exiting && host.exit_status == 42);
  /* five refused by the validator; the string and kill refusals are the runnerâs */
  assert(host.refused == 5 && host.rounds == 16 && host.syscalls == 9);
  close(tmp);
  unlink(path);
  /* the fault record writes without blocking, even to a full pipe */
  int full[2];
  assert(!pipe(full));
  {
    int flags = fcntl(full[1], F_GETFL);
    fcntl(full[1], F_SETFL, flags | O_NONBLOCK);
    char junk[65536];
    memset(junk, 0, sizeof junk);
    while (write(full[1], junk, sizeof junk) > 0)
      ;
    fcntl(full[1], F_SETFL, flags);
  }
  capstone_delegate_fault_record(full[1], &host, "image.dom", 24, 0x80001300, 0x1234);
  {
    char line[512];
    int out[2];
    assert(!pipe(out));
    capstone_delegate_fault_record(out[1], &host, "image.dom", 24, 0x80001300, 0x1234);
    long n = read(out[0], line, sizeof line - 1);
    assert(n > 0);
    line[n] = 0;
    assert(strstr(line, "cause=24 pc=0x80001300 address=0x1234 entry=0x80001234"));
    assert(strstr(line, "code=0x80000000-0x80010000 image=image.dom"));
  }
  puts("delegate-service-test: ok");
  return 0;
}
