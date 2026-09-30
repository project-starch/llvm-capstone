/* Native test of the launcher's dispatcher: entries in, real syscalls out,
 * through an exchange buffer, with the failures the wire ABI promises. */
#include "../../linux/delegate-service.h"
#include "capstone/spawn.h"
#include <sys/wait.h>
#include <signal.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
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

static long context_hook(struct capstone_delegate_host *h, const struct capstone_delegate_entry *r) {
  (void)h;
  return 1000 + (long)(r->nr & 0xff);
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
  /* Two independent string offsets, and symlinkat's dirfd is argument 1. */
  char link_path[128], target[128];
  snprintf(link_path, sizeof link_path, "%s-link", path);
  strcpy(exchange + 1024, link_path);
  host.private_fds[0] = tmp;
  host.private_count = 1;
  x = entry(CAPSTONE_SYS_symlinkat, 256, (uint64_t)tmp, 1024, 0, 0, 0);
  assert(serve(&x) == -EBADF);
  host.private_count = 0;
  x = entry(CAPSTONE_SYS_symlinkat, 256, (uint64_t)AT_FDCWD, 1024, 0, 0, 0);
  assert(serve(&x) == 0);
  ssize_t target_length = readlink(link_path, target, sizeof target);
  assert(target_length == (ssize_t)strlen(path) && !memcmp(target, path, target_length));
  assert(unlink(link_path) == 0);
  x = entry(CAPSTONE_SYS_sync_file_range, (uint64_t)tmp, 0, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  x = entry(CAPSTONE_SYS_flock, (uint64_t)tmp, LOCK_EX | LOCK_NB, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  int competing = open(path, O_RDWR);
  assert(competing >= 0);
  assert(flock(competing, LOCK_EX | LOCK_NB) == -1 && errno == EWOULDBLOCK);
  x = entry(CAPSTONE_SYS_flock, (uint64_t)tmp, LOCK_UN, 0, 0, 0, 0);
  assert(serve(&x) == 0 && flock(competing, LOCK_EX | LOCK_NB) == 0);
  close(competing);
  x = entry(CAPSTONE_SYS_fchmodat, (uint64_t)AT_FDCWD, 256, 0640, 0, 0, 0);
  assert(serve(&x) == 0);
  struct stat changed_mode;
  assert(stat(path, &changed_mode) == 0 && (changed_mode.st_mode & 0777) == 0640);
  x = entry(CAPSTONE_SYS_fstat, (uint64_t)fd, 512, 0, 0, 0, 0);
  assert(serve(&x) == 0);
  {
    struct stat st;
    memcpy(&st, exchange + 512, sizeof st < 128 ? sizeof st : 128);
    (void)st;
  }
  /* the pointer form of fcntl runs as fcntl with the buffer's address */
  {
    struct flock lock = {.l_type = F_WRLCK, .l_whence = SEEK_SET};
    memcpy(exchange + 768, &lock, sizeof lock);
    x = entry(CAPSTONE_NR_FCNTL_LOCK, (uint64_t)fd, F_GETLK, 768, 0, 0, 0);
    assert(serve(&x) == 0);
    memcpy(&lock, exchange + 768, sizeof lock);
    assert(lock.l_type == F_UNLCK);
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
  assert(host.refused == 5 && host.rounds == 23 && host.syscalls == 16);
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
    assert(strstr(line, "code=0x80000000-0x80010000 last=0x"));
    assert(strstr(line, " image=image.dom"));
  }
  /* spawn through the service, then wait4 through the service: the status
     must land in the exchange region where the entry pointed */
  {
    struct capstone_spawner spawner;
    struct capstone_delegate_host h2 = {.exchange = exchange, .exchange_bytes = EXCHANGE};
    char *argv[] = {"sh", "-c", "exit 3", NULL};
    char *envp[] = {"PATH=/usr/bin:/bin", NULL};
    size_t bytes;
    long pid;
    assert(!capstone_spawner_start(&spawner));
    h2.spawner = &spawner;
    assert(!capstone_spawn_pack(exchange + 1024, EXCHANGE - 1024, CAPSTONE_SPAWN_SEARCH_PATH, 0,
                                "sh", argv, envp, NULL, 0, NULL, &bytes));
    x = entry(CAPSTONE_NR_SPAWN, 1024, bytes, 0, 0, 0, 0);
    capstone_delegate_serve(&h2, &x);
    pid = (long)x.result;
    assert(pid > 0 && h2.child_count == 1);
    memset(exchange + 16, 0x66, 4);
    x = entry(CAPSTONE_SYS_wait4, (uint64_t)pid, 16, 0, 0, 0, 0);
    capstone_delegate_serve(&h2, &x);
    assert((long)x.result == pid);
    {
      int status;
      memcpy(&status, exchange + 16, sizeof status);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 3);
    }
    assert(h2.child_count == 0);
    /* a second wait for the same pid is refused: not a child any more */
    x = entry(CAPSTONE_SYS_wait4, (uint64_t)pid, 16, 0, 0, 0, 0);
    capstone_delegate_serve(&h2, &x);
    assert((long)x.result == -ECHILD);
    {
      /* A further context's host: its own exchange region, the owner's
         children and spawner, and no signal requests (delegation-threads). */
      static char other[EXCHANGE];
      struct capstone_delegate_host further = {.exchange = other, .exchange_bytes = EXCHANGE,
                                               .owner = &h2};
      assert(!capstone_spawn_pack(other + 1024, EXCHANGE - 1024, CAPSTONE_SPAWN_SEARCH_PATH, 0,
                                  "sh", argv, envp, NULL, 0, NULL, &bytes));
      x = entry(CAPSTONE_NR_SPAWN, 1024, bytes, 0, 0, 0, 0);
      capstone_delegate_serve(&further, &x);
      pid = (long)x.result;
      assert(pid > 0 && h2.child_count == 1 && further.child_count == 0);
      x = entry(CAPSTONE_SYS_wait4, (uint64_t)pid, 16, 0, 0, 0, 0);
      capstone_delegate_serve(&further, &x);
      assert((long)x.result == pid && h2.child_count == 0);
      memcpy(other + 64, "further\n", 8);
      int p2[2];
      assert(!pipe(p2));
      x = entry(CAPSTONE_SYS_write, (uint64_t)p2[1], 64, 8, 0, 0, 0);
      capstone_delegate_serve(&further, &x);
      assert((long)x.result == 8 && further.bytes_in == 8);
      close(p2[0]);
      close(p2[1]);
      uint64_t mask = 0;
      memcpy(other + 32, &mask, sizeof mask);
      struct capstone_delegate_entry refused[] = {
          entry(CAPSTONE_NR_HELLO, 1, 2, 3, 0, 0, 0),
          entry(CAPSTONE_NR_SIGACTION, SIGUSR1, CAPSTONE_SIGNAL_CAUGHT, 0, 0, 0, 0),
          entry(CAPSTONE_NR_SIGPOLL, 0, 0, 0, 0, 0, 0),
          entry(CAPSTONE_SYS_rt_sigprocmask, SIG_BLOCK, 32, 0, 8, 0, 0),
          entry(CAPSTONE_SYS_rt_sigsuspend, 32, 8, 0, 0, 0, 0),
      };
      for (unsigned i = 0; i < sizeof refused / sizeof refused[0]; ++i) {
        capstone_delegate_serve(&further, &refused[i]);
        assert((long)refused[i].result == -ENOSYS);
      }
      assert(!h2.hello_seen && further.refused == 5);
      capstone_delegate_host_free(&further);
    }
    capstone_spawner_stop(&spawner);
    capstone_delegate_host_free(&h2);
  }
  /* every context request reaches the launcher's hook; without one, ENOSYS */
  {
    struct capstone_delegate_host h3 = {.exchange = exchange, .exchange_bytes = EXCHANGE};
    const uint64_t numbers[] = {CAPSTONE_NR_CONTEXT_RESERVE, CAPSTONE_NR_CONTEXT_CREATE,
                                CAPSTONE_NR_CONTEXT_STEP, CAPSTONE_NR_CONTEXT_FORGET};
    for (unsigned i = 0; i < 4; ++i) {
      x = entry(numbers[i], 0, 0, 0, 0, 0, 0);
      capstone_delegate_serve(&h3, &x);
      assert((long)x.result == -ENOSYS);
      h3.context = context_hook;
      x = entry(numbers[i], 0, 0, 0, 0, 0, 0);
      capstone_delegate_serve(&h3, &x);
      assert((long)x.result == 1000 + (long)(numbers[i] & 0xff));
      h3.context = NULL;
    }
    capstone_delegate_host_free(&h3);
  }
  /* runtime numbers resolve by their low bits, apart from Linux's */
  assert(!strcmp(capstone_delegate_shape(CAPSTONE_NR_CONTEXT_RESERVE)->name, "context-reserve"));
  assert(!strcmp(capstone_delegate_shape(CAPSTONE_NR_CONTEXT_CREATE)->name, "context-create"));
  assert(!strcmp(capstone_delegate_shape(CAPSTONE_NR_HELLO)->name, "hello"));
  assert(capstone_delegate_shape(CAPSTONE_SYS_write)->group == CAPSTONE_GROUP_DELEGATED);
  assert(!capstone_delegate_shape(UINT64_C(0xC0DE0000) + CAPSTONE_SYS_write));
  assert(!capstone_delegate_shape(UINT64_C(0xC0DE00FF)));
  capstone_delegate_host_free(&host);
  puts("delegate-service-test: ok");
  return 0;
}
