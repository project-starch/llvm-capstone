/* One native case per plain row: integers, strings and flat buffers through
 * the launcher's dispatcher against the build host's kernel, the answer
 * checked against the libc's own call or against the file the row touched. */
#define _GNU_SOURCE
#include "../../linux/delegate-service.h"
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static char exchange[4096];
static struct capstone_delegate_host host = {.exchange = exchange, .exchange_bytes = sizeof exchange};

static long call6(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
  struct capstone_delegate_entry entry;
  uint64_t args[6] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&entry, nr, args));
  capstone_delegate_serve(&host, &entry);
  return entry.result;
}
static long call(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return call6(nr, a, b, c, d, 0, 0);
}
/* a string placed in the exchange region, its offset returned */
static uint64_t str(uint64_t offset, const char *s) {
  strcpy(exchange + offset, s);
  return offset;
}
static int temp(char *path) {
  strcpy(path, "/tmp/capstone-rows-XXXXXX");
  int fd = mkstemp(path);
  assert(fd >= 0 && write(fd, "hello", 5) == 5);
  return fd;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *c = argv[1];
  char path[64], other[80];
  struct stat st;
  if (!strcmp(c, "statfs")) {
    struct statfs fs, mine;
    assert(sizeof fs == 120);
    assert(call(CAPSTONE_SYS_statfs, str(0, "/"), 256, 0, 0) == 0);
    memcpy(&fs, exchange + 256, sizeof fs);
    assert(statfs("/", &mine) == 0 && fs.f_bsize == mine.f_bsize && fs.f_type == mine.f_type);
    int fd = open("/", O_RDONLY);
    assert(fd >= 0);
    memset(exchange + 256, 0, sizeof fs);
    assert(call(CAPSTONE_SYS_fstatfs, fd, 256, 0, 0) == 0);
    memcpy(&fs, exchange + 256, sizeof fs);
    assert(fs.f_bsize == mine.f_bsize);
    assert(call(CAPSTONE_SYS_statfs, str(0, "/no/such/place"), 256, 0, 0) == -ENOENT);
    close(fd);
  } else if (!strcmp(c, "statx")) {
    struct statx sx;
    assert(sizeof sx == 256);
    int fd = temp(path);
    assert(call6(CAPSTONE_SYS_statx, (uint64_t)AT_FDCWD, str(0, path), 0, STATX_BASIC_STATS, 512, 0) == 0);
    memcpy(&sx, exchange + 512, sizeof sx);
    assert((sx.stx_mask & STATX_SIZE) && sx.stx_size == 5 && S_ISREG(sx.stx_mode));
    assert(call6(CAPSTONE_SYS_statx, fd, str(0, ""), AT_EMPTY_PATH, STATX_BASIC_STATS, 512, 0) == 0);
    close(fd); unlink(path);
  } else if (!strcmp(c, "truncate")) {
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_truncate, str(0, path), 2, 0, 0) == 0);
    assert(fstat(fd, &st) == 0 && st.st_size == 2);
    assert(call(CAPSTONE_SYS_truncate, str(0, path), (uint64_t)-1, 0, 0) == -EINVAL);
    close(fd); unlink(path);
  } else if (!strcmp(c, "fallocate")) {
    /* on a memory file made through the row, so the file system is known */
    long fd = call(CAPSTONE_SYS_memfd_create, str(0, "rows"), MFD_CLOEXEC, 0, 0);
    assert(fd >= 0);
    assert(call(CAPSTONE_SYS_fallocate, (uint64_t)fd, 0, 0, 8192) == 0);
    assert(fstat((int)fd, &st) == 0 && st.st_size == 8192);
    assert(call(CAPSTONE_SYS_fallocate, (uint64_t)fd, 0, 0, 0) == -EINVAL);
    close((int)fd);
  } else if (!strcmp(c, "fchdir")) {
    char before[256], now[256];
    assert(getcwd(before, sizeof before));
    int fd = open("/tmp", O_RDONLY | O_DIRECTORY);
    assert(fd >= 0 && call(CAPSTONE_SYS_fchdir, fd, 0, 0, 0) == 0);
    assert(getcwd(now, sizeof now) && !strcmp(now, "/tmp"));
    assert(chdir(before) == 0);
    close(fd);
    assert(call(CAPSTONE_SYS_fchdir, fd, 0, 0, 0) == -EBADF);
  } else if (!strcmp(c, "fchmod")) {
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_fchmod, fd, 0640, 0, 0) == 0);
    assert(fstat(fd, &st) == 0 && (st.st_mode & 07777) == 0640);
    close(fd); unlink(path);
  } else if (!strcmp(c, "fchown")) {
    /* to the owner it has: permitted without privilege, and a no-op */
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_fchown, fd, getuid(), getgid(), 0) == 0);
    assert(call6(CAPSTONE_SYS_fchownat, (uint64_t)AT_FDCWD, str(0, path), (uint64_t)-1, (uint64_t)-1, 0, 0) == 0);
    assert(call6(CAPSTONE_SYS_fchownat, (uint64_t)AT_FDCWD, str(0, "/no/such/place"), (uint64_t)-1, (uint64_t)-1, 0, 0) == -ENOENT);
    assert(fstat(fd, &st) == 0 && st.st_uid == getuid());
    close(fd); unlink(path);
  } else if (!strcmp(c, "linkat")) {
    int fd = temp(path);
    snprintf(other, sizeof other, "%s-link", path);
    assert(call6(CAPSTONE_SYS_linkat, (uint64_t)AT_FDCWD, str(0, path), (uint64_t)AT_FDCWD, str(128, other), 0, 0) == 0);
    assert(fstat(fd, &st) == 0 && st.st_nlink == 2);
    assert(call6(CAPSTONE_SYS_linkat, (uint64_t)AT_FDCWD, 0, (uint64_t)AT_FDCWD, 128, 0, 0) == -EEXIST);
    unlink(other); close(fd); unlink(path);
  } else if (!strcmp(c, "mknodat")) {
    snprintf(path, sizeof path, "/tmp/capstone-rows-fifo-%ld", (long)getpid());
    assert(call(CAPSTONE_SYS_mknodat, (uint64_t)AT_FDCWD, str(0, path), S_IFIFO | 0600, 0) == 0);
    assert(stat(path, &st) == 0 && S_ISFIFO(st.st_mode));
    assert(call(CAPSTONE_SYS_mknodat, (uint64_t)AT_FDCWD, 0, S_IFIFO | 0600, 0) == -EEXIST);
    unlink(path);
  } else if (!strcmp(c, "faccessat2")) {
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_faccessat2, (uint64_t)AT_FDCWD, str(0, path), R_OK, AT_EACCESS) == 0);
    assert(call(CAPSTONE_SYS_faccessat2, (uint64_t)AT_FDCWD, 0, X_OK, AT_EACCESS) == -EACCES);
    close(fd); unlink(path);
  } else if (!strcmp(c, "sendfile")) {
    /* the file's five bytes into a pipe, the offset word advanced */
    int fd = temp(path), p[2];
    char out[8];
    uint64_t offset = 0;
    assert(!pipe(p));
    memcpy(exchange + 64, &offset, 8);
    assert(call(CAPSTONE_SYS_sendfile, p[1], fd, 64, 5) == 5);
    memcpy(&offset, exchange + 64, 8);
    assert(offset == 5 && read(p[0], out, 8) == 5 && !memcmp(out, "hello", 5));
    assert(call(CAPSTONE_SYS_sendfile, p[1], fd, 0, 5) == 0);   /* file position is at the end: nothing */
    close(p[0]); close(p[1]); close(fd); unlink(path);
  } else if (!strcmp(c, "copy-file-range")) {
    /* both offset words given, both advanced; the files' own positions untouched */
    int in = temp(path), out = temp(other);
    uint64_t offset = 0;
    char back[16];
    assert(ftruncate(out, 0) == 0);
    memcpy(exchange + 64, &offset, 8);
    memcpy(exchange + 72, &offset, 8);
    assert(call6(CAPSTONE_SYS_copy_file_range, in, 64, out, 72, 5, 0) == 5);
    memcpy(&offset, exchange + 64, 8);
    assert(offset == 5);
    memcpy(&offset, exchange + 72, 8);
    assert(offset == 5 && pread(out, back, 16, 0) == 5 && !memcmp(back, "hello", 5));
    assert(lseek(out, 0, SEEK_CUR) == 5 && lseek(in, 0, SEEK_CUR) == 5);
    close(in); close(out); unlink(path); unlink(other);
  } else if (!strcmp(c, "readahead")) {
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_readahead, fd, 0, 4096, 0) == 0);
    assert(call(CAPSTONE_SYS_fadvise64, fd, 0, 0, POSIX_FADV_SEQUENTIAL) == 0);
    assert(call(CAPSTONE_SYS_fadvise64, fd, 0, 0, 999) == -EINVAL);
    close(fd); unlink(path);
  } else if (!strcmp(c, "sync")) {
    int fd = temp(path);
    assert(call(CAPSTONE_SYS_sync, 0, 0, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_syncfs, fd, 0, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_syncfs, -1, 0, 0, 0) == -EBADF);
    close(fd); unlink(path);
  } else if (!strcmp(c, "memfd")) {
    long fd = call(CAPSTONE_SYS_memfd_create, str(0, "rows"), MFD_CLOEXEC, 0, 0);
    assert(fd >= 0 && write((int)fd, "x", 1) == 1 && fstat((int)fd, &st) == 0 && st.st_size == 1);
    assert(fcntl((int)fd, F_GETFD) & FD_CLOEXEC);
    close((int)fd);
  } else if (!strcmp(c, "clock-getres")) {
    struct timespec ts, mine;
    assert(call(CAPSTONE_SYS_clock_getres, CLOCK_MONOTONIC, 64, 0, 0) == 0);
    memcpy(&ts, exchange + 64, sizeof ts);
    assert(clock_getres(CLOCK_MONOTONIC, &mine) == 0 && ts.tv_sec == mine.tv_sec && ts.tv_nsec == mine.tv_nsec);
    assert(call(CAPSTONE_SYS_clock_getres, CLOCK_MONOTONIC, 0, 0, 0) == 0);   /* a null result is allowed */
    assert(call(CAPSTONE_SYS_clock_getres, 9999, 64, 0, 0) == -EINVAL);
  } else if (!strcmp(c, "getgroups")) {
    gid_t mine[64], theirs[64];
    long n = call(CAPSTONE_SYS_getgroups, 0, 0, 0, 0);
    assert(n >= 0 && n == getgroups(0, NULL) && n < 64);
    memset(exchange + 64, 0xff, sizeof theirs);
    assert(call(CAPSTONE_SYS_getgroups, 64, 64, 0, 0) == n);
    assert(getgroups(64, mine) == n);
    memcpy(theirs, exchange + 64, sizeof theirs);
    assert(!memcmp(mine, theirs, (size_t)n * sizeof(gid_t)));
    assert(call(CAPSTONE_SYS_getgroups, 1, 64, 0, 0) == (n > 1 ? -EINVAL : n));
  } else if (!strcmp(c, "getrusage")) {
    struct rusage ru;
    assert(sizeof ru == 144);
    assert(call(CAPSTONE_SYS_getrusage, RUSAGE_SELF, 64, 0, 0) == 0);
    memcpy(&ru, exchange + 64, sizeof ru);
    assert(ru.ru_maxrss > 0);
    assert(call(CAPSTONE_SYS_getrusage, (uint64_t)-1 /* RUSAGE_CHILDREN */, 64, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_getrusage, 7, 64, 0, 0) == -EINVAL);
  } else if (!strcmp(c, "priority")) {
    /* the raw answer is 20 - nice; this task only, a user or group is refused */
    long raw = call(CAPSTONE_SYS_getpriority, PRIO_PROCESS, 0, 0, 0);
    errno = 0;
    assert(raw == 20 - getpriority(PRIO_PROCESS, 0) && !errno);
    assert(call(CAPSTONE_SYS_setpriority, PRIO_PROCESS, 0, (uint64_t)(20 - raw), 0) == 0);
    assert(call(CAPSTONE_SYS_getpriority, PRIO_PROCESS, 0, 0, 0) == raw);
    assert(call(CAPSTONE_SYS_getpriority, PRIO_USER, 0, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_getpriority, PRIO_PROCESS, 1, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_setpriority, PRIO_PGRP, 0, 0, 0) == -EPERM);
  } else if (!strcmp(c, "getcpu")) {
    unsigned cpu = 99999, node = 99999;
    assert(call(CAPSTONE_SYS_getcpu, 64, 68, 0, 0) == 0);
    memcpy(&cpu, exchange + 64, 4);
    memcpy(&node, exchange + 68, 4);
    assert(cpu < (unsigned)sysconf(_SC_NPROCESSORS_CONF) && node < 1024);
    assert(call(CAPSTONE_SYS_getcpu, 0, 0, 0, 0) == 0);   /* both null */
  } else if (!strcmp(c, "affinity")) {
    /* the mask read is the mask the libc reads, written back it is accepted;
       another task is refused before the kernel sees the request */
    cpu_set_t mine, theirs;
    long n = call(CAPSTONE_SYS_sched_getaffinity, 0, sizeof theirs, 64, 0);
    assert(n > 0 && n <= (long)sizeof theirs);
    memset(&theirs, 0, sizeof theirs);
    memcpy(&theirs, exchange + 64, (size_t)n);
    assert(sched_getaffinity(0, sizeof mine, &mine) == 0 && CPU_EQUAL(&mine, &theirs));
    assert(call(CAPSTONE_SYS_sched_setaffinity, 0, (uint64_t)n, 64, 0) == 0);
    assert(call(CAPSTONE_SYS_sched_getaffinity, 1, sizeof theirs, 64, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_sched_setaffinity, (uint64_t)getppid(), (uint64_t)n, 64, 0) == -EPERM);
  } else if (!strcmp(c, "scheduler")) {
    /* the two readers musl issues; it answers sched_getscheduler and
       sched_getparam with ENOSYS itself, so those have no row */
    struct timespec slice;
    assert(call(CAPSTONE_SYS_sched_get_priority_max, SCHED_FIFO, 0, 0, 0) == sched_get_priority_max(SCHED_FIFO));
    assert(call(CAPSTONE_SYS_sched_get_priority_min, SCHED_FIFO, 0, 0, 0) == sched_get_priority_min(SCHED_FIFO));
    assert(call(CAPSTONE_SYS_sched_get_priority_max, 99, 0, 0, 0) == -EINVAL);
    assert(call(CAPSTONE_SYS_sched_rr_get_interval, 0, 64, 0, 0) == 0);
    memcpy(&slice, exchange + 64, sizeof slice);
    assert(slice.tv_sec >= 0 && slice.tv_nsec >= 0);
    assert(call(CAPSTONE_SYS_sched_rr_get_interval, 1, 64, 0, 0) == -EPERM);
  } else if (!strcmp(c, "setpgid")) {
    /* a child of the task into its own group; a task outside the scope is refused */
    int p[2];
    assert(!pipe(p));
    pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) { char b; close(p[1]); (void)!read(p[0], &b, 1); _exit(0); }
    host.children[host.child_count++] = pid;
    assert(call(CAPSTONE_SYS_setpgid, (uint64_t)pid, 0, 0, 0) == 0);
    assert(getpgid(pid) == pid);
    assert(call(CAPSTONE_SYS_setpgid, (uint64_t)pid, (uint64_t)getpgrp(), 0, 0) == 0);
    assert(getpgid(pid) == getpgrp());
    assert(call(CAPSTONE_SYS_setpgid, 1, 0, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_setpgid, (uint64_t)pid, 1, 0, 0) == -EPERM);
    close(p[0]); close(p[1]);
    int status;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
  } else if (!strcmp(c, "setsid")) {
    /* in a child, which is not a group leader: a new session led by it */
    pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) {
      long sid = call(CAPSTONE_SYS_setsid, 0, 0, 0, 0);
      _exit(sid == getpid() && getsid(0) == getpid() && getpgrp() == getpid() ? 0 : 1);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
  } else if (!strcmp(c, "private")) {
    /* the launcher's own descriptors stay out of reach in every fd position */
    int fd = temp(path), fd2 = temp(other);
    host.private_fds[host.private_count++] = fd;
    assert(call(CAPSTONE_SYS_fstatfs, fd, 256, 0, 0) == -EBADF);
    assert(call6(CAPSTONE_SYS_statx, fd, str(0, ""), AT_EMPTY_PATH, STATX_BASIC_STATS, 512, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_fchmod, fd, 0600, 0, 0) == -EBADF);
    assert(call6(CAPSTONE_SYS_linkat, fd, str(0, "a"), (uint64_t)AT_FDCWD, str(128, "b"), 0, 0) == -EBADF);
    assert(call6(CAPSTONE_SYS_linkat, (uint64_t)AT_FDCWD, 0, fd, 128, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_sendfile, fd2, fd, 0, 1) == -EBADF);
    assert(call(CAPSTONE_SYS_sendfile, fd, fd2, 0, 1) == -EBADF);
    assert(call6(CAPSTONE_SYS_copy_file_range, fd, 0, fd2, 0, 1, 0) == -EBADF);
    assert(call6(CAPSTONE_SYS_copy_file_range, fd2, 0, fd, 0, 1, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_fchdir, fd, 0, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_syncfs, fd, 0, 0, 0) == -EBADF);
    assert(fcntl(fd, F_GETFD) >= 0);
    close(fd); close(fd2); unlink(path); unlink(other);
  } else {
    abort();
  }
  capstone_delegate_host_free(&host);
  return 0;
}
