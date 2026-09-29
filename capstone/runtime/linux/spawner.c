#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "spawner.h"
#include "capstone/spawn.h"
#include <dirent.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

struct request {
  uint32_t bytes, count;
  uint64_t cloexec;
  int32_t numbers[CAPSTONE_SPAWNER_FDS];
};

struct reply {
  int64_t pid;
  int32_t error;
};

int capstone_spawner_is_image(const char *path) {
  Elf64_Ehdr h;
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  ssize_t n;
  if (fd < 0)
    return 0;
  n = read(fd, &h, sizeof h);
  close(fd);
  return n == (ssize_t)sizeof h && !memcmp(h.e_ident, ELFMAG, SELFMAG) && h.e_machine == 259;
}

unsigned capstone_spawner_descriptors(int *fds, int *numbers, uint64_t *cloexec,
                                      unsigned capacity, int skip) {
  DIR *dir = opendir("/proc/self/fd");
  struct dirent *entry;
  unsigned n = 0;
  int dirfd_self;
  *cloexec = 0;
  if (!dir)
    return 0;
  dirfd_self = dirfd(dir);
  if (capacity > 64)
    capacity = 64;
  while (n < capacity && (entry = readdir(dir))) {
    char *end;
    long fd = strtol(entry->d_name, &end, 10);
    int flags;
    if (*end || end == entry->d_name || fd < 0 || fd == dirfd_self || fd == skip)
      continue;
    flags = fcntl((int)fd, F_GETFD);
    if (flags < 0)
      continue;
    if (flags & FD_CLOEXEC)
      *cloexec |= UINT64_C(1) << n;
    fds[n] = (int)fd;
    numbers[n] = (int)fd;
    ++n;
  }
  closedir(dir);
  return n;
}

/* In the child: the received descriptors at the numbers the caller had them,
 * the block's actions in order, then exec. Errors go back through the pipe. */
static void child(const struct request *req, const int *received, const char *block,
                  const char *self, int status_pipe) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 3], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  int error = 0;
  sigset_t none;
  /* Park the received descriptors above any number they must land on; the
     base stays under the guest's descriptor limit, which is 1024 by default. */
  int parked[CAPSTONE_SPAWNER_FDS];
  int base = 0;
  for (unsigned i = 0; i < req->count; ++i)
    if (req->numbers[i] >= base)
      base = req->numbers[i] + 1;
  for (unsigned i = 0; i < req->count; ++i) {
    parked[i] = fcntl(received[i], F_DUPFD_CLOEXEC, base);
    if (parked[i] < 0) { error = errno; goto fail; }
  }
  for (unsigned i = 0; i < req->count; ++i)
    close(received[i]);
  for (unsigned i = 0; i < req->count; ++i) {
    if (dup2(parked[i], req->numbers[i]) < 0) { error = errno; goto fail; }
    close(parked[i]);
    /* the caller's close-on-exec flag survives, so exec drops what it should */
    if ((req->cloexec >> i) & 1)
      fcntl(req->numbers[i], F_SETFD, FD_CLOEXEC);
  }
  if (capstone_spawn_unpack(block, req->bytes, argv + 2, CAPSTONE_SPAWN_STRINGS + 1, envp,
                            CAPSTONE_SPAWN_STRINGS + 1, paths, CAPSTONE_SPAWN_ACTIONS, &view)) {
    error = EINVAL;
    goto fail;
  }
  for (unsigned i = 0; i < view.actions; ++i) {
    struct capstone_spawn_action a;
    int fd;
    memcpy(&a, &view.action[i], sizeof a);
    switch (a.cmd) {
    case CAPSTONE_SPAWN_CLOSE:
      close((int)a.fd);
      break;
    case CAPSTONE_SPAWN_DUP2:
      if (a.srcfd == a.fd) {
        int flags = fcntl((int)a.fd, F_GETFD);
        if (flags < 0 || fcntl((int)a.fd, F_SETFD, flags & ~FD_CLOEXEC)) { error = errno; goto fail; }
      } else if (dup2((int)a.srcfd, (int)a.fd) < 0) { error = errno; goto fail; }
      break;
    case CAPSTONE_SPAWN_OPEN:
      fd = open(paths[i], (int)a.oflag, (mode_t)a.mode);
      if (fd < 0) { error = errno; goto fail; }
      if (fd != (int)a.fd) {
        if (dup2(fd, (int)a.fd) < 0) { error = errno; goto fail; }
        close(fd);
      }
      break;
    case CAPSTONE_SPAWN_CHDIR:
      if (chdir(paths[i])) { error = errno; goto fail; }
      break;
    case CAPSTONE_SPAWN_FCHDIR:
      if (fchdir((int)a.fd)) { error = errno; goto fail; }
      break;
    }
  }
  if (view.flags & CAPSTONE_SPAWN_SETSID) {
    if (setsid() < 0) { error = errno; goto fail; }
  } else if (view.flags & CAPSTONE_SPAWN_SETPGROUP) {
    if (setpgid(0, (pid_t)view.pgroup)) { error = errno; goto fail; }
  }
  sigemptyset(&none);
  sigprocmask(SIG_SETMASK, &none, NULL);
  if (capstone_spawner_is_image(view.path)) {
    /* A Capstone image starts through the launcher, with the caller's argv
       handed to the image after the program name. */
    argv[0] = (char *)self;
    argv[1] = "--";
    argv[2] = (char *)view.path;
    execve(self, argv, envp);
  } else if (view.flags & CAPSTONE_SPAWN_SEARCH_PATH) {
    execvpe(view.path, argv + 2, envp);
  } else {
    execve(view.path, argv + 2, envp);
  }
  error = errno;
fail:
  (void)!write(status_pipe, &error, sizeof error);
  _exit(127);
}

static void serve(int sock, const char *self) {
  static char block[CAPSTONE_SPAWN_BYTES];
  for (;;) {
    struct request req;
    struct iovec iov[2] = {{&req, sizeof req}, {block, sizeof block}};
    char control[CMSG_SPACE(sizeof(int) * CAPSTONE_SPAWNER_FDS)];
    struct msghdr msg = {.msg_iov = iov, .msg_iovlen = 2, .msg_control = control,
                         .msg_controllen = sizeof control};
    struct cmsghdr *cmsg;
    int received[CAPSTONE_SPAWNER_FDS];
    unsigned got = 0;
    ssize_t n = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
    struct reply reply = {0, 0};
    int status_pipe[2];
    long pid;
    if (n <= 0)
      return;
    for (cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg))
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
        got = (unsigned)((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int));
        if (got > CAPSTONE_SPAWNER_FDS) got = CAPSTONE_SPAWNER_FDS;
        memcpy(received, CMSG_DATA(cmsg), got * sizeof(int));
      }
    if ((size_t)n < sizeof req || req.bytes > sizeof block || (size_t)n != sizeof req + req.bytes ||
        req.count != got || req.count > CAPSTONE_SPAWNER_FDS) {
      reply.error = EINVAL;
    } else if (pipe2(status_pipe, O_CLOEXEC)) {
      reply.error = errno;
    } else {
      /* The child is the launcher's, so its wait4 and kill apply. */
      pid = syscall(SYS_clone, CLONE_PARENT | SIGCHLD, 0, 0, 0, 0);
      if (pid == 0) {
        close(status_pipe[0]);
        child(&req, received, block, self, status_pipe[1]);
      }
      close(status_pipe[1]);
      if (pid < 0) {
        reply.error = errno;
      } else {
        int error = 0;
        reply.pid = pid;
        if (read(status_pipe[0], &error, sizeof error) == (ssize_t)sizeof error)
          reply.error = error;
      }
      close(status_pipe[0]);
    }
    for (unsigned i = 0; i < got; ++i)
      close(received[i]);
    if (send(sock, &reply, sizeof reply, MSG_NOSIGNAL) != (ssize_t)sizeof reply)
      return;
  }
}

int capstone_spawner_start(struct capstone_spawner *s) {
  int pair[2];
  ssize_t n;
  s->socket = -1;
  s->pid = 0;
  n = readlink("/proc/self/exe", s->self, sizeof s->self - 1);
  if (n <= 0)
    return errno ? errno : ENOENT;
  s->self[n] = 0;
  if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair))
    return errno;
  s->pid = fork();
  if (s->pid < 0) {
    int error = errno;
    close(pair[0]);
    close(pair[1]);
    return error;
  }
  if (s->pid == 0) {
    close(pair[0]);
    serve(pair[1], s->self);
    _exit(0);
  }
  close(pair[1]);
  s->socket = pair[0];
  return 0;
}

void capstone_spawner_stop(struct capstone_spawner *s) {
  if (s->socket >= 0) {
    close(s->socket);
    s->socket = -1;
  }
  if (s->pid > 0) {
    waitpid(s->pid, NULL, 0);
    s->pid = 0;
  }
}

long capstone_spawner_spawn(struct capstone_spawner *s, const void *block, size_t bytes,
                            const int *fds, const int *numbers, uint64_t cloexec,
                            unsigned count) {
  struct request req = {(uint32_t)bytes, count, cloexec, {0}};
  struct iovec iov[2] = {{&req, sizeof req}, {(void *)block, bytes}};
  char control[CMSG_SPACE(sizeof(int) * CAPSTONE_SPAWNER_FDS)];
  struct msghdr msg = {.msg_iov = iov, .msg_iovlen = 2};
  struct reply reply;
  if (s->socket < 0)
    return -ENOSYS;
  if (bytes > CAPSTONE_SPAWN_BYTES || count > CAPSTONE_SPAWNER_FDS)
    return -EINVAL;
  for (unsigned i = 0; i < count; ++i)
    req.numbers[i] = numbers[i];
  if (count) {
    struct cmsghdr *cmsg;
    memset(control, 0, sizeof control);
    msg.msg_control = control;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * count);
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * count);
    memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * count);
  }
  if (sendmsg(s->socket, &msg, MSG_NOSIGNAL) < 0)
    return -errno;
  if (recv(s->socket, &reply, sizeof reply, 0) != (ssize_t)sizeof reply)
    return -EIO;
  if (reply.error) {
    if (reply.pid > 0)
      waitpid((pid_t)reply.pid, NULL, 0);
    return -reply.error;
  }
  return (long)reply.pid;
}
