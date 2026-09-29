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
#include <sys/stat.h>
#include <sys/prctl.h>
#include <limits.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

struct request {
  uint32_t bytes, count;
  uint64_t cloexec;
  mode_t mask;
  sigset_t sigmask;      /* the launcher's logical mask: what the child inherits */
  uint64_t ignored;      /* the launcher's ignored signals, bit n-1 = signal n */
  int32_t numbers[CAPSTONE_SPAWNER_FDS];
};

/* Linux inheritance across fork and exec, applied by hand because the child is
 * the helper's clone, not the launcher's: ignored signals stay ignored, caught
 * ones fall back to default, SETSIGDEF names those reset to default anyway,
 * and the mask is the launcher's unless SETSIGMASK gives one. */
static void child_signals(const struct request *req, const struct capstone_spawn_view *view) {
  sigset_t mask;
  for (int sig = 1; sig <= 64; ++sig) {
    if (sig == SIGKILL || sig == SIGSTOP) continue;
    int ignored = (req->ignored >> (sig - 1)) & 1;
    if ((view->flags & CAPSTONE_SPAWN_SETSIGDEF) && ((view->sigdefault >> (sig - 1)) & 1))
      ignored = 0;
    signal(sig, ignored ? SIG_IGN : SIG_DFL);
  }
  if (view->flags & CAPSTONE_SPAWN_SETSIGMASK) {
    sigemptyset(&mask);
    for (int sig = 1; sig <= 64; ++sig)
      if ((view->sigmask >> (sig - 1)) & 1) sigaddset(&mask, sig);
  } else {
    mask = req->sigmask;
  }
  sigprocmask(SIG_SETMASK, &mask, NULL);
}

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

int capstone_spawner_descriptors(int *fds, int *numbers, uint64_t *cloexec,
                                      unsigned capacity, int skip) {
  DIR *dir = opendir("/proc/self/fd");
  struct dirent *entry;
  unsigned n = 0;
  int dirfd_self;
  *cloexec = 0;
  if (!dir)
    return -errno;
  dirfd_self = dirfd(dir);
  if (capacity > 64)
    capacity = 64;
  while ((entry = readdir(dir))) {
    char *end;
    long fd = strtol(entry->d_name, &end, 10);
    int flags;
    if (*end || end == entry->d_name || fd < 0 || fd == dirfd_self || fd == skip)
      continue;
    flags = fcntl((int)fd, F_GETFD);
    if (flags < 0)
      continue;
    if (n == capacity) {
      closedir(dir);
      return -EMFILE;
    }
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
  static char *argv[CAPSTONE_SPAWN_STRINGS + 5], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  int error = 0;
  if (capstone_spawn_unpack(block, req->bytes, argv + 4, CAPSTONE_SPAWN_STRINGS + 1, envp,
                            CAPSTONE_SPAWN_STRINGS + 1, paths, CAPSTONE_SPAWN_ACTIONS, &view)) {
    error = EINVAL;
    goto fail;
  }
  if (fchdir(received[req->count])) { error = errno; goto fail; }
  close(received[req->count]);
  umask(req->mask);
  /* Park the received descriptors above any number they must land on; the
     base stays under the guest's descriptor limit, which is 1024 by default. */
  int parked[CAPSTONE_SPAWNER_FDS];
  int base = 0;
  for (unsigned i = 0; i < req->count; ++i)
    if (req->numbers[i] >= base)
      base = req->numbers[i] + 1;
  for (unsigned i = 0; i < view.actions; ++i) {
    struct capstone_spawn_action a;
    memcpy(&a, &view.action[i], sizeof a);
    if ((int)a.fd >= base) base = (int)a.fd + 1;
    if ((int)a.srcfd >= base) base = (int)a.srcfd + 1;
  }
  /* No file action may close or replace the exec-error channel. */
  int safe_status = fcntl(status_pipe, F_DUPFD_CLOEXEC, base);
  if (safe_status < 0) { error = errno; goto fail; }
  close(status_pipe);
  status_pipe = safe_status;
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
  }
  if (view.flags & CAPSTONE_SPAWN_SETPGROUP) {
    if (setpgid(0, (pid_t)view.pgroup)) { error = errno; goto fail; }
  }
  child_signals(req, &view);
  /* Resolve PATH before image detection, using the caller's environment. */
  const char *program = view.path;
  char resolved[PATH_MAX];
  if ((view.flags & CAPSTONE_SPAWN_SEARCH_PATH) && !strchr(program, '/')) {
    const char *search = "/bin:/usr/bin";
    for (unsigned i = 0; i < view.envc; ++i)
      if (!strncmp(envp[i], "PATH=", 5)) { search = envp[i] + 5; break; }
    int denied = 0;
    for (;;) {
      const char *end = strchr(search, ':');
      size_t len = end ? (size_t)(end - search) : strlen(search);
      if (len + strlen(program) + 2 <= sizeof resolved) {
        memcpy(resolved, search, len);
        size_t used = len;
        if (len) resolved[used++] = '/';
        strcpy(resolved + used, program);
        if (!access(resolved, X_OK)) { program = resolved; break; }
        if (errno == EACCES) denied = 1;
      }
      if (!end) { error = denied ? EACCES : ENOENT; goto fail; }
      search = end + 1;
    }
  }
  if (capstone_spawner_is_image(program)) {
    /* Keep the image path separate from its arbitrary argv[0]. */
    argv[0] = (char *)self;
    argv[1] = "--application-argv";
    argv[2] = (char *)program;
    argv[3] = "--";
    execve(self, argv, envp);
  } else {
    execve(program, argv + 4, envp);
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
    char control[CMSG_SPACE(sizeof(int) * (CAPSTONE_SPAWNER_FDS + 1))];
    struct msghdr msg = {.msg_iov = iov, .msg_iovlen = 2, .msg_control = control,
                         .msg_controllen = sizeof control};
    struct cmsghdr *cmsg;
    int received[CAPSTONE_SPAWNER_FDS + 1];
    unsigned got = 0;
    ssize_t n;
    do { n = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC); } while (n < 0 && errno == EINTR);
    struct reply reply = {0, 0};
    int status_pipe[2];
    long pid;
    if (n <= 0)
      return;
    for (cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg))
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
        got = (unsigned)((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int));
        if (got > CAPSTONE_SPAWNER_FDS + 1) got = CAPSTONE_SPAWNER_FDS + 1;
        memcpy(received, CMSG_DATA(cmsg), got * sizeof(int));
      }
    if ((size_t)n < sizeof req || req.bytes > sizeof block || (size_t)n != sizeof req + req.bytes ||
        req.count + 1 != got || req.count > CAPSTONE_SPAWNER_FDS ||
        (msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
      reply.error = EINVAL;
    } else if (pipe2(status_pipe, O_CLOEXEC)) {
      reply.error = errno;
    } else {
      /* The child is the launcher's, so its wait4 and kill apply. */
      pid = syscall(SYS_clone, CLONE_PARENT | SIGCHLD, 0, 0, 0, 0);
      if (pid == 0) {
        close(status_pipe[0]);
        close(sock);
        child(&req, received, block, self, status_pipe[1]);
      }
      close(status_pipe[1]);
      if (pid < 0) {
        reply.error = errno;
      } else {
        int error = 0;
        reply.pid = pid;
        ssize_t received_error;
        do { received_error = read(status_pipe[0], &error, sizeof error); }
        while (received_error < 0 && errno == EINTR);
        if (received_error == (ssize_t)sizeof error) reply.error = error;
        else if (received_error != 0) reply.error = EIO;
      }
      close(status_pipe[0]);
    }
    for (unsigned i = 0; i < got; ++i)
      close(received[i]);
    ssize_t sent;
    do { sent = send(sock, &reply, sizeof reply, MSG_NOSIGNAL); } while (sent < 0 && errno == EINTR);
    if (sent != (ssize_t)sizeof reply) return;
  }
}

int capstone_spawner_start(struct capstone_spawner *s) {
  int pair[2];
  pid_t parent = getpid();
  ssize_t n;
  s->socket = -1;
  s->pid = 0;
  n = readlink("/proc/self/exe", s->self, sizeof s->self - 1);
  if (n <= 0)
    return errno ? errno : ENOENT;
  s->self[n] = 0;
  if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair))
    return errno;
  /* No exit signal: the helper's death is not a SIGCHLD the application
     could mistake for a child of its own. Waited with __WALL. */
  s->pid = (pid_t)syscall(SYS_clone, 0, 0, 0, 0, 0);
  if (s->pid < 0) {
    int error = errno;
    close(pair[0]);
    close(pair[1]);
    return error;
  }
  if (s->pid == 0) {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != parent) _exit(125);
    /* The helper keeps none of the launcher's dispositions (they would record
       into a copy of its ring) and survives the terminal's signals: an
       application that handles Ctrl-C must not lose its spawner to it. Each
       child gets the launcher's view back from child_signals(). */
    for (int sig = 1; sig <= 64; ++sig)
      if (sig != SIGKILL && sig != SIGSTOP) signal(sig, SIG_DFL);
    signal(SIGINT, SIG_IGN); signal(SIGQUIT, SIG_IGN); signal(SIGTSTP, SIG_IGN);
    signal(SIGTTIN, SIG_IGN); signal(SIGTTOU, SIG_IGN);
    sigset_t none;
    sigemptyset(&none);
    sigprocmask(SIG_SETMASK, &none, NULL);
    close(pair[0]);
    /* The helper owns only its socket. Inherited pipes and device files would
       keep applications alive and resurrect descriptors the caller closed. */
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) _exit(125);
    struct dirent *entry;
    while ((entry = readdir(dir))) {
      char *end;
      long fd = strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && !*end && fd != pair[1] && fd != dirfd(dir))
        close((int)fd);
    }
    closedir(dir);
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
    while (waitpid(s->pid, NULL, __WALL) < 0 && errno == EINTR) {}
    s->pid = 0;
  }
}

long capstone_spawner_spawn(struct capstone_spawner *s, const void *block, size_t bytes,
                            const int *fds, const int *numbers, uint64_t cloexec,
                            unsigned count, uint64_t ignored, uint64_t logical_mask) {
  struct request req = {.bytes = (uint32_t)bytes, .count = count, .cloexec = cloexec,
                        .ignored = ignored};
  struct iovec iov[2] = {{&req, sizeof req}, {(void *)block, bytes}};
  char control[CMSG_SPACE(sizeof(int) * (CAPSTONE_SPAWNER_FDS + 1))];
  struct msghdr msg = {.msg_iov = iov, .msg_iovlen = 2};
  struct reply reply;
  if (s->socket < 0)
    return -ENOSYS;
  if (bytes > CAPSTONE_SPAWN_BYTES || count > CAPSTONE_SPAWNER_FDS)
    return -EINVAL;
  for (unsigned i = 0; i < count; ++i)
    req.numbers[i] = numbers[i];
  int passed[CAPSTONE_SPAWNER_FDS + 1];
  int cwd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (cwd < 0) return -errno;
  req.mask = umask(0);
  umask(req.mask);
  sigemptyset(&req.sigmask);
  for (int sig = 1; sig <= 64; ++sig)
    if ((logical_mask >> (sig - 1)) & 1) sigaddset(&req.sigmask, sig);
  for (unsigned i = 0; i < count; ++i) passed[i] = fds[i];
  passed[count] = cwd;
  {
    struct cmsghdr *cmsg;
    memset(control, 0, sizeof control);
    msg.msg_control = control;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * (count + 1));
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * (count + 1));
    memcpy(CMSG_DATA(cmsg), passed, sizeof(int) * (count + 1));
  }
  ssize_t n;
  do { n = sendmsg(s->socket, &msg, MSG_NOSIGNAL); } while (n < 0 && errno == EINTR);
  int error = errno;
  close(cwd);
  if (n < 0) return -error;
  do { n = recv(s->socket, &reply, sizeof reply, 0); } while (n < 0 && errno == EINTR);
  if (n != (ssize_t)sizeof reply) return -EIO;
  if (reply.error) {
    if (reply.pid > 0)
      while (waitpid((pid_t)reply.pid, NULL, 0) < 0 && errno == EINTR) {}
    return -reply.error;
  }
  return (long)reply.pid;
}
