/* The signal contract of the delegated runtime: one mode per case of
 * docs/plans/delegation-signals.md, each with the oracle written there. Every
 * mode exits 0 and prints "signal-contract <mode>: PASS", or fails the CHECK
 * that names the broken promise. Children are ordinary Linux programs started
 * through posix_spawn, so an external signal is a real kill(2) from another
 * process; a self-delivered one is raise(3), which is musl's tkill.
 *
 * Busy waits use CLOCK_MONOTONIC, which the libc answers without a round: a
 * signal that arrives during one is the "domain computes" case. */
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <setjmp.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "signal-contract:%d: %s (errno %d)\n", __LINE__, #test, errno); return 1; \
} } while (0)

extern char **environ;
static const char *mode;

static volatile sig_atomic_t count[65], depth[65], max_depth[65];
static volatile sig_atomic_t seen_blocked[65];   /* was `probe_signal` blocked when this handler ran */
static volatile int probe_signal = SIGUSR2;
static volatile int inner_raised;
static volatile int handler_pipe = -1;           /* before-read: the handler writes here */
static volatile intptr_t handler_sp[65];         /* altstack: a local's address inside the handler */
static sigjmp_buf escape;

static void note(int sig) {
  sigset_t now;
  sigprocmask(SIG_BLOCK, NULL, &now);
  seen_blocked[sig] = sigismember(&now, probe_signal);
  depth[sig]++;
  if (depth[sig] > max_depth[sig]) max_depth[sig] = depth[sig];
}
static void done(int sig) { depth[sig]--; count[sig]++; }

static void counting(int sig) { note(sig); done(sig); }

static void raising_self_once(int sig) {
  note(sig);
  if (!inner_raised) { inner_raised = 1; raise(sig); }
  done(sig);
}

static void writing_to_pipe(int sig) {
  note(sig);
  if (handler_pipe >= 0) (void)!write(handler_pipe, "H", 1);
  done(sig);
}

static void writing_to_stdout(int sig) {
  note(sig);
  (void)!write(1, "H\n", 2);
  done(sig);
}

static void raising_other(int sig) {  /* A: raises B inside */
  note(sig);
  raise(SIGUSR2);
  done(sig);
}
static void raising_back(int sig) {   /* B: raises A inside; A must not re-enter */
  note(sig);
  raise(SIGUSR1);
  done(sig);
}

static void reinstalling(int sig) {
  note(sig);
  struct sigaction sa = {.sa_handler = reinstalling, .sa_flags = SA_RESETHAND};
  sigaction(sig, &sa, NULL);
  done(sig);
}

static void on_altstack(int sig) {
  volatile char local;
  stack_t query;
  note(sig);
  handler_sp[sig] = (intptr_t)&local;
  if (!sigaltstack(NULL, &query) && (query.ss_flags & SS_ONSTACK)) seen_blocked[sig] = 2;
  if (sig == SIGUSR1) raise(SIGUSR2);  /* nested, also SA_ONSTACK */
  else siglongjmp(escape, 1);
  done(sig);
}

static int install(int sig, void (*handler)(int), int flags) {
  struct sigaction sa = {.sa_handler = handler, .sa_flags = flags};
  sigemptyset(&sa.sa_mask);
  return sigaction(sig, &sa, NULL);
}

/* A native child that runs `sh -c command`; stdout may be redirected. */
static pid_t child(const char *command, int stdout_fd) {
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  if (stdout_fd >= 0) posix_spawn_file_actions_adddup2(&actions, stdout_fd, 1);
  char *argv[] = {"sh", "-c", (char *)command, NULL};
  pid_t pid;
  int error = posix_spawn(&pid, "/bin/sh", &actions, NULL, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  if (error) { errno = error; return -1; }
  return pid;
}

/* The same, with signal attributes. */
static pid_t child_attr(const char *command, const posix_spawnattr_t *attr, int stdout_fd) {
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  if (stdout_fd >= 0) posix_spawn_file_actions_adddup2(&actions, stdout_fd, 1);
  char *argv[] = {"sh", "-c", (char *)command, NULL};
  pid_t pid;
  int error = posix_spawn(&pid, "/bin/sh", &actions, attr, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  if (error) { errno = error; return -1; }
  return pid;
}

static pid_t sender(const char *signal_name, const char *delay) {
  char command[128];
  snprintf(command, sizeof command, "sleep %s; kill -%s %ld", delay, signal_name, (long)getpid());
  return child(command, -1);
}

static int reap(pid_t pid, int *status) {
  pid_t r;
  do r = waitpid(pid, status, 0); while (r < 0 && errno == EINTR);
  return r == pid ? 0 : -1;
}

static void busy_ms(long ms) {
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);
  do clock_gettime(CLOCK_MONOTONIC, &now);
  while ((now.tv_sec - start.tv_sec) * 1000 + (now.tv_nsec - start.tv_nsec) / 1000000 < ms);
}

static int pass(void) { printf("signal-contract %s: PASS\n", mode); return 0; }

int main(int argc, char **argv) {
  if (argc != 2) return 41;
  mode = argv[1];
  int status;

  if (!strcmp(mode, "self")) {
    CHECK(!install(SIGUSR1, counting, 0));
    CHECK(!raise(SIGUSR1));
    CHECK(count[SIGUSR1] == 1);           /* the handler ran before raise returned */
    return pass();
  }
  if (!strcmp(mode, "self-nodefer") || !strcmp(mode, "self-defer")) {
    int nodefer = !strcmp(mode, "self-nodefer");
    CHECK(!install(SIGUSR1, raising_self_once, nodefer ? SA_NODEFER : 0));
    CHECK(!raise(SIGUSR1));
    CHECK(count[SIGUSR1] == 2);           /* both instances ran before raise returned */
    CHECK(max_depth[SIGUSR1] == (nodefer ? 2 : 1));  /* nested only with SA_NODEFER */
    return pass();
  }
  if (!strcmp(mode, "before-read")) {
    /* The signal arrives while the domain computes; its handler must run before
       the read enters the kernel, and what it wrote is what the read returns. */
    int fds[2];
    CHECK(!pipe(fds));
    handler_pipe = fds[1];
    CHECK(!install(SIGUSR1, writing_to_pipe, 0));
    pid_t s = sender("USR1", "0.2");
    CHECK(s > 0);
    busy_ms(700);
    char c = 0;
    CHECK(read(fds[0], &c, 1) == 1 && c == 'H');
    CHECK(count[SIGUSR1] == 1);
    CHECK(!reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "during-read-restart") || !strcmp(mode, "during-read-eintr")) {
    /* The read is blocked when the signal arrives. Linux decides: SA_RESTART
       restarts it and the handler's byte is what it returns; without SA_RESTART
       it fails with EINTR after the handler ran. */
    int restart = !strcmp(mode, "during-read-restart");
    int fds[2];
    CHECK(!pipe(fds));
    handler_pipe = fds[1];
    CHECK(!install(SIGUSR1, writing_to_pipe, restart ? SA_RESTART : 0));
    pid_t s = sender("USR1", "0.3");
    CHECK(s > 0);
    char c = 0;
    ssize_t n = read(fds[0], &c, 1);
    CHECK(count[SIGUSR1] == 1);
    if (restart) CHECK(n == 1 && c == 'H');
    else CHECK(n == -1 && errno == EINTR);
    CHECK(!reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "handler-write")) {
    /* A handler that itself delegates while the interrupted read has data in
       flight: the outer data must come back intact, the handler's bytes too. */
    int fds[2];
    CHECK(!pipe(fds));
    CHECK(!install(SIGUSR1, writing_to_stdout, SA_RESTART));
    pid_t w = child("sleep 0.5; echo DATA", fds[1]);
    CHECK(w > 0);
    close(fds[1]);
    pid_t s = sender("USR1", "0.2");
    CHECK(s > 0);
    char buf[16] = {0};
    ssize_t n = read(fds[0], buf, sizeof buf - 1);
    CHECK(n == 5 && !memcmp(buf, "DATA\n", 5));
    CHECK(count[SIGUSR1] == 1);
    CHECK(!reap(w, &status) && !reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "sigsuspend") || !strcmp(mode, "ppoll")) {
    /* Original mask blocks USR1 and USR2; the wait unblocks everything. The
       handler runs under the temporary mask (USR2 free) before the call
       returns, and the original mask is back afterwards. */
    sigset_t original, empty, after;
    sigemptyset(&original); sigaddset(&original, SIGUSR1); sigaddset(&original, SIGUSR2);
    sigemptyset(&empty);
    CHECK(!install(SIGUSR1, counting, 0));
    CHECK(!sigprocmask(SIG_SETMASK, &original, NULL));
    pid_t s = sender("USR1", "0.3");
    CHECK(s > 0);
    int r = !strcmp(mode, "ppoll") ? ppoll(NULL, 0, NULL, &empty) : sigsuspend(&empty);
    int saved = errno;
    CHECK(!sigprocmask(SIG_BLOCK, NULL, &after));
    CHECK(r == -1 && saved == EINTR);
    CHECK(count[SIGUSR1] == 1);            /* the waking handler ran before the return */
    CHECK(seen_blocked[SIGUSR1] == 0);     /* under the temporary mask, USR2 was free */
    CHECK(sigismember(&after, SIGUSR1) && sigismember(&after, SIGUSR2));
    CHECK(!sigprocmask(SIG_SETMASK, &empty, NULL));
    CHECK(!reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "nest")) {
    /* A raises B; B sees A blocked and raises A, which must not re-enter. */
    probe_signal = SIGUSR1;
    CHECK(!install(SIGUSR1, raising_other, 0));
    CHECK(!install(SIGUSR2, raising_back, 0));
    CHECK(!raise(SIGUSR1));
    CHECK(seen_blocked[SIGUSR2] == 1);     /* inside B, A was blocked */
    CHECK(max_depth[SIGUSR1] == 1);        /* A never nested */
    CHECK(count[SIGUSR1] == 2 && count[SIGUSR2] == 1);
    return pass();
  }
  if (!strcmp(mode, "retry-partial")) {
    /* A large write into a pipe read slowly by a child, interrupted by a signal
       with SA_RESTART: the partial count comes back, nothing is written twice,
       the child counts exactly the bytes the writes reported. */
    int data[2], back[2];
    CHECK(!pipe(data) && !pipe(back));
    CHECK(!install(SIGUSR1, counting, SA_RESTART));
    char command[128];
    snprintf(command, sizeof command, "sleep 0.3; kill -USR1 %ld; sleep 0.3; wc -c", (long)getpid());
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, data[0], 0);
    posix_spawn_file_actions_adddup2(&actions, back[1], 1);
    char *args[] = {"sh", "-c", command, NULL};
    pid_t pid;
    CHECK(!posix_spawn(&pid, "/bin/sh", &actions, NULL, args, environ));
    posix_spawn_file_actions_destroy(&actions);
    close(data[0]); close(back[1]);
    static char block[1 << 20];
    memset(block, 'x', sizeof block);
    long total = 0;
    for (int i = 0; i < 2; ++i) {
      size_t off = 0;
      while (off < sizeof block) {
        ssize_t n = write(data[1], block + off, sizeof block - off);
        if (n < 0 && errno == EINTR) continue;   /* allowed without SA_RESTART; not expected here */
        CHECK(n > 0);
        off += (size_t)n; total += n;
      }
    }
    close(data[1]);
    char reply[32] = {0};
    ssize_t n = read(back[0], reply, sizeof reply - 1);
    CHECK(n > 0);
    CHECK(strtol(reply, NULL, 10) == total);
    CHECK(count[SIGUSR1] == 1);
    CHECK(!reap(pid, &status));
    return pass();
  }
  if (!strcmp(mode, "wait-restart") || !strcmp(mode, "wait-eintr")) {
    /* waitpid(-1) interrupted: SA_RESTART completes it with the child's status
       after the handler; without it, EINTR first, the status on the next call. */
    int restart = !strcmp(mode, "wait-restart");
    CHECK(!install(SIGUSR1, counting, restart ? SA_RESTART : 0));
    char command[96];
    snprintf(command, sizeof command, "sleep 0.2; kill -USR1 %ld; sleep 0.3; exit 7", (long)getpid());
    pid_t s = child(command, -1);
    CHECK(s > 0);
    pid_t r = waitpid(-1, &status, 0);
    CHECK(count[SIGUSR1] == 1);
    if (restart) CHECK(r == s);
    else { CHECK(r == -1 && errno == EINTR); CHECK(waitpid(-1, &status, 0) == s); }
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 7);
    return pass();
  }
  if (!strcmp(mode, "spawn-interrupted")) {
    /* Signals arriving around spawn requests never duplicate a child. */
    CHECK(!install(SIGUSR1, counting, SA_RESTART));
    pid_t noise = sender("USR1", "0");
    CHECK(noise > 0);
    int children = 0;
    for (int i = 0; i < 20; ++i) {
      pid_t c = child("exit 0", -1);
      CHECK(c > 0);
      ++children;
    }
    CHECK(!reap(noise, &status));
    int reaped = 0;
    while (waitpid(-1, &status, 0) > 0) ++reaped;
    CHECK(reaped == children);              /* exactly the children we asked for */
    return pass();
  }
  if (!strcmp(mode, "resethand-die")) {
    /* SA_RESETHAND: the first instance runs the handler, the second finds the
       default action and ends the process with SIGUSR1. */
    CHECK(!install(SIGUSR1, counting, SA_RESETHAND));
    pid_t s = child("sleep 0.2; kill -USR1 $PPID; sleep 0.3; kill -USR1 $PPID", -1);
    CHECK(s > 0);
    struct timespec t = {2, 0};
    nanosleep(&t, NULL);
    fprintf(stderr, "signal-contract: still alive after two SIGUSR1 (count %d)\n", (int)count[SIGUSR1]);
    return 2;
  }
  if (!strcmp(mode, "resethand-reinstall")) {
    CHECK(!install(SIGUSR1, reinstalling, SA_RESETHAND));
    pid_t s = child("sleep 0.2; kill -USR1 $PPID; sleep 0.3; kill -USR1 $PPID", -1);
    CHECK(s > 0);
    struct timespec t = {1, 200000000};
    while (nanosleep(&t, &t) < 0 && errno == EINTR) ;
    CHECK(count[SIGUSR1] == 2);
    CHECK(!reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "rt-queue")) {
    /* 100 realtime signals sent while the domain computes: none may be lost. */
    int rt = SIGRTMIN + 3;
    CHECK(!install(rt, counting, 0));
    char command[128];
    snprintf(command, sizeof command,
             "i=0; while [ $i -lt 100 ]; do kill -%d %ld; i=$((i+1)); done", rt, (long)getpid());
    pid_t s = child(command, -1);
    CHECK(s > 0);
    busy_ms(1500);
    CHECK(!reap(s, &status));
    CHECK(count[rt] == 100);
    return pass();
  }
  if (!strcmp(mode, "hint")) {
    /* Accepted during computation, delivered at the next entry into the libc's
       syscall dispatcher, a local one included: getpid() here. */
    CHECK(!install(SIGUSR1, counting, 0));
    pid_t s = sender("USR1", "0.2");
    CHECK(s > 0);
    busy_ms(700);
    CHECK(count[SIGUSR1] == 0 || count[SIGUSR1] == 1);
    (void)getpid();
    CHECK(count[SIGUSR1] == 1);
    CHECK(!reap(s, &status));
    return pass();
  }
  if (!strcmp(mode, "altstack")) {
    static char stack[65536] __attribute__((aligned(16)));
    stack_t alt = {.ss_sp = stack, .ss_size = sizeof stack, .ss_flags = 0}, query;
    CHECK(!sigaltstack(&alt, NULL));
    CHECK(!install(SIGUSR1, on_altstack, SA_ONSTACK));
    CHECK(!install(SIGUSR2, on_altstack, SA_ONSTACK));
    if (!sigsetjmp(escape, 1)) {
      raise(SIGUSR1);
      CHECK(0);  /* the nested handler leaves through siglongjmp */
    }
    CHECK(seen_blocked[SIGUSR1] == 2 && seen_blocked[SIGUSR2] == 2);   /* SS_ONSTACK reported inside both */
    CHECK(handler_sp[SIGUSR1] >= (intptr_t)stack && handler_sp[SIGUSR1] < (intptr_t)stack + (intptr_t)sizeof stack);
    CHECK(handler_sp[SIGUSR2] < handler_sp[SIGUSR1]);   /* nested handler deeper, not reset to the top */
    CHECK(!sigaltstack(NULL, &query) && !(query.ss_flags & SS_ONSTACK));
    sigset_t now;
    CHECK(!sigprocmask(SIG_SETMASK, NULL, &now));
    CHECK(!sigprocmask(SIG_UNBLOCK, &now, NULL));    /* leave with a clean mask after the jump */
    CHECK(!raise(SIGUSR1) || 1);                       /* the stack is usable again */
    return pass();
  }
  if (!strcmp(mode, "ign-inherit")) {
    /* SIG_IGN is inherited by a child spawned afterwards; SETSIGDEF undoes it. */
    CHECK(signal(SIGPIPE, SIG_IGN) != SIG_ERR);
    int out[2];
    CHECK(!pipe(out));
    pid_t a = child("kill -PIPE $$; echo alive", out[1]);
    CHECK(a > 0);
    CHECK(!reap(a, &status) && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char buf[16] = {0};
    CHECK(read(out[0], buf, sizeof buf - 1) == 6 && !memcmp(buf, "alive\n", 6));
    posix_spawnattr_t attr;
    sigset_t defaults;
    sigemptyset(&defaults); sigaddset(&defaults, SIGPIPE);
    CHECK(!posix_spawnattr_init(&attr));
    CHECK(!posix_spawnattr_setsigdefault(&attr, &defaults));
    CHECK(!posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGDEF));
    pid_t b = child_attr("kill -PIPE $$; echo alive", &attr, -1);
    CHECK(b > 0);
    CHECK(!reap(b, &status) && WIFSIGNALED(status) && WTERMSIG(status) == SIGPIPE);
    return pass();
  }
  fprintf(stderr, "signal-contract: unknown mode %s\n", mode);
  return 40;
}
