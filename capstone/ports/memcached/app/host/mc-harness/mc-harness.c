/* mc-harness: the memcached port's oracle driver.
 *
 *   mc-harness --out DIR [--port P] [--conns N] [--stop TERM|USR1] [--signal-child] [--perturb none|value|cas]
 *              -- SERVER-COMMAND...
 *   mc-harness --out DIR [--port P] --fixture N -- SERVER-COMMAND...
 *
 * Starts SERVER-COMMAND (memcached natively, or capstone-job ... capstone-exec memcached.dom in the guest),
 * waits until it answers `version`, runs one fixed script, stops it with the chosen signal and records
 * its wait status. The same source is built for the host (the native reference) and for the guest.
 *
 * The script avoids every line that depends on sizeof(item) or sizeof(void *) (raw stats, slab and item
 * stats, values near the item size limit, memory pressure) and every TTL that depends on the clock. CAS
 * uniques are the one value rewritten: to per-connection ordinals in order of first appearance.
 *
 * DIR/transcript.raw   every response byte, phase and connection headers between them
 * DIR/transcript.norm  the same with CAS uniques rewritten: the file compared against the native run
 * DIR/identity.txt     the raw `stats` pointer_size line (64 natively, 128 in a domain): not compared
 * DIR/status.txt       the server's wait status, and how long the stop took
 * DIR/server.out|err   the server's stdout and stderr
 *
 * --fixture N (the Safety milestone) runs no script: once the server listens it sends the hidden
 * `mc_capstone_fixture N` (patch 0005), reads until the server closes the connection, and records how
 * the server ended without signalling it (DIR/fixture-reply.txt, DIR/status.txt "stop=none ...").
 *
 * --signal-child sends the stop signal to SERVER-COMMAND's child (its process group) instead of to
 * SERVER-COMMAND itself. capstone-job forwards SIGINT, SIGTERM and SIGHUP to its child and nothing
 * else (runtime/linux/job.c), so a SIGUSR1 sent to capstone-job kills the helper and never reaches the
 * domain; with this option it reaches capstone-exec, and capstone-job still records its status.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int port = 11211, nconns = 8, perturb_value, perturb_cas;
static const char *outdir;

/* ---- a growable text buffer: one per connection, joined into the transcript at the end ---- */
struct buf { char *p; size_t n, cap; };
static void bput(struct buf *b, const char *s, size_t n) {
  if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2; b->p = realloc(b->p, b->cap); }
  memcpy(b->p + b->n, s, n); b->n += n; b->p[b->n] = 0;
}
static void bprintf(struct buf *b, const char *fmt, ...) {
  char tmp[512]; va_list ap; va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap);
  bput(b, tmp, (size_t)n);
}

/* ---- one connection: a socket, a read buffer, its transcript and its CAS ordinals ---- */
struct conn {
  int fd, dead; char rb[1 << 16]; size_t rn, rpos;
  struct buf raw, norm;
  unsigned long long cas_seen[4096]; int ncas;
  unsigned long long last_cas;
};

static int dial(void) {
  struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons((unsigned short)port)};
  inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) return -1;
  if (connect(s, (struct sockaddr *)&sa, sizeof sa)) { close(s); return -1; }
  int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  return s;
}
static void sendall(struct conn *c, const char *p, size_t n) {
  while (n) { ssize_t w = write(c->fd, p, n); if (w <= 0) { if (errno == EINTR) continue; return; } p += w; n -= (size_t)w; }
}
static void sends(struct conn *c, const char *s) { sendall(c, s, strlen(s)); }
/* 60 s without a byte is recorded as a timeout, so a server that stops answering ends the run with a
   transcript that says where, instead of hanging it */
static int timed_out;
static int fill(struct conn *c) {
  if (c->rpos < c->rn) return 1;
  /* a connection that timed out once stays timed out: one that lost its place in the protocol would
     otherwise wait 60 s for every response still owed to it */
  if (c->dead) return 0;
  for (;;) {
    struct pollfd pf = {.fd = c->fd, .events = POLLIN};
    int pr = poll(&pf, 1, 60 * 1000);
    if (pr == 0) { timed_out = 1; c->dead = 1; bput(&c->raw, "<TIMEOUT>\n", 10); bput(&c->norm, "<TIMEOUT>\n", 10); return 0; }
    if (pr < 0 && errno == EINTR) continue;
    ssize_t r = read(c->fd, c->rb, sizeof c->rb);
    if (r > 0) { c->rn = (size_t)r; c->rpos = 0; return 1; }
    if (r < 0 && errno == EINTR) continue;
    return 0;
  }
}
/* a line including its \r\n; 0 at EOF */
static size_t readline(struct conn *c, char *line, size_t max) {
  size_t n = 0;
  while (n + 1 < max) { if (!fill(c)) break; char ch = c->rb[c->rpos++]; line[n++] = ch; if (ch == '\n') break; }
  line[n] = 0; return n;
}
static void readn(struct conn *c, struct buf *dst, size_t n) {
  while (n) {
    if (!fill(c)) return;
    size_t take = c->rn - c->rpos < n ? c->rn - c->rpos : n;
    if (dst) bput(dst, c->rb + c->rpos, take);
    c->rpos += take;
    n -= take;
  }
}

/* CAS uniques: the value, replaced by its ordinal in this connection's order of first appearance */
static unsigned cas_ordinal(struct conn *c, unsigned long long v) {
  for (int i = 0; i < c->ncas; i++) if (c->cas_seen[i] == v) return (unsigned)i + 1;
  if (c->ncas < 4096) c->cas_seen[c->ncas++] = v;
  return (unsigned)c->ncas;
}
/* append one response line to raw, and to norm with its CAS rewritten */
static void record(struct conn *c, const char *line, size_t n) {
  bput(&c->raw, line, n);
  char out[1024]; unsigned long long v;
  char key[256]; unsigned flags, bytes;
  if (sscanf(line, "VALUE %255s %u %u %llu", key, &flags, &bytes, &v) == 4) {
    c->last_cas = v; snprintf(out, sizeof out, "VALUE %s %u %u cas#%u\r\n", key, flags, bytes, cas_ordinal(c, v));
    bput(&c->norm, out, strlen(out)); return;
  }
  /* meta: a " c<digits>" flag in an HD/VA/OK line */
  const char *cp = strstr(line, " c");
  if ((!strncmp(line, "HD", 2) || !strncmp(line, "VA ", 3)) && cp && cp[2] >= '0' && cp[2] <= '9') {
    v = strtoull(cp + 2, NULL, 10); c->last_cas = v;
    size_t pre = (size_t)(cp - line); const char *rest = cp + 2; while (*rest >= '0' && *rest <= '9') rest++;
    snprintf(out, sizeof out, "%.*s ccas#%u%s", (int)pre, line, cas_ordinal(c, v), rest);
    bput(&c->norm, out, strlen(out)); return;
  }
  bput(&c->norm, line, n);
}
/* one response, read by the protocol's grammar: VALUE blocks to END, a VA line and its data, STAT lines
   to END, or a single line */
static void response_values(struct conn *c) {
  char line[1024];
  for (;;) {
    size_t n = readline(c, line, sizeof line);
    if (!n) { bput(&c->raw, "<EOF>\n", 6); bput(&c->norm, "<EOF>\n", 6); return; }
    record(c, line, n);
    unsigned flags, bytes; char key[256];
    if (sscanf(line, "VALUE %255s %u %u", key, &flags, &bytes) == 3) {
      struct buf tmp = {0}; readn(c, &tmp, bytes + 2);
      bput(&c->raw, tmp.p ? tmp.p : "", tmp.n); bput(&c->norm, tmp.p ? tmp.p : "", tmp.n); free(tmp.p); continue;
    }
    if (!strncmp(line, "VA ", 3) && sscanf(line, "VA %u", &bytes) == 1) {
      struct buf tmp = {0}; readn(c, &tmp, bytes + 2);
      bput(&c->raw, tmp.p ? tmp.p : "", tmp.n); bput(&c->norm, tmp.p ? tmp.p : "", tmp.n); free(tmp.p); return;
    }
    if (!strncmp(line, "STAT ", 5)) continue;
    return;
  }
}
static void cmd(struct conn *c, const char *s) { sends(c, s); response_values(c); }
static void cmdf(struct conn *c, const char *fmt, ...) {
  char tmp[512]; va_list ap; va_start(ap, fmt); vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap); cmd(c, tmp);
}
/* a value of n bytes, deterministic, so both runs store the same */
static char *payload(size_t n, unsigned seed) {
  char *p = malloc(n + 1);
  for (size_t i = 0; i < n; i++) p[i] = (char)('a' + (i * 7 + seed) % 26);
  p[n] = 0; return p;
}
static void store(struct conn *c, const char *verb, const char *key, unsigned flags, const char *data, size_t n) {
  char hdr[300]; snprintf(hdr, sizeof hdr, "%s %s %u 0 %zu\r\n", verb, key, flags, n);
  sends(c, hdr); sendall(c, data, n); sends(c, "\r\n"); response_values(c);
}

/* ---- phase 1: one connection, every command family ---- */
static void phase1(struct conn *c) {
  cmd(c, "version\r\n");
  store(c, "set", "k1", 0, "hello", 5);
  cmd(c, "get k1\r\n");
  store(c, "add", "k1", 0, "x", 1);                 /* NOT_STORED */
  store(c, "add", "k2", 5, "world", 5);
  store(c, "replace", "k3", 0, "nope", 4);           /* NOT_STORED */
  store(c, "replace", "k2", 6, "WORLD", 5);
  store(c, "append", "k1", 0, "-tail", 5);
  store(c, "prepend", "k1", 0, "head-", 5);
  cmd(c, "get k1 k2 k3\r\n");
  cmd(c, "gets k1\r\n");
  unsigned long long cas = c->last_cas;
  char tmp[200];
  snprintf(tmp, sizeof tmp, "cas k1 0 0 3 %llu\r\nabc\r\n", cas + (perturb_cas ? 1000 : 0)); cmd(c, tmp); /* STORED */
  snprintf(tmp, sizeof tmp, "cas k1 0 0 3 %llu\r\nxyz\r\n", cas); cmd(c, tmp);         /* EXISTS: the token is stale */
  cmd(c, "cas nokey 0 0 1 1\r\nq\r\n");                                                /* NOT_FOUND */
  store(c, "set", "n", 0, "10", 2);
  cmd(c, "incr n 5\r\n"); cmd(c, "decr n 20\r\n");                                     /* 15, then 0 */
  cmd(c, "incr n 18446744073709551615\r\n");                                           /* wraps */
  cmd(c, "incr k2 1\r\n");                                                              /* non-numeric */
  cmd(c, "incr nokey 1\r\n");
  cmd(c, "touch k2 0\r\n"); cmd(c, "touch nokey 0\r\n");
  cmd(c, "gat 0 k2\r\n");
  store(c, "set", "gone", 0, "x", 1);
  cmd(c, "touch gone -1\r\n"); cmd(c, "get gone\r\n");                                 /* expired at once */
  cmd(c, "set past 0 -1 1\r\nx\r\n"); cmd(c, "get past\r\n");                         /* stored expired */
  cmd(c, "delete k2\r\n"); cmd(c, "delete k2\r\n");
  sends(c, "set quiet 0 0 2 noreply\r\nqq\r\n"); cmd(c, "mn\r\n");                    /* noreply, then a no-op */
  cmd(c, "get quiet\r\n");
  /* meta commands, deterministic flags only (no t: it is the clock) */
  cmd(c, "ms mk1 3 F7\r\nabc\r\n");
  cmd(c, "mg mk1 v f s k\r\n");
  cmd(c, "mg mk1 c\r\n");
  cmd(c, "mg missing v\r\n");
  cmd(c, "ms mk2 1 MA\r\nx\r\n");                                                      /* append to nothing */
  cmd(c, "ma mk1\r\n");                                                                 /* not numeric */
  cmd(c, "ms num 1\r\n5\r\n"); cmd(c, "ma num D4 v\r\n"); cmd(c, "ma num MD D2 v\r\n");
  cmd(c, "md mk1\r\n"); cmd(c, "md mk1\r\n");
  cmd(c, "mn\r\n");
  /* errors */
  cmd(c, "bogus command\r\n");
  { char key[300], line[320]; memset(key, 'k', 251); key[251] = 0;   /* one past the 250-byte limit */
    snprintf(line, sizeof line, "get %s\r\n", key); cmd(c, line);
    /* this error is two lines, "CLIENT_ERROR bad command line format\r\n\r\n": take the second here,
       or every later response is read one command late */
    char extra[64]; size_t n = readline(c, extra, sizeof extra); record(c, extra, n); }
  cmd(c, "set badlen 0 0 notanumber\r\n");
  cmd(c, "flush_all\r\n"); cmd(c, "get k1 n\r\n");
}

/* ---- phase 2: values across slab classes, chunked values, one too large ---- */
static void phase2(struct conn *c) {
  static const size_t sizes[] = {1, 100, 1000, 10000, 100000, 300000, 600000, 900000};
  for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
    char key[32]; snprintf(key, sizeof key, "big%u", i);
    char *p = payload(sizes[i], i);
    if (perturb_value && i == 3) p[sizes[i] / 2] ^= 1;
    store(c, "set", key, i, p, sizes[i]); free(p);
  }
  for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) cmdf(c, "get big%u\r\n", i);
  { size_t n = 1536 * 1024; char *p = payload(n, 99); store(c, "set", "toolarge", 0, p, n); free(p); }
  cmd(c, "get toolarge\r\n");
  cmd(c, "mn\r\n");
}

/* ---- phase 3: nconns connections at once, each pipelining a whole batch before reading ---- */
struct worker { struct conn *c; int idx; };
static void *phase3_conn(void *arg) {
  struct worker *w = arg; struct conn *c = w->c;
  struct buf out = {0};
  for (int j = 0; j < 40; j++) {
    char key[64], val[64]; snprintf(key, sizeof key, "c%d:%d", w->idx, j);
    int vn = snprintf(val, sizeof val, "conn%d-item%d", w->idx, j);
    bprintf(&out, "set %s %d 0 %d\r\n%s\r\n", key, j, vn, val);
    bprintf(&out, "get %s\r\n", key);
    if (j % 5 == 0) bprintf(&out, "incr %s 1\r\n", key);
  }
  bprintf(&out, "mn\r\n");
  sendall(c, out.p, out.n);
  for (int j = 0; j < 40; j++) { response_values(c); response_values(c); if (j % 5 == 0) response_values(c); }
  response_values(c);
  free(out.p);
  return NULL;
}

/* ---- phase 4: the stats that do not depend on item or pointer size ---- */
static const char *stat_whitelist[] = {"curr_items", "total_items", "cmd_get", "cmd_set", "get_hits", "get_misses",
  "cas_hits", "cas_badval", "cas_misses", "incr_hits", "incr_misses", "decr_hits", "decr_misses", "touch_hits",
  "touch_misses", "delete_hits", "delete_misses", "evictions", "threads", NULL};
static void phase4(struct conn *c, struct buf *dst, FILE *identity) {
  sends(c, "stats\r\n");
  char line[1024];
  for (;;) {
    size_t n = readline(c, line, sizeof line);
    if (!n || !strcmp(line, "END\r\n")) break;
    char name[128];
    if (sscanf(line, "STAT %127s", name) != 1) continue;
    if (!strcmp(name, "pointer_size")) fputs(line, identity);
    for (int i = 0; stat_whitelist[i]; i++) if (!strcmp(name, stat_whitelist[i])) bput(dst, line, n);
  }
}

static void writefile(const char *name, const char *p, size_t n) {
  char path[1024]; snprintf(path, sizeof path, "%s/%s", outdir, name);
  FILE *f = fopen(path, "w"); if (!f) { perror(path); exit(2); } fwrite(p, 1, n, f); fclose(f);
}
/* what the first connection has so far, so a run that stops early still says where */
static void dump_partial(struct conn *c) {
  writefile("transcript.partial.raw", c->raw.p ? c->raw.p : "", c->raw.n);
  writefile("transcript.partial.norm", c->norm.p ? c->norm.p : "", c->norm.n);
}
/* the one child of `parent`, from /proc/<pid>/stat (field 4 is the ppid); 0 for none, -1 for several */
static pid_t only_child(pid_t parent) {
  DIR *d = opendir("/proc"); struct dirent *e; pid_t found = 0;
  if (!d) return 0;
  while ((e = readdir(d))) {
    char path[300], buf[512]; int pid = atoi(e->d_name); if (pid <= 0) continue;
    snprintf(path, sizeof path, "/proc/%d/stat", pid);
    FILE *f = fopen(path, "r"); if (!f) continue;
    size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
    char *rp = strrchr(buf, ')'); int ppid = 0;
    if (rp && sscanf(rp + 1, " %*c %d", &ppid) == 1 && ppid == parent) found = found ? -1 : pid;
  }
  closedir(d); return found;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

int main(int argc, char **argv) {
  int stopsig = SIGTERM, signal_child = 0, fixture = 0, i;
  for (i = 1; i < argc && strcmp(argv[i], "--"); i++) {
    if (!strcmp(argv[i], "--out")) outdir = argv[++i];
    else if (!strcmp(argv[i], "--port")) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--conns")) nconns = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stop")) stopsig = !strcmp(argv[++i], "USR1") ? SIGUSR1 : SIGTERM;
    else if (!strcmp(argv[i], "--signal-child")) signal_child = 1;
    else if (!strcmp(argv[i], "--fixture")) fixture = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--perturb")) { i++; perturb_value = !strcmp(argv[i], "value"); perturb_cas = !strcmp(argv[i], "cas"); }
    else { fprintf(stderr, "mc-harness: unknown option %s\n", argv[i]); return 2; }
  }
  if (!outdir || i + 1 >= argc || nconns < 1 || nconns > 64) {
    fprintf(stderr, "usage: mc-harness --out DIR [--port P] [--conns N] [--stop TERM|USR1] [--signal-child] [--perturb none|value|cas] -- SERVER...\n");
    return 2;
  }
  char **server = argv + i + 1;
  signal(SIGPIPE, SIG_IGN);
  char path[1024];
  pid_t pid = fork();
  if (pid == 0) {
    setpgid(0, 0);
    snprintf(path, sizeof path, "%s/server.out", outdir); int o = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    snprintf(path, sizeof path, "%s/server.err", outdir); int e = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (o >= 0) dup2(o, 1);
    if (e >= 0) dup2(e, 2);
    execvp(server[0], server); perror("exec"); _exit(127);
  }
  /* up when `version` answers; 120 s for a domain's start */
  struct conn *c0 = calloc(1, sizeof *c0);
  for (int t = 0; t < 1200; t++) {
    c0->fd = dial();
    if (c0->fd >= 0) break;
    int st; if (waitpid(pid, &st, WNOHANG) == pid) { fprintf(stderr, "mc-harness: server exited before listening\n"); return 3; }
    usleep(100 * 1000);
  }
  if (c0->fd < 0) { fprintf(stderr, "mc-harness: server never listened\n"); kill(pid, SIGKILL); return 3; }

  if (fixture) {
    char cmdline[64]; snprintf(cmdline, sizeof cmdline, "mc_capstone_fixture %d\r\n", fixture);
    sends(c0, cmdline);
    struct buf reply = {0};
    while (fill(c0)) { bput(&reply, c0->rb + c0->rpos, c0->rn - c0->rpos); c0->rpos = c0->rn; }
    writefile("fixture-reply.txt", reply.p ? reply.p : "", reply.n);
    close(c0->fd);
    if (timed_out) { fprintf(stderr, "mc-harness: fixture %d: no EOF in 60 s; killing the server\n", fixture); kill(pid, SIGKILL); }
    int st = 0; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) ;
    char status[128];
    int sn = snprintf(status, sizeof status, "stop=none %s=%d\n", WIFSIGNALED(st) ? "signal" : "exit",
                      WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
    writefile("status.txt", status, (size_t)sn);
    printf("mc-harness: fixture %d: %zu reply bytes, %s", fixture, reply.n, status);
    return timed_out ? 5 : 0;
  }
  struct buf all_raw = {0}, all_norm = {0};
  bput(&c0->raw, "== phase 1\n", 11); bput(&c0->norm, "== phase 1\n", 11); phase1(c0); dump_partial(c0);
  bput(&c0->raw, "== phase 2\n", 11); bput(&c0->norm, "== phase 2\n", 11); phase2(c0); dump_partial(c0);
  bput(&all_raw, c0->raw.p, c0->raw.n); bput(&all_norm, c0->norm.p, c0->norm.n);

  struct conn **cs = calloc((size_t)nconns, sizeof *cs);
  struct worker *ws = calloc((size_t)nconns, sizeof *ws);
  pthread_t *ts = calloc((size_t)nconns, sizeof *ts);
  for (int k = 0; k < nconns; k++) {
    cs[k] = calloc(1, sizeof **cs); cs[k]->fd = dial();
    if (cs[k]->fd < 0) {
      fprintf(stderr, "mc-harness: phase 3 connection %d failed\n", k);
      int st; if (waitpid(pid, &st, WNOHANG) == pid) fprintf(stderr, "mc-harness: the server had ended: %s %d\n",
        WIFSIGNALED(st) ? "signal" : "exit", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
      return 4;
    }
    ws[k].c = cs[k]; ws[k].idx = k;
  }
  for (int k = 0; k < nconns; k++) pthread_create(&ts[k], NULL, phase3_conn, &ws[k]);
  for (int k = 0; k < nconns; k++) pthread_join(ts[k], NULL);
  for (int k = 0; k < nconns; k++) {
    char hdr[64]; int n = snprintf(hdr, sizeof hdr, "== phase 3 connection %d\n", k);
    bput(&all_raw, hdr, (size_t)n); bput(&all_raw, cs[k]->raw.p, cs[k]->raw.n);
    bput(&all_norm, hdr, (size_t)n); bput(&all_norm, cs[k]->norm.p, cs[k]->norm.n);
    close(cs[k]->fd);
  }

  snprintf(path, sizeof path, "%s/identity.txt", outdir);
  FILE *identity = fopen(path, "w");
  struct buf stats = {0};
  phase4(c0, &stats, identity); fclose(identity);
  bput(&all_raw, "== phase 4\n", 11); bput(&all_raw, stats.p ? stats.p : "", stats.n);
  bput(&all_norm, "== phase 4\n", 11); bput(&all_norm, stats.p ? stats.p : "", stats.n);
  close(c0->fd);

  pid_t target = pid;
  if (signal_child) {
    target = only_child(pid);
    if (target <= 0) { fprintf(stderr, "mc-harness: --signal-child: %s child of %d\n", target ? "more than one" : "no", (int)pid);
      kill(pid, SIGKILL); return 6; }
    if (getpgid(target) == target) target = -target;   /* capstone-job makes its child a group leader */
    fprintf(stderr, "mc-harness: stop signal to %s %d\n", target < 0 ? "process group" : "process", (int)(target < 0 ? -target : target));
  }
  double t0 = now();
  kill(target, stopsig);
  int st = 0; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) ;
  double t1 = now();
  char status[256];
  int sn = snprintf(status, sizeof status, "stop=%s %s=%d stop_seconds=%.2f\n", stopsig == SIGUSR1 ? "USR1" : "TERM",
                    WIFSIGNALED(st) ? "signal" : "exit", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st), t1 - t0);
  writefile("status.txt", status, (size_t)sn);
  writefile("transcript.raw", all_raw.p, all_raw.n);
  writefile("transcript.norm", all_norm.p, all_norm.n);
  printf("mc-harness: %zu bytes of transcript, %s", all_norm.n, status);
  if (timed_out) { printf("mc-harness: a read timed out (see <TIMEOUT> in the transcript)\n"); return 5; }
  return 0;
}
