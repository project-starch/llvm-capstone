/* OS-facing stubs. Every one of these is on a path the domain does not take; they exist so
 * the engine LINKS. Each traps loudly rather than returning a plausible value, so if the
 * engine ever does reach one we find out instead of getting a quiet wrong answer.
 *
 * The three PHP symbols at the bottom are the files deliberately excluded from the build
 * (reflection, highlight, the ini scanner) -- see compile-sweep.sh's TU list.
 */
typedef unsigned long size_t;

/* Set by the domain before zend_startup; read by the fault reporter below. */
volatile unsigned php_capstone_trap_code;
#define TRAP(code) do { php_capstone_trap_code = (code); __builtin_trap(); } while (0)

/* --- process --- */
void exit(int status)  { (void)status; TRAP(0xE1); __builtin_unreachable(); }
void abort(void)       { TRAP(0xE2); __builtin_unreachable(); }

/* --- errno --- */
int errno;

/* --- file I/O ---------------------------------------------------------------
 * Unreachable by construction: the script is compiled with compile_string() from a
 * .rodata buffer, so zend_stream / zend_language_scanner's fopen path is never entered.
 * See the plan's Phase 2 "script input" note. */
typedef struct _IO_FILE FILE;
FILE *stdin;
FILE *stdout;
FILE *stderr;
FILE  *fopen(const char *p, const char *m)              { (void)p;(void)m; TRAP(0xF0); return 0; }
FILE  *fdopen(int fd, const char *m)                    { (void)fd;(void)m; TRAP(0xF1); return 0; }
int    fclose(FILE *f)                                  { (void)f; TRAP(0xF2); return 0; }
size_t fread(void *b, size_t s, size_t n, FILE *f)      { (void)b;(void)s;(void)n;(void)f; TRAP(0xF3); return 0; }
size_t fwrite(const void *b, size_t s, size_t n, FILE *f){(void)b;(void)s;(void)n;(void)f; TRAP(0xF4); return 0; }
int    fileno(FILE *f)                                  { (void)f; TRAP(0xF5); return -1; }
int    isatty(int fd)                                   { (void)fd; return 0; }
int    fflush(FILE *f)                                  { (void)f; return 0; }

/* --- signals / timers -------------------------------------------------------
 * zend_execute_API.c:1220-1242 (max_execution_time watchdog) and :128 (debug SIGSEGV
 * handler). Harmless no-ops: the watchdog simply never fires in a domain. */
typedef void (*__sighandler_t)(int);
typedef unsigned long sigset_t;
struct itimerval;
__sighandler_t signal(int s, __sighandler_t h)          { (void)s; return h; }
__sighandler_t sigset(int s, __sighandler_t h)          { (void)s; return h; }
int sigemptyset(sigset_t *s)                            { if (s) *s = 0; return 0; }
int sigaddset(sigset_t *s, int n)                       { (void)s;(void)n; return 0; }
int sigprocmask(int h, const sigset_t *a, sigset_t *b)  { (void)h;(void)a;(void)b; return 0; }
int setitimer(int w, const struct itimerval *a, struct itimerval *b) { (void)w;(void)a;(void)b; return 0; }

/* --- dynamic loading --------------------------------------------------------
 * A domain is one static image. Returning NULL is the honest answer and PHP handles it:
 * zend_extensions.c treats a NULL handle as "extension not loadable".
 * These MUST be real declarations returning void* -- see stubinc/dlfcn.h on why an
 * implicit int return would truncate a capability. */
void *dlopen(const char *f, int flag)   { (void)f;(void)flag;
#ifdef PHP_CAPSTONE_TRAP_DLOPEN
  /* Diagnostic: is zend_load_extension reached at all? With extensions == NULL it should
   * not be. Trapping here answers that in one boot instead of inferring it from a pc that
   * QEMU only reports to translation-block granularity. */
  TRAP(0xD0);
#endif
  return 0; }
void *dlsym(void *h, const char *s)     { (void)h;(void)s; return 0; }
int   dlclose(void *h)                  { (void)h; return 0; }
char *dlerror(void)                     { return 0; }

/* The three PHP symbols for the excluded TUs are NOT here: they need PHP's own header
 * types and live in php_capstone_php_stubs.c. Sizing one of them by eye here is exactly
 * what cost a debugging cycle -- see that file's header comment. */
