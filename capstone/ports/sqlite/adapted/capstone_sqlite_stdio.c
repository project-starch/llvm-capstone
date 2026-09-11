/* The formatter behind capstone_sqlite_stdio.h. See that header for what is supported and why the
 * set is this small. No libc is used here: this file is compiled into a freestanding capability
 * domain and everything it needs it builds itself. */
#include "capstone_sqlite_stdio.h"

unsigned long capstone_stdio_dropped = 0;

/* Distinct addresses, no state. See the header. */
static struct capstone_sqlite_file capstone_stdout_obj_;
static struct capstone_sqlite_file capstone_stderr_obj_;
FILE *const capstone_stdout = &capstone_stdout_obj_;
FILE *const capstone_stderr = &capstone_stderr_obj_;

/* One line's worth. Formatting is assembled here and handed to the sink in one call, so a partial
 * conversion can never be split across two sink calls and reordered by a caller in between. 512 is
 * comfortably above speedtest1's longest line (a 28-column name plus a time field). A format that
 * would exceed it is TRUNCATED AND COUNTED rather than silently shortened. */
#define CAPSTONE_STDIO_LINE 512

typedef struct {
  char buf[CAPSTONE_STDIO_LINE];
  unsigned long len;
  unsigned long over;      /* characters that did not fit */
} lbuf;

static void lb_putc(lbuf *b, char c) {
  if (b->len < CAPSTONE_STDIO_LINE)
    b->buf[b->len++] = c;
  else
    b->over++;
}

static void lb_pad(lbuf *b, char c, long n) {
  while (n-- > 0)
    lb_putc(b, c);
}

/* Unsigned to text in base 10 or 16, lowest digit first into tmp, then reversed. No division helper
 * from libc, and no 128-bit anything: base is 10 or 16 only. */
static unsigned long u_to_text(unsigned long long v, unsigned base, char *tmp) {
  const char *digits = "0123456789abcdef";
  unsigned long n = 0;
  if (v == 0) {
    tmp[n++] = '0';
    return n;
  }
  while (v) {
    tmp[n++] = digits[v % base];
    v /= base;
  }
  return n;
}

static void emit_number(lbuf *b, unsigned long long v, unsigned base, int negative,
                        long width, int left, int zero, long prec) {
  char tmp[32];
  unsigned long n = u_to_text(v, base, tmp);
  long digits = (long)n;
  long body;

  /* A precision on an integer is a MINIMUM digit count, and it suppresses zero-padding. */
  long zeros = (prec >= 0 && prec > digits) ? prec - digits : 0;
  if (prec >= 0)
    zero = 0;

  body = digits + zeros + (negative ? 1 : 0);

  if (!left && !zero)
    lb_pad(b, ' ', width - body);
  if (negative)
    lb_putc(b, '-');
  if (!left && zero)
    lb_pad(b, '0', width - body);
  lb_pad(b, '0', zeros);
  while (n--)
    lb_putc(b, tmp[n]);
  if (left)
    lb_pad(b, ' ', width - body);
}

static void emit_string(lbuf *b, const char *s, long width, int left, long prec) {
  long n = 0;
  if (!s)
    s = "(null)";
  while (s[n] && (prec < 0 || n < prec))
    n++;
  if (!left)
    lb_pad(b, ' ', width - n);
  for (long i = 0; i < n; i++)
    lb_putc(b, s[i]);
  if (left)
    lb_pad(b, ' ', width - n);
}

int capstone_vfprintf(FILE *stream, const char *fmt, capstone_va_list ap) {
  lbuf b;
  (void)stream;                 /* both streams share one sink; see the header */
  b.len = 0;
  b.over = 0;

  for (const char *p = fmt; p && *p; p++) {
    long width = 0, prec = -1;
    int left = 0, zero = 0, lng = 0;

    if (*p != '%') {
      lb_putc(&b, *p);
      continue;
    }
    p++;
    if (*p == '%') {            /* the one flag-free case */
      lb_putc(&b, '%');
      continue;
    }
    for (;; p++) {              /* flags */
      if (*p == '-') left = 1;
      else if (*p == '0') zero = 1;
      else break;
    }
    if (*p == '*') {            /* star WIDTH; not used by speedtest1 but free to support here */
      width = (long)__builtin_va_arg(ap, int);
      if (width < 0) { left = 1; width = -width; }
      p++;
    } else {
      while (*p >= '0' && *p <= '9')
        width = width * 10 + (*p++ - '0');
    }
    if (*p == '.') {            /* precision, including the %.*s speedtest1 uses */
      p++;
      prec = 0;
      if (*p == '*') {
        prec = (long)__builtin_va_arg(ap, int);
        p++;
      } else {
        while (*p >= '0' && *p <= '9')
          prec = prec * 10 + (*p++ - '0');
      }
      if (prec < 0)
        prec = -1;
    }
    while (*p == 'l') {         /* 'l' and 'll' */
      lng++;
      p++;
    }
    if (*p == 'h')              /* accepted and ignored: promotion makes it a no-op here */
      p++;

    switch (*p) {
      case 'd': {
        long long v = lng ? __builtin_va_arg(ap, long long) : (long long)__builtin_va_arg(ap, int);
        int neg = v < 0;
        unsigned long long m = neg ? (unsigned long long)(-(v + 1)) + 1ULL : (unsigned long long)v;
        emit_number(&b, m, 10, neg, width, left, zero, prec);
        break;
      }
      case 'u': {
        unsigned long long v = lng ? __builtin_va_arg(ap, unsigned long long)
                                   : (unsigned long long)__builtin_va_arg(ap, unsigned int);
        emit_number(&b, v, 10, 0, width, left, zero, prec);
        break;
      }
      case 'x': {
        unsigned long long v = lng ? __builtin_va_arg(ap, unsigned long long)
                                   : (unsigned long long)__builtin_va_arg(ap, unsigned int);
        emit_number(&b, v, 16, 0, width, left, zero, prec);
        break;
      }
      case 's':
        emit_string(&b, __builtin_va_arg(ap, const char *), width, left, prec);
        break;
      default:
        /* AN UNSUPPORTED CONVERSION IS LOUD. Printing it verbatim would produce plausible-looking
         * output with a silently missing value, and the argument list would then be misaligned for
         * everything after it -- every later field wrong, nothing obviously broken. */
        lb_putc(&b, '%');
        lb_putc(&b, *p ? *p : '?');
        capstone_stdio_dropped++;
        break;
    }
    if (!*p)
      break;
  }

  {
    unsigned long took = capstone_stdio_sink(b.buf, b.len);
    capstone_stdio_dropped += (b.len - took) + b.over;
    return (int)took;
  }
}

int capstone_printf(const char *fmt, ...) {
  capstone_va_list ap;
  int n;
  __builtin_va_start(ap, fmt);
  n = capstone_vfprintf(capstone_stdout, fmt, ap);
  __builtin_va_end(ap);
  return n;
}

int capstone_fprintf(FILE *stream, const char *fmt, ...) {
  capstone_va_list ap;
  int n;
  __builtin_va_start(ap, fmt);
  n = capstone_vfprintf(stream, fmt, ap);
  __builtin_va_end(ap);
  return n;
}

int capstone_fflush(FILE *stream) {
  (void)stream;                 /* the sink is not buffered */
  return 0;
}

/* speedtest1 calls exit() only from fatal_error, i.e. after an SQL error it cannot continue past.
 * A domain cannot exit a process. Returning would let the caller carry on as if nothing had
 * happened; spinning would WEDGE THE CORE, and on the board a wedge costs every stage after it in
 * that boot. So the domain gets to unwind and report, and the spin is only the last resort for a
 * hook that wrongly returns. */
volatile int capstone_stdio_exit_code = -1;
void capstone_exit(int code) {
  capstone_stdio_exit_code = code;
  capstone_stdio_on_exit(code);
  for (;;)
    ;
}

int capstone_atoi(const char *s) {
  int v = 0, neg = 0;
  if (!s)
    return 0;
  while (*s == ' ' || *s == '\t')
    s++;
  if (*s == '-') { neg = 1; s++; }
  else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9')
    v = v * 10 + (*s++ - '0');
  return neg ? -v : v;
}

/* No files in a domain. These fail rather than pretend; see the header for why they exist at all. */
FILE *capstone_fopen(const char *path, const char *mode) {
  (void)path; (void)mode;
  return 0;
}

int capstone_fclose(FILE *stream) {
  (void)stream;
  return 0;
}

unsigned long capstone_fwrite(const void *ptr, unsigned long size, unsigned long n, FILE *stream) {
  (void)ptr; (void)size; (void)n; (void)stream;
  return 0;
}

int capstone_unlink(const char *path) {
  (void)path;
  return 0;
}
