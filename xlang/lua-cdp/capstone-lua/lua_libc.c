/* Freestanding libc gap-fill for the reference-Lua-5.4.7 Capstone domain.
 *
 * Implements ONLY what Lua's core + base lib need that is NOT already provided
 * by another linked TU. Do NOT duplicate those here — they would collide at link:
 *   - memcpy/memmove/memset/memcmp/strlen/strcmp/strcpy  -> beebs_freestanding_string.c
 *   - floor/ceil/fabs/sqrt/exp/log/pow                    -> beebs_softfloat_libm.c
 *   - malloc/free/realloc                                 -> xlang/common/revoke_arena_domain.c
 *   - setjmp/longjmp                                      -> capstone_setjmp.S
 *
 * Compiled with the same freestanding recipe and the force-included
 * capstone_lua_libc.h (which supplies size_t/va_list, the FILE typedef, EOF/NULL,
 * and the prototypes below), so every definition here matches a declaration there.
 *
 * Self-check: build native and run the asserts —
 *   cc -DLUA_LIBC_SELFTEST -O2 lua_libc.c -lm -o /tmp/lua_libc_t && /tmp/lua_libc_t
 */
#ifdef LUA_LIBC_SELFTEST
/* Native self-test build: real stdio for printf, real libm for floor/ceil/fabs so
 * the ACTUAL freestanding dtoa below is exercised, and our own snprintf/ctype/
 * string under test. <string.h> is deliberately NOT included — its fortify
 * _Generic macros would collide with our same-named definitions. */
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
static int streq(const char *a, const char *b) {
  while (*a && *a == *b) {
    a++;
    b++;
  }
  return *a == *b;
}
#else
/* Freestanding domain build. capstone_lua_libc.h is force-included by the build. */
#include <stddef.h>
#include <stdarg.h>
/* From beebs_softfloat_libm.c — declared here so we can use them in dtoa/fmod. */
double floor(double), ceil(double), fabs(double);
#endif

/* ------------------------------------------------------------------ ctype -- */
/* ASCII-only, which is all reference Lua assumes on a freestanding target. */
int isdigit(int c) { return c >= '0' && c <= '9'; }
int isupper(int c) { return c >= 'A' && c <= 'Z'; }
int islower(int c) { return c >= 'a' && c <= 'z'; }
int isalpha(int c) { return isupper(c) || islower(c); }
int isalnum(int c) { return isalpha(c) || isdigit(c); }
int isspace(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
int iscntrl(int c) { return (c >= 0 && c < 32) || c == 127; }
int isxdigit(int c) {
  return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
int isgraph(int c) { return c > ' ' && c < 127; }
int isprint(int c) { return c >= ' ' && c < 127; }
int ispunct(int c) { return isgraph(c) && !isalnum(c); }
int toupper(int c) { return islower(c) ? c - ('a' - 'A') : c; }
int tolower(int c) { return isupper(c) ? c + ('a' - 'A') : c; }

/* ----------------------------------------------------------------- string -- */
void *memchr(const void *s, int c, size_t n) {
  const unsigned char *p = (const unsigned char *)s;
  for (size_t i = 0; i < n; i++)
    if (p[i] == (unsigned char)c)
      return (void *)(p + i);
  return NULL;
}

int strncmp(const char *a, const char *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    unsigned char ca = (unsigned char)a[i], cb = (unsigned char)b[i];
    if (ca != cb)
      return (int)ca - (int)cb;
    if (ca == 0)
      return 0;
  }
  return 0;
}

char *strchr(const char *s, int c) {
  for (;; s++) {
    if (*s == (char)c)
      return (char *)s;
    if (*s == 0)
      return NULL;
  }
}

char *strpbrk(const char *s, const char *set) {
  for (; *s; s++)
    for (const char *t = set; *t; t++)
      if (*s == *t)
        return (char *)s;
  return NULL;
}

size_t strspn(const char *s, const char *set) {
  size_t n = 0;
  for (; s[n]; n++) {
    const char *t = set;
    for (; *t && *t != s[n]; t++)
      ;
    if (*t == 0)
      break;
  }
  return n;
}

char *strstr(const char *hay, const char *needle) {
  if (*needle == 0)
    return (char *)hay;
  for (; *hay; hay++) {
    const char *h = hay, *n = needle;
    while (*h && *n && *h == *n) {
      h++;
      n++;
    }
    if (*n == 0)
      return (char *)hay;
  }
  return NULL;
}

/* strcoll: no locale on a freestanding domain, so a plain byte compare (== the
 * C locale, which is exactly what Lua expects when LC_COLLATE is "C"). */
int strcoll(const char *a, const char *b) {
  while (*a && *a == *b) {
    a++;
    b++;
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/* ----------------------------------------------------------------- stdlib -- */
int abs(int x) { return x < 0 ? -x : x; }

#ifndef LUA_LIBC_SELFTEST
double strtod(const char *s, char **endptr) {
  const char *p = s;
  while (*p == ' ' || (*p >= '\t' && *p <= '\r'))
    p++;
  int neg = 0;
  if (*p == '+' || *p == '-') {
    neg = (*p == '-');
    p++;
  }
  double val = 0.0;
  int any = 0;
  for (; isdigit((unsigned char)*p); p++) {
    val = val * 10.0 + (*p - '0');
    any = 1;
  }
  if (*p == '.') {
    p++;
    double scale = 0.1;
    for (; isdigit((unsigned char)*p); p++) {
      val += (*p - '0') * scale;
      scale *= 0.1;
      any = 1;
    }
  }
  if (any && (*p == 'e' || *p == 'E')) {
    const char *e = p + 1;
    int eneg = 0;
    if (*e == '+' || *e == '-') {
      eneg = (*e == '-');
      e++;
    }
    if (isdigit((unsigned char)*e)) {
      int exp = 0;
      for (; isdigit((unsigned char)*e); e++)
        exp = exp * 10 + (*e - '0');
      double f = 1.0, b = 10.0;
      for (int k = exp; k; k >>= 1) {
        if (k & 1)
          f *= b;
        b *= b;
      }
      val = eneg ? val / f : val * f;
      p = e;
    }
  }
  if (endptr)
    *endptr = (char *)(any ? p : s); /* nothing parsed -> endptr == s */
  return neg ? -val : val;
}
/* ponytail: decimal only. Hex floats ("0x1p4") never reach here — Lua 5.4 ships
 * its own lua_strx2number for the 'x' path (lobject.c). inf/nan are rejected by
 * l_str2d before it calls strtod. Upgrade path: add hex-mantissa parsing if a
 * future engine feeds hex floats through strtod directly. */
#endif

/* ------------------------------------------------------------------- math -- */
#ifndef LUA_LIBC_SELFTEST
static double m_trunc(double x) { return x < 0 ? ceil(x) : floor(x); }

double fmod(double x, double y) {
  if (y == 0.0)
    return 0.0 / 0.0; /* NaN, as C requires */
  return x - m_trunc(x / y) * y;
}

double frexp(double x, int *e) {
  double a = fabs(x);
  int ex = 0;
  if (x == 0.0 || a != a || a == a * 2.0) { /* 0, NaN, or +/-inf */
    *e = 0;
    return x;
  }
  while (a >= 1.0) {
    a *= 0.5;
    ex++;
  }
  while (a < 0.5) {
    a *= 2.0;
    ex--;
  }
  *e = ex;
  return x < 0 ? -a : a;
}

double ldexp(double x, int e) {
  double r = x;
  if (e > 0)
    while (e--)
      r *= 2.0;
  else
    while (e++)
      r *= 0.5;
  return r;
}

double modf(double x, double *ip) {
  double t = m_trunc(x);
  *ip = t;
  return x - t;
}
#endif

/* -------------------------------------------------- snprintf / vsnprintf -- */
struct outbuf {
  char *buf;
  size_t cap;
  size_t len; /* total chars that WOULD be written (C99 return) */
};

static void ob_putc(struct outbuf *o, char c) {
  if (o->len + 1 < o->cap)
    o->buf[o->len] = c;
  o->len++;
}
static void ob_puts(struct outbuf *o, const char *s) {
  while (*s)
    ob_putc(o, *s++);
}

/* Emit a numeric body with width/flags/precision. `body` is the unsigned-magnitude
 * digits; `prefix` is a sign or "0x"/"0" adornment placed before zero-padding. */
static void ob_padnum(struct outbuf *o, const char *prefix, const char *body,
                      int width, int prec, int left, int zero) {
  int blen = 0, plen = 0;
  for (const char *p = body; *p; p++)
    blen++;
  for (const char *p = prefix; *p; p++)
    plen++;
  int zeros = 0;
  if (prec >= 0 && blen < prec)
    zeros = prec - blen; /* precision => minimum digits */
  else if (zero && !left && prec < 0) {
    int total = plen + blen;
    if (total < width)
      zeros = width - total; /* '0' flag pads with zeros after the prefix */
  }
  int content = plen + zeros + blen;
  int pad = width - content;
  if (pad < 0)
    pad = 0;
  if (!left)
    for (int i = 0; i < pad; i++)
      ob_putc(o, ' ');
  ob_puts(o, prefix);
  for (int i = 0; i < zeros; i++)
    ob_putc(o, '0');
  ob_puts(o, body);
  if (left)
    for (int i = 0; i < pad; i++)
      ob_putc(o, ' ');
}

/* Unsigned magnitude of a value into `out` (reversed then corrected) in `base`. */
static void utoa_base(unsigned long long v, unsigned base, int upper, char *out) {
  const char *digs = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  char tmp[24];
  int n = 0;
  if (v == 0)
    tmp[n++] = '0';
  while (v) {
    tmp[n++] = digs[v % base];
    v /= base;
  }
  int i = 0;
  while (n)
    out[i++] = tmp[--n];
  out[i] = 0;
}

/* --- double formatting (%f/%e/%g) ----------------------------------------- */
static double ipow10(int n) {
  double r = 1.0, b = 10.0;
  int neg = n < 0;
  if (neg)
    n = -n;
  for (int k = n; k; k >>= 1) {
    if (k & 1)
      r *= b;
    b *= b;
  }
  return neg ? 1.0 / r : r;
}
/* Decimal digits (MSB-first) of a non-negative integral double. */
static int dbl_digits(double whole, char *out) {
  if (whole < 1.0) {
    out[0] = '0';
    out[1] = 0;
    return 1;
  }
  char rev[512];
  int rn = 0;
  while (whole >= 1.0 && rn < 511) {
    double q = floor(whole / 10.0);
    int d = (int)(whole - q * 10.0);
    rev[rn++] = (char)('0' + d);
    whole = q;
  }
  for (int i = 0; i < rn; i++)
    out[i] = rev[rn - 1 - i];
  out[rn] = 0;
  return rn;
}
/* Fixed-point magnitude (x >= 0) into dst with `prec` fraction digits. */
static int emit_fixed(char *dst, double x, int prec, int forcedot) {
  char digs[560];
  double scaled = floor(x * ipow10(prec) + 0.5);
  int nd = dbl_digits(scaled, digs);
  int intlen = nd - prec;
  int k = 0;
  if (intlen <= 0) {
    dst[k++] = '0';
    if (prec > 0 || forcedot)
      dst[k++] = '.';
    for (int i = 0; i < -intlen; i++)
      dst[k++] = '0';
    for (int i = 0; i < nd; i++)
      dst[k++] = digs[i];
  } else {
    for (int i = 0; i < intlen; i++)
      dst[k++] = digs[i];
    if (prec > 0 || forcedot) {
      dst[k++] = '.';
      for (int i = intlen; i < nd; i++)
        dst[k++] = digs[i];
    }
  }
  dst[k] = 0;
  return k;
}
/* Scientific magnitude (x >= 0) into dst: d.ddde+NN with `prec` fraction digits. */
static int emit_sci(char *dst, double x, int prec, int upper, int forcedot) {
  int e = 0;
  double m = x;
  if (m != 0.0) {
    while (m >= 10.0) {
      m /= 10.0;
      e++;
    }
    while (m < 1.0) {
      m *= 10.0;
      e--;
    }
  }
  char digs[560];
  double r = floor(m * ipow10(prec) + 0.5);
  int nd = dbl_digits(r, digs);
  if (nd > prec + 1) { /* rounding carried 9.99..->10.0.. : renormalise */
    e++;
    nd = prec + 1; /* drop the trailing zero the carry produced */
  }
  while (nd < prec + 1) /* pad (only when x==0) */
    digs[nd++] = '0';
  digs[nd] = 0;
  int k = 0;
  dst[k++] = digs[0];
  if (prec > 0 || forcedot) {
    dst[k++] = '.';
    for (int i = 1; i <= prec; i++)
      dst[k++] = digs[i];
  }
  dst[k++] = upper ? 'E' : 'e';
  if (e < 0) {
    dst[k++] = '-';
    e = -e;
  } else
    dst[k++] = '+';
  char eb[8];
  int en = 0;
  if (e == 0)
    eb[en++] = '0';
  while (e) {
    eb[en++] = (char)('0' + e % 10);
    e /= 10;
  }
  while (en < 2)
    eb[en++] = '0';
  while (en)
    dst[k++] = eb[--en];
  dst[k] = 0;
  return k;
}
/* %g magnitude (x >= 0): shortest of %e/%f for P significant digits, zeros stripped. */
static int emit_g(char *dst, double x, int P, int upper, int alt) {
  if (P <= 0)
    P = 1;
  int e = 0;
  double m = x;
  if (m != 0.0) {
    while (m >= 10.0) {
      m /= 10.0;
      e++;
    }
    while (m < 1.0) {
      m *= 10.0;
      e--;
    }
  }
  /* Does rounding to P sig digits carry into the next exponent? */
  char probe[560];
  double r = floor(m * ipow10(P - 1) + 0.5);
  int nd = dbl_digits(r, probe);
  int X = e + (nd > P ? 1 : 0);
  int len;
  if (X < -4 || X >= P)
    len = emit_sci(dst, x, P - 1, upper, alt);
  else
    len = emit_fixed(dst, x, P - 1 - X, alt);
  if (!alt) { /* strip trailing zeros in the fraction, and a bare '.' */
    int dot = -1, ee = -1;
    for (int i = 0; i < len; i++) {
      if (dst[i] == '.')
        dot = i;
      else if (dst[i] == 'e' || dst[i] == 'E')
        ee = i;
    }
    if (dot >= 0) {
      int fracend = (ee >= 0) ? ee : len;
      int last = fracend - 1;
      while (last > dot && dst[last] == '0')
        last--;
      if (last == dot)
        last--; /* remove the '.' too */
      int tail = len - fracend;
      for (int i = 0; i < tail; i++)
        dst[last + 1 + i] = dst[fracend + i];
      len = last + 1 + tail;
      dst[len] = 0;
    }
  }
  return len;
}

int lua_vsnprintf_impl(char *buf, size_t size, const char *fmt, va_list ap);

int lua_vsnprintf_impl(char *buf, size_t size, const char *fmt, va_list ap) {
  struct outbuf o = {buf, size, 0};
  for (const char *f = fmt; *f; f++) {
    if (*f != '%') {
      ob_putc(&o, *f);
      continue;
    }
    f++;
    if (*f == '%') {
      ob_putc(&o, '%');
      continue;
    }
    int left = 0, zero = 0, plus = 0, space = 0, alt = 0;
    for (;; f++) {
      if (*f == '-')
        left = 1;
      else if (*f == '0')
        zero = 1;
      else if (*f == '+')
        plus = 1;
      else if (*f == ' ')
        space = 1;
      else if (*f == '#')
        alt = 1;
      else
        break;
    }
    int width = 0;
    if (*f == '*') {
      width = va_arg(ap, int);
      if (width < 0) {
        left = 1;
        width = -width;
      }
      f++;
    } else
      while (isdigit((unsigned char)*f))
        width = width * 10 + (*f++ - '0');
    int prec = -1;
    if (*f == '.') {
      f++;
      prec = 0;
      if (*f == '*') {
        prec = va_arg(ap, int);
        f++;
      } else
        while (isdigit((unsigned char)*f))
          prec = prec * 10 + (*f++ - '0');
      if (prec < 0)
        prec = -1;
    }
    int lmod = 0; /* 0=int, 1=long, 2=long long, 3=size_t */
    if (*f == 'l') {
      lmod = 1;
      f++;
      if (*f == 'l') {
        lmod = 2;
        f++;
      }
    } else if (*f == 'z') {
      lmod = 3;
      f++;
    } else if (*f == 'h') {
      f++;
      if (*f == 'h')
        f++;
    } else if (*f == 'L' || *f == 'j' || *f == 't') {
      f++;
    }
    char conv = *f;
    char body[600];
    char prefix[4];
    prefix[0] = 0;
    switch (conv) {
    case 'd':
    case 'i': {
      long long v;
      if (lmod == 2)
        v = va_arg(ap, long long);
      else if (lmod == 1)
        v = va_arg(ap, long);
      else if (lmod == 3)
        v = (long long)va_arg(ap, size_t);
      else
        v = va_arg(ap, int);
      unsigned long long mag = v < 0 ? (unsigned long long)(-(v + 1)) + 1ULL
                                     : (unsigned long long)v;
      utoa_base(mag, 10, 0, body);
      int i = 0;
      if (v < 0)
        prefix[i++] = '-';
      else if (plus)
        prefix[i++] = '+';
      else if (space)
        prefix[i++] = ' ';
      prefix[i] = 0;
      ob_padnum(&o, prefix, body, width, prec, left, zero);
      break;
    }
    case 'u':
    case 'x':
    case 'X':
    case 'o': {
      unsigned long long v;
      if (lmod == 2)
        v = va_arg(ap, unsigned long long);
      else if (lmod == 1)
        v = va_arg(ap, unsigned long);
      else if (lmod == 3)
        v = (unsigned long long)va_arg(ap, size_t);
      else
        v = va_arg(ap, unsigned int);
      unsigned base = (conv == 'o') ? 8 : (conv == 'u') ? 10 : 16;
      utoa_base(v, base, conv == 'X', body);
      int i = 0;
      if (alt && v != 0 && (conv == 'x' || conv == 'X')) {
        prefix[i++] = '0';
        prefix[i++] = (conv == 'X') ? 'X' : 'x';
      }
      prefix[i] = 0;
      ob_padnum(&o, prefix, body, width, prec, left, zero);
      break;
    }
    case 'c': {
      char ch = (char)va_arg(ap, int);
      char one[2] = {ch, 0};
      ob_padnum(&o, "", one, width, -1, left, 0);
      break;
    }
    case 's': {
      const char *s = va_arg(ap, const char *);
      if (!s)
        s = "(null)";
      int slen = 0;
      while (s[slen] && (prec < 0 || slen < prec))
        slen++;
      int pad = width - slen;
      if (pad < 0)
        pad = 0;
      if (!left)
        for (int i = 0; i < pad; i++)
          ob_putc(&o, ' ');
      for (int i = 0; i < slen; i++)
        ob_putc(&o, s[i]);
      if (left)
        for (int i = 0; i < pad; i++)
          ob_putc(&o, ' ');
      break;
    }
    case 'p': {
      void *ptr = va_arg(ap, void *);
      unsigned long long v = (unsigned long long)(unsigned long)ptr;
      if (!ptr)
        ob_puts(&o, "(nil)");
      else {
        utoa_base(v, 16, 0, body);
        ob_padnum(&o, "0x", body, width, -1, left, zero);
      }
      break;
    }
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G': {
      double d = va_arg(ap, double);
      int neg = 0;
      if (d < 0.0 || (d == 0.0 && 1.0 / d < 0.0)) {
        neg = 1;
        d = -d;
      }
      int up = (conv == 'F' || conv == 'E' || conv == 'G');
      /* Guard non-finite BEFORE the emit_* routines: normalize()'s scaling loop
       * would spin forever on +inf (inf/10 == inf). nan compares false everywhere
       * so it would misformat rather than hang, but handle it here too. */
      if (d != d) {
        ob_padnum(&o, "", up ? "NAN" : "nan", width, -1, left, 0);
        break;
      }
      if (d != 0.0 && d + d == d) { /* +/-inf: finite x has x+x != x */
        const char *pf = neg ? "-" : (plus ? "+" : (space ? " " : ""));
        ob_padnum(&o, pf, up ? "INF" : "inf", width, -1, left, 0);
        break;
      }
      int p = prec < 0 ? 6 : prec;
      if (conv == 'f' || conv == 'F')
        emit_fixed(body, d, p, alt);
      else if (conv == 'e' || conv == 'E')
        emit_sci(body, d, p, up, alt);
      else
        emit_g(body, d, p, up, alt);
      int i = 0;
      if (neg)
        prefix[i++] = '-';
      else if (plus)
        prefix[i++] = '+';
      else if (space)
        prefix[i++] = ' ';
      prefix[i] = 0;
      ob_padnum(&o, prefix, body, width, -1, left, zero);
      break;
    }
    case 0:
      f--; /* trailing '%' */
      break;
    default:
      ob_putc(&o, '%');
      ob_putc(&o, conv);
      break;
    }
  }
  if (o.cap > 0)
    o.buf[o.len < o.cap ? o.len : o.cap - 1] = 0;
  return (int)o.len;
}

#ifdef LUA_LIBC_SELFTEST
int lua_vsnprintf(char *b, size_t s, const char *f, va_list ap) {
  return lua_vsnprintf_impl(b, s, f, ap);
}
int lua_snprintf(char *b, size_t s, const char *f, ...) {
  va_list ap;
  va_start(ap, f);
  int r = lua_vsnprintf_impl(b, s, f, ap);
  va_end(ap);
  return r;
}
#else
int vsnprintf(char *b, size_t s, const char *f, va_list ap) {
  return lua_vsnprintf_impl(b, s, f, ap);
}
int snprintf(char *b, size_t s, const char *f, ...) {
  va_list ap;
  va_start(ap, f);
  int r = lua_vsnprintf_impl(b, s, f, ap);
  va_end(ap);
  return r;
}
#endif

/* --------------------------------------------------- stdio / locale / os -- */
#ifndef LUA_LIBC_SELFTEST
struct capstone_lua_FILE {
  int which;
};
static struct capstone_lua_FILE __std_in = {0};
static struct capstone_lua_FILE __std_out = {1};
static struct capstone_lua_FILE __std_err = {2};
FILE *stdin = &__std_in;
FILE *stdout = &__std_out;
FILE *stderr = &__std_err;

int errno = 0;

/* print()/io.write route through fwrite(stdout); the domain implements the sink. */
extern void lua_host_write(const char *s, unsigned long n);

unsigned long fwrite(const void *ptr, unsigned long size, unsigned long nmemb,
                     FILE *stream) {
  (void)stream;
  unsigned long total = size * nmemb;
  if (total)
    lua_host_write((const char *)ptr, total);
  return nmemb;
}
int fflush(FILE *stream) {
  (void)stream;
  return 0;
}
int fprintf(FILE *stream, const char *fmt, ...) {
  (void)stream;
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int n = lua_vsnprintf_impl(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  unsigned long w = (unsigned long)(n < (int)sizeof tmp ? n : (int)sizeof tmp - 1);
  lua_host_write(tmp, w);
  return n;
}

/* File I/O is never reached: scripts load via luaL_loadbufferx, not the file
 * loader. These exist only so lauxlib/loslib link; each returns an error. */
FILE *fopen(const char *a, const char *b) {
  (void)a;
  (void)b;
  return NULL;
}
FILE *freopen(const char *a, const char *b, FILE *c) {
  (void)a;
  (void)b;
  (void)c;
  return NULL;
}
int fclose(FILE *a) {
  (void)a;
  return 0;
}
unsigned long fread(void *a, unsigned long b, unsigned long c, FILE *d) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  return 0;
}
int getc(FILE *a) {
  (void)a;
  return -1; /* EOF */
}
int ungetc(int a, FILE *b) {
  (void)a;
  (void)b;
  return -1;
}
int feof(FILE *a) {
  (void)a;
  return 1;
}
int ferror(FILE *a) {
  (void)a;
  return 0;
}

char *strerror(int e) {
  (void)e;
  return (char *)"error";
}

/* Sort-pivot randomization and the GC seed: fixed, deterministic constants — a
 * freestanding domain has no clock. Determinism is a feature for a test domain. */
long clock(void) { return 0; }
long time(long *t) {
  if (t)
    *t = 0;
  return 0;
}

static struct lconv __lconv = {(char *)"."};
struct lconv *localeconv(void) { return &__lconv; }
#endif /* !LUA_LIBC_SELFTEST */

/* ------------------------------------------------------------ self-test -- */
#ifdef LUA_LIBC_SELFTEST
static int fails;
static void ck(const char *got, const char *want, const char *tag) {
  if (!streq(got, want)) {
    printf("FAIL %-10s got=<%s> want=<%s>\n", tag, got, want);
    fails++;
  }
}
int main(void) {
  char b[128];
  lua_snprintf(b, sizeof b, "%lld", 400LL);
  ck(b, "400", "int-lld");
  lua_snprintf(b, sizeof b, "%d/%i/%u", -5, 7, 9u);
  ck(b, "-5/7/9", "int-mix");
  lua_snprintf(b, sizeof b, "%x/%X/%o", 255u, 255u, 8u);
  ck(b, "ff/FF/10", "int-base");
  lua_snprintf(b, sizeof b, "%5d|%-5d|%05d", 42, 42, 42);
  ck(b, "   42|42   |00042", "int-width");
  lua_snprintf(b, sizeof b, "[%s][%.3s][%8s]", "hi", "abcdef", "x");
  ck(b, "[hi][abc][       x]", "str");
  lua_snprintf(b, sizeof b, "%c%c", 'O', 'K');
  ck(b, "OK", "char");
  lua_snprintf(b, sizeof b, "%.14g", 400.0);
  ck(b, "400", "g-int");
  lua_snprintf(b, sizeof b, "%.14g", 3.14159);
  ck(b, "3.14159", "g-frac");
  lua_snprintf(b, sizeof b, "%.14g", 0.1);
  ck(b, "0.1", "g-tenth");
  lua_snprintf(b, sizeof b, "%.14g", 1e20);
  ck(b, "1e+20", "g-big");
  lua_snprintf(b, sizeof b, "%.2f", 3.14159);
  ck(b, "3.14", "f");
  lua_snprintf(b, sizeof b, "%.3e", 12345.678);
  ck(b, "1.235e+04", "e");
  lua_snprintf(b, sizeof b, "%g", -0.5);
  ck(b, "-0.5", "g-neg");
  /* ctype / string spot checks */
  if (!isdigit('7') || isdigit('a') || !isspace(' ') || !isxdigit('F') ||
      toupper('a') != 'A' || tolower('Z') != 'z' || !ispunct('!'))
    fails++, printf("FAIL ctype\n");
  if (strncmp("abc", "abd", 2) != 0 || strncmp("abc", "abd", 3) >= 0)
    fails++, printf("FAIL strncmp\n");
  if (!strchr("hello", 'l') || strchr("hello", 'z'))
    fails++, printf("FAIL strchr\n");
  if (strspn("aabbc", "ab") != 4)
    fails++, printf("FAIL strspn\n");
  if (!strstr("hello world", "wor") || strstr("abc", "xyz"))
    fails++, printf("FAIL strstr\n");
  if (!strpbrk("hello", "xl") || strpbrk("hello", "xyz"))
    fails++, printf("FAIL strpbrk\n");
  printf("lua_libc self-test: %s (%d failures)\n", fails ? "FAIL" : "ok", fails);
  return fails ? 1 : 0;
}
#endif
