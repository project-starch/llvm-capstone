/* printf / sprintf / fprintf for the engine domain.
 *
 * NOT a full printf, and deliberately so. PHP carries its own formatter in
 * main/snprintf.c + main/spprintf.c (both in the REQUIRED set), which is what every
 * php_printf / zend_error / spprintf call actually goes through. What is left over is a
 * handful of direct libc printf calls on error paths; this covers the conversions those
 * use and COUNTS what it could not render rather than emitting silent garbage.
 *
 * Output goes to the domain's sink (php_capstone_sink), which the domain points at its
 * hostcall payload. Before a sink is installed, output is dropped and counted.
 */
typedef unsigned long size_t;
typedef __builtin_va_list va_list;
#define va_start __builtin_va_start
#define va_end   __builtin_va_end
#define va_arg   __builtin_va_arg

void (*php_capstone_sink)(const char *, unsigned long);
unsigned long php_capstone_dropped;      /* bytes that had no sink */
unsigned long php_capstone_unhandled;    /* conversions this formatter does not know */

static void emit(const char *s, unsigned long n)
{
    if (php_capstone_sink) { php_capstone_sink(s, n); }
    else { php_capstone_dropped += n; }
}

static unsigned fmt_u(char *out, unsigned long v, unsigned base, int upper)
{
    char tmp[24]; unsigned n = 0;
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    if (v == 0) { tmp[n++] = '0'; }
    while (v) { tmp[n++] = digits[v % base]; v /= base; }
    for (unsigned i = 0; i < n; i++) { out[i] = tmp[n - 1 - i]; }
    return n;
}

static int vformat(char *dst, size_t cap, const char *f, va_list ap)
{
    size_t n = 0;
    #define PUT(c) do { if (dst) { if (n + 1 < cap) dst[n] = (c); } n++; } while (0)
    for (; *f; f++) {
        if (*f != '%') { PUT(*f); continue; }
        f++;
        while (*f == '-' || *f == '+' || *f == ' ' || *f == '0' || *f == '#') f++;
        while (*f >= '0' && *f <= '9') f++;
        if (*f == '.') { f++; while (*f >= '0' && *f <= '9') f++; }
        int lng = 0;
        while (*f == 'l' || *f == 'z' || *f == 'h') { if (*f == 'l' || *f == 'z') lng = 1; f++; }
        char nb[24]; unsigned k;
        switch (*f) {
        case 'd': case 'i': {
            long v = lng ? va_arg(ap, long) : (long)va_arg(ap, int);
            if (v < 0) { PUT('-'); v = -v; }
            k = fmt_u(nb, (unsigned long)v, 10, 0);
            for (unsigned i = 0; i < k; i++) PUT(nb[i]);
            break; }
        case 'u': {
            unsigned long v = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned);
            k = fmt_u(nb, v, 10, 0); for (unsigned i = 0; i < k; i++) PUT(nb[i]); break; }
        case 'x': case 'X': {
            unsigned long v = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned);
            k = fmt_u(nb, v, 16, *f == 'X'); for (unsigned i = 0; i < k; i++) PUT(nb[i]); break; }
        case 'p': {
            void *v = va_arg(ap, void *);
            PUT('0'); PUT('x');
            k = fmt_u(nb, (unsigned long)__builtin_capstone_cap_get_cursor(v), 16, 0);
            for (unsigned i = 0; i < k; i++) PUT(nb[i]); break; }
        case 's': {
            const char *v = va_arg(ap, const char *);
            if (!v) v = "(null)";
            while (*v) { PUT(*v); v++; }
            break; }
        case 'c': { int v = va_arg(ap, int); PUT((char)v); break; }
        case '%': PUT('%'); break;
        default:
            /* Counted, never silently dropped: an unrendered conversion in an error
             * message is exactly the case where a wrong diagnosis gets expensive. */
            php_capstone_unhandled++;
            PUT('%'); PUT(*f ? *f : '?');
            break;
        }
        if (!*f) break;
    }
    if (dst && cap) { dst[n < cap ? n : cap - 1] = 0; }
    #undef PUT
    return (int)n;
}

int vsnprintf(char *d, size_t c, const char *f, va_list ap) { return vformat(d, c, f, ap); }
int vsprintf(char *d, const char *f, va_list ap)            { return vformat(d, (size_t)-1, f, ap); }

int snprintf(char *d, size_t c, const char *f, ...)
{ va_list ap; va_start(ap, f); int r = vformat(d, c, f, ap); va_end(ap); return r; }

int sprintf(char *d, const char *f, ...)
{ va_list ap; va_start(ap, f); int r = vformat(d, (size_t)-1, f, ap); va_end(ap); return r; }

static char pbuf[1024];
int printf(const char *f, ...)
{ va_list ap; va_start(ap, f); int r = vformat(pbuf, sizeof pbuf, f, ap); va_end(ap);
  emit(pbuf, (unsigned long)(r < (int)sizeof pbuf ? r : (int)sizeof pbuf - 1)); return r; }

typedef struct _IO_FILE FILE;
int fprintf(FILE *fp, const char *f, ...)
{ (void)fp; va_list ap; va_start(ap, f); int r = vformat(pbuf, sizeof pbuf, f, ap); va_end(ap);
  emit(pbuf, (unsigned long)(r < (int)sizeof pbuf ? r : (int)sizeof pbuf - 1)); return r; }

int vfprintf(FILE *fp, const char *f, va_list ap)
{ (void)fp; int r = vformat(pbuf, sizeof pbuf, f, ap);
  emit(pbuf, (unsigned long)(r < (int)sizeof pbuf ? r : (int)sizeof pbuf - 1)); return r; }

int fputs(const char *s, FILE *fp) { (void)fp; unsigned long n = 0; while (s[n]) n++; emit(s, n); return 0; }
int fputc(int c, FILE *fp)         { (void)fp; char b = (char)c; emit(&b, 1); return c; }
int puts(const char *s)            { unsigned long n = 0; while (s[n]) n++; emit(s, n); emit("\n", 1); return 0; }
