/* The printf helpers, in a domain.
 *
 * A domain has no stdout. It writes into a payload region the host reads and
 * prints after the domain returns, which is how every other domain in this
 * repository speaks. pg_domain_payload names that region once.
 *
 * The formatter understands what the memory manager and the replay actually
 * emit and nothing else: %s, %c, %d, %u, %ld, %lu, %zu, %p and %%, with an
 * optional width that is ignored. PostgreSQL's own vsnprintf is a thousand
 * lines and none of what it does beyond this is reached from here. A
 * conversion this does not know is copied out verbatim, so a message that
 * needs more says so in the output rather than silently losing a value.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

typedef struct _A11_FILE FILE;
static char out_stdout, out_stderr;
FILE *stdout = (FILE *) &out_stdout;
FILE *stderr = (FILE *) &out_stderr;

static char *pay_base;
static unsigned long *pay_len;
static unsigned long pay_cap;

void
pg_domain_payload(char *base, unsigned long *length, unsigned long capacity)
{
    pay_base = base;
    pay_len = length;
    pay_cap = capacity;
}

static void
put(char c)
{
    if (!pay_base || !pay_len || *pay_len + 1 >= pay_cap)
        return;
    pay_base[(*pay_len)++] = c;
}

void
pg_domain_text(const char *s)
{
    while (s && *s)
        put(*s++);
}

void
pg_domain_uint(unsigned long v)
{
    char digits[24];
    unsigned n = 0;

    do {
        digits[n++] = (char) ('0' + v % 10UL);
        v /= 10UL;
    } while (v);
    while (n)
        put(digits[--n]);
}

/* The one formatter. Every entry point below is a wrapper on it: into the
   payload when `buf` is null, into `buf` otherwise. */
static char *fmt_buf;
static size_t fmt_room, fmt_used;

static void
sink(char c)
{
    if (fmt_buf) {
        if (fmt_used + 1 < fmt_room)
            fmt_buf[fmt_used] = c;
        fmt_used++;
    } else {
        put(c);
    }
}

static void
sink_str(const char *s)
{
    while (s && *s)
        sink(*s++);
}

static void
sink_uint(unsigned long v)
{
    char digits[24];
    unsigned n = 0;

    do {
        digits[n++] = (char) ('0' + v % 10UL);
        v /= 10UL;
    } while (v);
    while (n)
        sink(digits[--n]);
}

static int
render(const char *fmt, va_list ap)
{
    fmt_used = 0;
    for (const char *p = fmt; p && *p; p++) {
        if (*p != '%') {
            sink(*p);
            continue;
        }
        const char *start = p++;
        while (*p == '-' || *p == '+' || *p == ' ' || *p == '0' || *p == '#'
               || (*p >= '1' && *p <= '9') || *p == '.')
            p++;
        int lng = 0;
        while (*p == 'l' || *p == 'z' || *p == 'h') {
            if (*p == 'l' || *p == 'z')
                lng = 1;
            p++;
        }
        switch (*p) {
            case 's': sink_str(va_arg(ap, const char *)); break;
            case 'c': sink((char) va_arg(ap, int)); break;
            case 'd': case 'i': {
                long v = lng ? va_arg(ap, long) : (long) va_arg(ap, int);
                if (v < 0) { sink('-'); sink_uint((unsigned long) -(v + 1) + 1UL); }
                else sink_uint((unsigned long) v);
                break;
            }
            case 'u': sink_uint(lng ? va_arg(ap, unsigned long)
                                    : (unsigned long) va_arg(ap, unsigned int)); break;
            case 'p': {
                void *v = va_arg(ap, void *);
                sink('0'); sink('x');
                /* one way, to print it: no capability is made from this */
                unsigned long a = (unsigned long) (uintptr_t) v;
                char d[24]; unsigned n = 0;
                do { unsigned x = (unsigned) (a & 0xf);
                     d[n++] = (char) (x < 10 ? '0' + x : 'a' + x - 10); a >>= 4; } while (a);
                while (n) sink(d[--n]);
                break;
            }
            case '%': sink('%'); break;
            case '\0': sink('%'); p--; break;
            default:
                /* a conversion this does not know: copy it out, so a message
                   that needs more says so rather than losing a value */
                while (start <= p) sink(*start++);
                break;
        }
    }
    if (fmt_buf && fmt_room)
        fmt_buf[fmt_used < fmt_room ? fmt_used : fmt_room - 1] = '\0';
    return (int) fmt_used;
}

int
pg_vsnprintf(char *str, size_t count, const char *fmt, va_list ap)
{
    fmt_buf = str; fmt_room = count;
    int n = render(fmt, ap);
    fmt_buf = NULL; fmt_room = 0;
    return n;
}
int
pg_snprintf(char *str, size_t count, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt);
    int n = pg_vsnprintf(str, count, fmt, ap);
    va_end(ap);
    return n;
}
int
pg_vfprintf(FILE *stream, const char *fmt, va_list ap)
{
    (void) stream;
    fmt_buf = NULL; fmt_room = 0;
    return render(fmt, ap);
}
int
pg_fprintf(FILE *stream, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt);
    int n = pg_vfprintf(stream, fmt, ap);
    va_end(ap);
    return n;
}
int
pg_printf(const char *fmt,...)
{
    va_list ap; va_start(ap, fmt);
    int n = pg_vfprintf(stdout, fmt, ap);
    va_end(ap);
    return n;
}
int
pg_vprintf(const char *fmt, va_list ap)
{
    return pg_vfprintf(stdout, fmt, ap);
}

/* elog's fatal path ends the process; a domain returns to the monitor. */
void
abort(void)
{
    pg_domain_text("\npg-mmgr: abort\n");
    for (;;)
        ;
}
void
exit(int code)
{
    (void) code;
    abort();
}
int
fflush(FILE *stream)
{
    (void) stream;
    return 0;
}
