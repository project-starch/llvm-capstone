/* Freestanding libc for the PHP engine domain — only what the linker actually asked for.
 *
 * memcpy/memmove/memset/strlen/strcmp/strcpy are NOT here: they come from
 * capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c, which has the
 * tag-preserving ldc/stc fast path. A byte-loop memcpy strips the tag off any capability
 * it copies, and PHP copies zvals -- whose union holds pointers -- constantly.
 */
typedef unsigned long size_t;

/* --- string --- */
char *strncpy(char *d, const char *s, size_t n)
{
    size_t i = 0;
    for (; i < n && s[i]; i++) { d[i] = s[i]; }
    for (; i < n; i++) { d[i] = 0; }
    return d;
}
char *strcat(char *d, const char *s)
{
    char *p = d; while (*p) p++;
    while ((*p = *s)) { p++; s++; }
    return d;
}
char *strncat(char *d, const char *s, size_t n)
{
    char *p = d; while (*p) p++;
    while (n-- && *s) { *p++ = *s++; }
    *p = 0;
    return d;
}
int strncmp(const char *a, const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        unsigned char x = (unsigned char)a[i], y = (unsigned char)b[i];
        if (x != y) { return (int)x - (int)y; }
        if (!x) { return 0; }
    }
    return 0;
}
char *strchr(const char *s, int c)
{
    for (;; s++) {
        if (*s == (char)c) { return (char *)s; }
        if (!*s) { return (char *)0; }
    }
}
char *strrchr(const char *s, int c)
{
    const char *last = (const char *)0;
    for (;; s++) { if (*s == (char)c) last = s; if (!*s) break; }
    return (char *)last;
}
char *strstr(const char *h, const char *n)
{
    if (!*n) { return (char *)h; }
    for (; *h; h++) {
        const char *a = h, *b = n;
        while (*a && *b && *a == *b) { a++; b++; }
        if (!*b) { return (char *)h; }
    }
    return (char *)0;
}
void *memchr(const void *p, int c, size_t n)
{
    const unsigned char *s = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { if (s[i] == (unsigned char)c) { return (void *)(s + i); } }
    return (void *)0;
}
/* memcmp is NOT defined here: beebs_freestanding_string.c provides it, along with
 * memcpy/memmove/memset/strlen/strcmp/strcpy. Those are the tag-preserving versions and
 * must be the only ones in the link -- defining a second memcpy here would silently give
 * some call sites a byte loop that strips capability tags out of copied zvals. */

/* ASCII only: glibc's tolower goes through __ctype_b_loc, a locale table a domain
 * cannot populate. */
static int lc(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

int strcasecmp(const char *a, const char *b)
{
    for (;; a++, b++) {
        int x = lc((unsigned char)*a), y = lc((unsigned char)*b);
        if (x != y) { return x - y; }
        if (!x) { return 0; }
    }
}
int strncasecmp(const char *a, const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        int x = lc((unsigned char)a[i]), y = lc((unsigned char)b[i]);
        if (x != y) { return x - y; }
        if (!x) { return 0; }
    }
    return 0;
}

void *malloc(size_t);
char *strdup(const char *s)
{
    size_t n = 0; while (s[n]) n++;
    char *p = (char *)malloc(n + 1);
    if (p) { for (size_t i = 0; i <= n; i++) { p[i] = s[i]; } }
    return p;
}

/* --- conversion --- */
int abs(int v) { return v < 0 ? -v : v; }

static int digit_of(int c, int base)
{
    int d;
    if (c >= '0' && c <= '9')      { d = c - '0'; }
    else if (c >= 'a' && c <= 'z') { d = c - 'a' + 10; }
    else if (c >= 'A' && c <= 'Z') { d = c - 'A' + 10; }
    else                           { return -1; }
    return d < base ? d : -1;
}

unsigned long strtoul(const char *s, char **end, int base)
{
    const char *p = s;
    while (*p == ' ' || (*p >= 9 && *p <= 13)) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { p += 2; base = 16; }
    else if (base == 0) { base = (p[0] == '0') ? 8 : 10; }
    unsigned long v = 0; int any = 0, d;
    while ((d = digit_of((unsigned char)*p, base)) >= 0) { v = v * (unsigned long)base + (unsigned long)d; p++; any = 1; }
    if (end) { *end = (char *)(any ? p : s); }
    return neg ? (unsigned long)(-(long)v) : v;
}
long strtol(const char *s, char **end, int base)
{
    return (long)strtoul(s, end, base);
}
int atoi(const char *s)  { return (int)strtol(s, (char **)0, 10); }
long atol(const char *s) { return strtol(s, (char **)0, 10); }

/* strtod: PHP calls it from zend_operators (numeric string conversion) and zend_ini.
 * Handles the decimal forms a script actually produces; no hex floats, no inf/nan
 * spellings. Exponent by repeated multiply, so no libm dependency. */
double strtod(const char *s, char **end)
{
    const char *p = s;
    while (*p == ' ' || (*p >= 9 && *p <= 13)) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
    double v = 0.0; int any = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10.0 + (double)(*p - '0'); p++; any = 1; }
    if (*p == '.') {
        p++;
        double scale = 0.1;
        while (*p >= '0' && *p <= '9') { v += (double)(*p - '0') * scale; scale *= 0.1; p++; any = 1; }
    }
    if (any && (*p == 'e' || *p == 'E')) {
        const char *save = p;
        p++;
        int eneg = 0;
        if (*p == '+' || *p == '-') { eneg = (*p == '-'); p++; }
        if (*p >= '0' && *p <= '9') {
            int e = 0;
            while (*p >= '0' && *p <= '9') { e = e * 10 + (*p - '0'); p++; }
            for (int i = 0; i < e; i++) { v = eneg ? v / 10.0 : v * 10.0; }
        } else { p = save; }
    }
    if (end) { *end = (char *)(any ? p : s); }
    return neg ? -v : v;
}

double atof(const char *s) { return strtod(s, (char **)0); }

/* --- math --- */
int finite(double d) { return !(d != d) && d != (double)1e308 * 10.0 && d != -((double)1e308 * 10.0); }
int isnan(double d)  { return d != d; }
double fabs(double d){ return d < 0.0 ? -d : d; }

/* pow: only integral exponents are reachable from the engine's numeric paths
 * (zend_operators' ZEND_POW does not exist in 5.0.0; this is the ini/parser path). */
double pow(double b, double e)
{
    long n = (long)e;
    double r = 1.0;
    int neg = n < 0;
    if (neg) { n = -n; }
    for (long i = 0; i < n; i++) { r *= b; }
    return neg ? 1.0 / r : r;
}
double floor(double d) { double t = (double)(long)d; return (t > d) ? t - 1.0 : t; }
double ceil(double d)  { double t = (double)(long)d; return (t < d) ? t + 1.0 : t; }
double fmod(double a, double b) { if (b == 0.0) return 0.0; return a - b * (double)(long)(a / b); }

/* --- alloca, off-stack: see stubinc/alloca.h for why --- */
#ifndef PHP_CAPSTONE_ALLOCA_BYTES
#define PHP_CAPSTONE_ALLOCA_BYTES (256u * 1024u)
#endif
static unsigned char alloca_pool[PHP_CAPSTONE_ALLOCA_BYTES] __attribute__((aligned(16)));
static unsigned long alloca_off;
unsigned long php_capstone_alloca_total;
unsigned long php_capstone_alloca_peak;

/* The depth watchdog is ALSO sited here, not just in malloc. malloc turned out never to be
 * called during the 2.6 MB stack excursion, but do_alloca IS on the startup path
 * (zend_API.c:1300 inside zend_register_functions' loop, zend_builtin_functions.c:817,
 * zend_execute_API.c:888), so this catches excursions that allocate no heap. */
extern unsigned long php_depth_floor;
extern unsigned long php_depth_low;
extern int           php_depth_armed;
extern void        (*php_depth_trip_fn)(void);
void php_depth_walk(void);

void *php_capstone_alloca(unsigned long n)
{
    unsigned long c = __builtin_capstone_cap_get_cursor(__builtin_frame_address(0));
    if (c < php_depth_low) { php_depth_low = c; }
    if (php_depth_armed && c < php_depth_floor && php_depth_trip_fn) {
        php_depth_armed = 0;
        php_depth_walk();
        php_depth_trip_fn();
    }
    n = (n + 15UL) & ~15UL;
    php_capstone_alloca_total += n;
    if (alloca_off + n > (unsigned long)PHP_CAPSTONE_ALLOCA_BYTES) {
        return (void *)0;          /* caller sees NULL rather than silent corruption */
    }
    unsigned char *p = &alloca_pool[alloca_off];
    alloca_off += n;
    if (alloca_off > php_capstone_alloca_peak) { php_capstone_alloca_peak = alloca_off; }
    return p;
}

#include <locale.h>
#include <langinfo.h>

/* The "C" locale, and the only one a domain has. ext/standard/string.c's php_localeconv_r copies
 * this struct wholesale on the number_format path; no trigger reaches it, but it must link. The
 * CHAR_MAX entries mean "unspecified", exactly as the C standard says for the C locale. */
struct lconv *localeconv(void)
{
    static char empty[] = "";
    static char point[] = ".";
    static struct lconv c_locale;
    c_locale.decimal_point     = point;
    c_locale.thousands_sep     = empty;
    c_locale.grouping          = empty;
    c_locale.int_curr_symbol   = empty;
    c_locale.currency_symbol   = empty;
    c_locale.mon_decimal_point = empty;
    c_locale.mon_thousands_sep = empty;
    c_locale.mon_grouping      = empty;
    c_locale.positive_sign     = empty;
    c_locale.negative_sign     = empty;
    c_locale.int_frac_digits   = (char)127;
    c_locale.frac_digits       = (char)127;
    c_locale.p_cs_precedes     = (char)127;
    c_locale.p_sep_by_space    = (char)127;
    c_locale.n_cs_precedes     = (char)127;
    c_locale.n_sep_by_space    = (char)127;
    c_locale.p_sign_posn       = (char)127;
    c_locale.n_sign_posn       = (char)127;
    return &c_locale;
}

/* A domain has the "C" locale and no means to acquire another. Querying (locale == NULL) reports
 * "C"; asking for "C" or "" succeeds trivially; anything else fails, which is what a C library
 * without locale data is required to do. PHP_FUNCTION(setlocale) then returns false, correctly. */
char *setlocale(int category, const char *locale)
{
    static char c_name[] = "C";
    (void)category;
    if (!locale) { return c_name; }
    if ((locale[0] == 'C' && locale[1] == 0) || locale[0] == 0) { return c_name; }
    return (char *)0;
}

char *nl_langinfo(nl_item item)
{
    static char empty[] = "";
    (void)item;
    return empty;
}

/* --- string/locale helpers ext/standard's pure TUs import from libc ---
 * None is on a trigger path; each must link, and each does the C-locale-correct thing rather than
 * returning a placeholder, so that a future trigger reaching one behaves predictably. */

/* In the C locale, collation order IS byte order. */
int strcoll(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

char *strpbrk(const char *s, const char *accept)
{
    for (; *s; s++) {
        for (const char *a = accept; *a; a++) { if (*s == *a) { return (char *)s; } }
    }
    return (char *)0;
}

/* Single-byte locale: every non-NUL byte is a complete character, and mblen(NULL, n) reports
 * that the encoding is stateless by returning 0, as C requires. */
int mblen(const char *s, unsigned long n)
{
    if (!s) { return 0; }
    if (n == 0) { return -1; }
    return *s ? 1 : 0;
}

/* Monetary formatting has no C-locale definition and PHP only reaches it from money_format().
 * -1 with no output is the documented failure, which is the honest answer here. */
long strfmon(char *s, unsigned long max, const char *format, ...)
{
    (void)s; (void)max; (void)format;
    return -1;
}

/* --- the three libc entry points formatted_print.c's float path needs --- */

int isascii(int c) { return (c & ~0x7f) == 0; }

/* IEEE-754 binary64: exponent all ones with a zero mantissa is an infinity. Done by bit pattern
 * rather than by comparison, because a comparison against HUGE_VAL needs a working HUGE_VAL. */
int isinf(double d)
{
    union { double d; unsigned long u; } v;
    v.d = d;
    return ((v.u >> 52) & 0x7FFUL) == 0x7FFUL && (v.u & 0xFFFFFFFFFFFFFUL) == 0UL;
}

/* Split into integral and fractional parts, both carrying the sign of the argument, as C requires. */
double modf(double value, double *iptr)
{
    double ip = (double)(long)value;        /* truncation toward zero */
    if (iptr) { *iptr = ip; }
    return value - ip;
}
