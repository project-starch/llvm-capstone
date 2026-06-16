/*
 * Capstone-adapted replacement for fast.fwd.inc.c (stringsearch1 benchmark).
 *
 * Provides prep1 and exec1.  The upstream source has:
 *   d[*p] = pe - p;   (prep1, line 78)
 * On Capstone, pointer differences are ptrdiff_t = i128.  Storing an i128 into
 * a Tab (= long = i64) variable causes the compiler to emit `store i128` which
 * the backend cannot split (same non-vector SHL i128 Bug #3 root cause as
 * matmult).  Fix: replace pointer difference with an integer shift counter.
 *
 * Also provides: memcpy stub (called by both prep1 and prep2), strlen stub
 * (called by stringsearch1.c benchmark()).
 */

#ifndef CHARTYPE
#define CHARTYPE unsigned char
#endif
#define MAXPAT 256

#ifndef TABTYPE
#define TABTYPE long
#endif
typedef TABTYPE Tab;

static struct {
    int patlen;
    CHARTYPE pat[MAXPAT];
    Tab delta[256];
    int lastchar;
} pat;

/* memcpy stub — shared across all three stringsearch1 object files */
void *memcpy(void *dest, const void *src, unsigned long n)
{
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    unsigned long i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
    return dest;
}

/* strlen stub — called by stringsearch1.c */
unsigned long strlen(const char *s)
{
    unsigned long n = 0;
    while (*s++)
        n++;
    return n;
}

void
prep1(CHARTYPE *base, int m)
{
    CHARTYPE *pe, *p;
    int j, shift;
    Tab *d;

    pat.patlen = m;
    if (m > MAXPAT)
        return;   /* never reached: m = size = 3 */
    memcpy(pat.pat, base, (unsigned long)m);
    d = pat.delta;
    for (j = 0; j < 256; j++)
        d[j] = pat.patlen;
    /* Original: d[*p] = pe-p  (ptrdiff_t = i128, causes backend crash).
     * Fix: integer counter shift = m-1 .. 1, matching the pointer difference. */
    for (p = pat.pat, pe = p + m - 1, shift = m - 1; p < pe; p++, shift--)
        d[*p] = (Tab)shift;
    pat.lastchar = *p;
}

int
exec1(CHARTYPE *base, int n)
{
    int nmatch = 0;
    CHARTYPE *e, *s;
    Tab *d0 = pat.delta;
    Tab lastdelta;
    CHARTYPE *p, *q;
    CHARTYPE *ep;
    Tab n1;
    /* Ptr-subtraction workaround (Bug #3 / sub i128 not selectable):
     * The DAGCombiner folds add(ptr, neg(x)) → sub(ptr, x), and the backend
     * has no selector for sub i128.  Fix: store the negative offset in a Tab
     * (long) local so it is loaded as a plain i64 at the use site — the
     * DAGCombiner cannot trace through a stack load and leaves the add alone.
     * This works at -O0 where all locals are spilled.  See Bug #3 notes. */
    Tab lastdelta_neg, n1_neg;

    lastdelta = (Tab)n + (Tab)pat.patlen;
    d0[pat.lastchar] = lastdelta;
    lastdelta_neg = -lastdelta;
    s = base + (Tab)(pat.patlen - 1);
    e = base + n;
    ep = pat.pat + (Tab)(pat.patlen - 1);
    n1 = (Tab)(pat.patlen - 1);
    n1_neg = -n1;
    while (s < e) {
        while ((s += d0[*s]) < e)
            ;
        if (s < e + (Tab)pat.patlen)
            break;
        s += lastdelta_neg;    /* s -= lastdelta, via loaded Tab to avoid sub i128 */
        for (p = pat.pat, q = s + n1_neg; p < ep; ) {   /* q = s - n1 */
            if (*q++ != *p++)
                goto mismatch;
        }
        nmatch++;
    mismatch:
        s++;
    }
    return nmatch;
}
