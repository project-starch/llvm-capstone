/*
 * Capstone-adapted replacement for fast.rev.d12.c (stringsearch1 benchmark).
 *
 * Provides prep2 and exec2.  The upstream source has three pointer-difference
 * patterns that trigger the non-vector SHL i128 backend crash (Bug #3):
 *
 *   prep2 line 82: d[*p] = pe-p;
 *     ptrdiff_t (i128) stored into Tab (long, i64) → store i128 not split.
 *     Fix: integer counter (same as fwd file).
 *
 *   exec2 line 166: k2 = d2[p-pat.pat];
 *     ptrdiff_t (i128) used as array index → SHL i128 in GEP.
 *     Fix: integer index pidx, decremented in lock-step with p.
 *
 *   exec2 line 170: k2 = q+k2-RH;
 *     pointer difference (i128) assigned to int k2 → missing trunc i128→i32.
 *     Fix: integer counter scan_count = -(q-RH), so k2 = k2-scan_count.
 *
 * memcpy is defined in beebs_stringsearch1_fwd_capstone.c (one definition per
 * link unit).
 */

extern void *memcpy(void *, const void *, unsigned long);

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
    Tab delta1[256];
    Tab delta2[257];
} pat;

void
prep2(CHARTYPE *base, int m)
{
    CHARTYPE *pe, *p;
    int j, shift;
    Tab *d, *d2;
    int q1, t, qp, jp, kp;
    Tab f[256], f1[256];

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
    d2 = pat.delta1;
    for (j = 0; j < 256; j++)
        d2[j] = m;
    for (j = 0; j < m; j++)
        d2[base[j]] = m - 1 - j;
    d2 = pat.delta2;
    for (j = 1; j < m; j++)
        d2[j] = 2 * m - j;
    for (j = m, t = m + 1; j > 0; j--, t--) {
        f[j] = t;
        while ((t <= m) && (base[t - 1] != base[j - 1])) {
            if ((m - j) < d2[t])
                d2[t] = m - j;
            t = f[t];
        }
    }
    q1 = t;
    t = m + 1 - q1;
    qp = 1;
    for (jp = 1, kp = 0; kp < t; jp++, kp++) {
        f1[jp] = kp;
        while ((kp >= 1) && (base[jp - 1] != base[kp - 1]))
            kp = f1[kp];
    }
    while (q1 < m) {
        for (j = qp; j <= q1; j++)
            if (m + q1 - j < d2[j])
                d2[j] = m + q1 - j;
        qp = q1 + 1;
        q1 += t - f1[t];
        t = f1[t];
    }
    d2[0] = m + 1;
}

int
exec2(CHARTYPE *base, int n)
{
    int nmatch = 0;
    CHARTYPE *e, *s;
    Tab *d0 = pat.delta;
    Tab lastdelta;
    CHARTYPE *p, *q;
    CHARTYPE *prev = pat.pat + pat.patlen - 1;
    Tab *d1 = pat.delta1;
    Tab *d2 = pat.delta2 + 1;
    int k1, k2;
    /* pidx    = p - pat.pat: decremented in lock-step with p.
     * scan_count = number of loop iterations = -(q - s) at mismatch point. */
    int pidx, scan_count;
    /* Ptr-subtraction workaround: see exec1 comment in fwd file. */
    Tab lastdelta_neg;

    lastdelta = (Tab)n + (Tab)pat.patlen;
    d0[pat.lastchar] = lastdelta;
    lastdelta_neg = -lastdelta;
    s = base + (Tab)(pat.patlen - 1);
    e = base + n;
    while (s < e) {
        while ((s += d0[*s]) < e)
            ;
        if (s < e + (Tab)pat.patlen)
            break;
        s += lastdelta_neg;    /* s -= lastdelta, via loaded Tab to avoid sub i128 */

        pidx = pat.patlen - 1;
        scan_count = 0;
        for (p = prev, q = s; p > pat.pat; ) {
            --q; --p; --pidx; ++scan_count;
            if (*q != *p)
                goto mismatch;
        }
        nmatch++;
    mismatch:
        /* Original: k2 = d2[p-pat.pat]  — ptrdiff_t used as index (Bug #3).
         * Fix: d2[pidx] where pidx tracks p-pat.pat as an int. */
        k2 = (int)d2[pidx];
        k1 = (int)d1[*q];
        if (k2 < k1) k2 = k1;
        /* Original: k2 = q+k2-RH  — ptrdiff_t assigned to int (Bug #3).
         * q-s = -scan_count, so q+k2-s = k2-scan_count. */
        k2 = k2 - scan_count;
        s += k2;
    }
    return nmatch;
}
