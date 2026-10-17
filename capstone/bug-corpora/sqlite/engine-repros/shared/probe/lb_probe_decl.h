/* ---------------- LadyBug reachability probe: declarations ----------------
 *
 * WHY THIS EXISTS. A CheriBSD run that ends `RETURNED rc=0` tells us the SQL
 * statement executed. It does NOT tell us the defective code path inside that
 * statement was traversed, and without that the silence means nothing: "the
 * mechanism did not catch it" and "the input never got there" produce the same
 * log. Every site below is the exact sqlite3.c line where the host ASan oracle,
 * running THIS case source with THESE flags, reported the memory error.
 *
 * Cost when not armed: one load of lb_site and a compare. The arming check is a
 * macro, not a call, because two of the sites (sqlite3Fts5GetVarint,
 * sqlite3Fts5GetVarint32) run millions of times per case.
 *
 * LB_HIT is a statement; LB_CHK is an expression that always evaluates to 1, for
 * the sites that sit in a loop condition or an `else if` where no statement can
 * be inserted without changing control flow.
 */
#ifndef LB_PROBE_DECL_H
#define LB_PROBE_DECL_H
extern int lb_site;                                     /* armed site id, 0 = none */
extern int lb_hit_slow(int id, const void *p, long n);  /* always returns 1 */
extern void lb_report(void);                            /* hit count; safe in a fault handler */
/* Output helpers, for a site that reports more than LB_HIT's verdict -- a site in an
 * extension's own translation unit cannot reach the static ones. */
extern void lb_puts(const char *s);
extern void lb_putu(unsigned long v);
extern void lb_puti(long v);
extern void lb_putx(unsigned long long v);
#define LB_HIT(id, p, n)  do{ if(lb_site==(id)) (void)lb_hit_slow((id),(const void*)(p),(long)(n)); }while(0)
#define LB_CHK(id, p, n)  (lb_site==(id) ? lb_hit_slow((id),(const void*)(p),(long)(n)) : 1)
#endif
