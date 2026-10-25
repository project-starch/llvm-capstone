/* Parent, sibling and successor controls at Perl 5.36.3's SV head arena
 * (pre-registration: capstone/docs/plans/2026-10-11-nested-lifetime-controls.md).
 *
 * libperl embedded in a C harness, so the nested allocator is driven through its own API:
 * a head is taken by newSV/newSViv (new_SV -> uproot_SV) and released by SvREFCNT_dec to 0
 * (sv_free2 -> del_SV -> plant_SV). Under PERL_CAPSTONE_SUBLET (ports/perl/sublet patch 0001)
 * uproot_SV hands the head out as a CDERIVE child of its arena and plant_SV revokes it.
 *
 *   svhead-controls parent      release A, then read and write A's former head through the
 *                               arena's own pointer to it (Perl_sv_arena_head)
 *   svhead-controls sibling     A and B from one arena, release A, read and write B through B
 *   svhead-controls successor   release A, newSV until a head at A's address comes back (C),
 *                               read and write through C
 *   svhead-controls stale       as successor, then read through the old pointer to A
 *
 * Every memory access the control is about goes through one of the noinline probe_* functions
 * below, so a fault's pc names it, and each is announced first on stdout (unbuffered) as
 *   PROBE <label> <file>:<line of the call>
 * Addresses are the pointers' addresses (ptraddr), taken while the pointer is live.
 * Each mode ends with "RESULT <mode> PASS|FAIL <why>" and exits 0 on PASS, 1 on FAIL. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "EXTERN.h"
#include "perl.h"

static PerlInterpreter *my_perl;

#define MARK_A    ((IV)0x0A0A0A01)
#define MARK_B    ((IV)0x0B0B0B02)
#define MARK_B2   ((IV)0x0B0B0B03)
#define MARK_C    ((IV)0x0C0C0C04)
#define MARK_C2   ((IV)0x0C0C0C05)
#define MARK_RC   ((U32)0x5A5A5A5A)

/* ---- the probes: one access each, never inlined ---------------------------------------- */
static __attribute__((noinline)) U32 probe_read_flags(const SV *p)
{ return *(const volatile U32 *)&p->sv_flags; }
static __attribute__((noinline)) U32 probe_read_refcnt(const SV *p)
{ return *(const volatile U32 *)&p->sv_refcnt; }
static __attribute__((noinline)) void probe_write_refcnt(SV *p, U32 v)
{ *(volatile U32 *)&p->sv_refcnt = v; }
static __attribute__((noinline)) UV probe_read_any(const SV *p)
{ void *const volatile *slot = (void *const volatile *)&p->sv_any; return PTR2UV(*slot); }
static __attribute__((noinline)) IV probe_read_iv(const SV *p)
{ return *(const volatile IV *)&p->sv_u.svu_iv; }
static __attribute__((noinline)) void probe_write_iv(SV *p, IV v)
{ *(volatile IV *)&p->sv_u.svu_iv = v; }

#define ANNOUNCE(label) \
    (printf("PROBE %s %s:%d\n", (label), "svhead-controls.c", __LINE__))

static unsigned long cap_base(const void *p) { return __builtin_capstone_cap_get_base((void *)p); }
static unsigned long cap_end(const void *p) { return __builtin_capstone_cap_get_end((void *)p); }

/* ---- the arena's own pointer -------------------------------------------------------------
 * Under the patch, Perl_sv_arena_head (sv.c) is the patch's own function: the arena's pointer
 * to the head at an address, which plant_SV and SvIS_FREED use. The release build has no such
 * function, so the same walk is written out here: PL_sv_arenaroot's arenas, the first slot of
 * each not an SV, PERL_ARENA_SIZE / sizeof(SV) - 1 heads after it (sv_add_arena, Perl_more_sv).
 * Only the address of `sv` is used; it is never read. */
#define SV_ARENA_HEADS (PERL_ARENA_SIZE / sizeof(SV) - 1)

static SV *arena_of(UV addr)
{
    SV *sva;
    for (sva = PL_sv_arenaroot; sva; sva = MUTABLE_SV(SvANY(sva)))
        if (addr - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
            return sva;
    return NULL;
}

static SV *walk_arena_head(UV addr)
{
    SV *const sva = arena_of(addr);
    return sva ? sva + 1 + (addr - PTR2UV(sva + 1)) / sizeof(SV) : NULL;
}

static SV *arena_head(const SV *sv, UV addr)
{
#ifdef PERL_CAPSTONE_SUBLET
    SV *const own = Perl_sv_arena_head(sv);
    SV *const walked = walk_arena_head(addr);
    printf("ARENA_HEAD via=Perl_sv_arena_head addr=%#lx walk=%#lx same=%s\n",
           (unsigned long)PTR2UV(own), (unsigned long)PTR2UV(walked),
           PTR2UV(own) == PTR2UV(walked) ? "yes" : "NO");
    return own;
#else
    PERL_UNUSED_ARG(sv);
    SV *const walked = walk_arena_head(addr);
    printf("ARENA_HEAD via=walk addr=%#lx\n", (unsigned long)PTR2UV(walked));
    return walked;
#endif
}

static void show(const char *name, const void *p)
{
    printf("PTR %s addr=%#lx base=%#lx end=%#lx\n", name, (unsigned long)PTR2UV(p),
           cap_base(p), cap_end(p));
}

static int result(const char *mode, int ok, const char *why)
{
    printf("RESULT %s %s %s\n", mode, ok ? "PASS" : "FAIL", why);
    return ok ? 0 : 1;
}

/* ---- 1. parent ---------------------------------------------------------------------------- */
static int control_parent(void)
{
    SV *a = newSViv(MARK_A);
    const UV a_addr = PTR2UV(a);
    SV *sva = arena_of(a_addr);
    show("A", a);
    if (!sva)
        return result("parent", 0, "A is in no SV arena");
    show("arena", sva);
    const UV root_before = PTR2UV(PL_sv_root);
    printf("FREE A refcnt=%u (SvREFCNT_dec to 0: sv_free2 -> del_SV -> plant_SV)\n",
           (unsigned)SvREFCNT(a));
    SvREFCNT_dec(a);            /* `a` is passed below by address only, as SvIS_FREED does */
    printf("PL_sv_root addr=%#lx (A=%#lx) link_expected=%#lx\n", (unsigned long)PTR2UV(PL_sv_root),
           (unsigned long)a_addr, (unsigned long)root_before);

    SV *p = arena_head(a, a_addr);
    if (!p || PTR2UV(p) != a_addr)
        return result("parent", 0, "the arena pointer is not at A's address");
    show("parent", p);

    ANNOUNCE("parent-read-flags");   U32 flags = probe_read_flags(p);
    ANNOUNCE("parent-read-link");    UV link = probe_read_any(p);
    ANNOUNCE("parent-read-refcnt");  U32 rc = probe_read_refcnt(p);
    ANNOUNCE("parent-write-refcnt"); probe_write_refcnt(p, MARK_RC);
    ANNOUNCE("parent-read-refcnt");  U32 rc2 = probe_read_refcnt(p);
    ANNOUNCE("parent-write-refcnt"); probe_write_refcnt(p, rc);   /* put it back */
    ANNOUNCE("parent-read-refcnt");  U32 rc3 = probe_read_refcnt(p);
    printf("READ flags=%#x (SVTYPEMASK=%#x) link=%#lx refcnt=%#x wrote=%#x read_back=%#x restored=%#x\n",
           (unsigned)flags, (unsigned)SVTYPEMASK, (unsigned long)link, (unsigned)rc,
           (unsigned)MARK_RC, (unsigned)rc2, (unsigned)rc3);

    const int freed = flags == SVTYPEMASK;
    const int on_list = PTR2UV(PL_sv_root) == a_addr && link == root_before;
    const int wrote = rc2 == MARK_RC && rc3 == rc;
    char why[160];
    snprintf(why, sizeof why, "freed_flags=%s free_list_link=%s write_read_back=%s",
             freed ? "yes" : "NO", on_list ? "yes" : "NO", wrote ? "yes" : "NO");
    return result("parent", freed && on_list && wrote, why);
}

/* ---- 2. sibling --------------------------------------------------------------------------- */
static int control_sibling(void)
{
    /* A and B are two consecutive heads the allocator hands out from one arena. The head after
       A on the free list can lie in another arena (first run, 2026-10-11: after interpreter
       start-up A's arena had no other free head, so no B taken after A was ever in it), so the
       pair is taken where two consecutive heads share an arena; the heads skipped stay allocated.
       Perl_more_sv carves a fresh arena whole, so this ends within one arena's worth. */
    SV *a = newSViv(MARK_A), *b = NULL;
    int tries;
    for (tries = 1; tries <= 4096; tries++) {
        b = newSViv(MARK_B);
        if (arena_of(PTR2UV(b)) && arena_of(PTR2UV(b)) == arena_of(PTR2UV(a)))
            break;
        a = b;          /* the previous one stays allocated */
        sv_setiv(a, MARK_A);
    }
    if (tries > 4096)
        return result("sibling", 0, "no two consecutive heads in one arena");
    const UV a_addr = PTR2UV(a), b_addr = PTR2UV(b);
    show("A", a);
    show("B", b);
    printf("SAME_ARENA arena=%#lx tries=%d B_minus_A=%ld\n", (unsigned long)PTR2UV(arena_of(a_addr)),
           tries, (long)(b_addr - a_addr));
    printf("FREE A refcnt=%u\n", (unsigned)SvREFCNT(a));
    SvREFCNT_dec(a);
    a = NULL;

    ANNOUNCE("sibling-read-flags");  U32 flags = probe_read_flags(b);
    ANNOUNCE("sibling-read-iv");     IV iv = probe_read_iv(b);
    ANNOUNCE("sibling-read-refcnt"); U32 rc = probe_read_refcnt(b);
    ANNOUNCE("sibling-write-iv");    probe_write_iv(b, MARK_B2);
    ANNOUNCE("sibling-read-iv");     IV iv2 = probe_read_iv(b);
    const IV api = SvIV(b);                                  /* and through Perl's own API */
    sv_setiv(b, MARK_B);
    const IV api2 = SvIV(b);
    printf("READ B flags=%#x iv=%#lx refcnt=%u wrote=%#lx read_back=%#lx SvIV=%#lx after_sv_setiv=%#lx\n",
           (unsigned)flags, (unsigned long)iv, (unsigned)rc, (unsigned long)MARK_B2,
           (unsigned long)iv2, (unsigned long)api, (unsigned long)api2);

    const int intact = SvTYPE(b) == SVt_IV && (flags & SVf_IOK) && iv == MARK_B && rc == 1
                       && PTR2UV(b) == b_addr;
    const int wrote = iv2 == MARK_B2 && api == MARK_B2 && api2 == MARK_B;
    char why[160];
    snprintf(why, sizeof why, "intact=%s write_read_back=%s", intact ? "yes" : "NO", wrote ? "yes" : "NO");
    return result("sibling", intact && wrote, why);
}

/* ---- 3. successor, and the stale reference ------------------------------------------------ */
static int control_successor(int stale)
{
    const char *mode = stale ? "stale" : "successor";
    SV *a = newSViv(MARK_A);
    const UV a_addr = PTR2UV(a);
    show("A", a);
    printf("FREE A refcnt=%u\n", (unsigned)SvREFCNT(a));
    SvREFCNT_dec(a);            /* `a` is kept: the stale variant reads through it */

    SV *c = NULL;
    int n;
    for (n = 1; n <= 4096; n++) {   /* heads that are not at A's address stay allocated */
        c = newSV(0);
        if (PTR2UV(c) == a_addr)
            break;
    }
    if (n > 4096)
        return result(mode, 0, "no head at A's address in 4096 newSV");
    show("C", c);
    printf("SUCCESSOR newSV_calls=%d A=%#lx C=%#lx equal=%s\n", n, (unsigned long)a_addr,
           (unsigned long)PTR2UV(c), PTR2UV(c) == a_addr ? "yes" : "NO");
    sv_setiv(c, MARK_C);

    ANNOUNCE("successor-read-flags"); U32 cflags = probe_read_flags(c);
    ANNOUNCE("successor-read-iv");    IV civ = probe_read_iv(c);
    ANNOUNCE("successor-write-iv");   probe_write_iv(c, MARK_C2);
    ANNOUNCE("successor-read-iv");    IV civ2 = probe_read_iv(c);
    printf("READ C flags=%#x iv=%#lx wrote=%#lx read_back=%#lx\n", (unsigned)cflags,
           (unsigned long)civ, (unsigned long)MARK_C2, (unsigned long)civ2);
    const int c_ok = PTR2UV(c) == a_addr && civ == MARK_C && civ2 == MARK_C2
                     && SvTYPE(c) == SVt_IV;
    if (!stale)
        return result(mode, c_ok, c_ok ? "C_at_A=yes C_read_write=yes" : "C check failed");
    if (!c_ok)
        return result(mode, 0, "C check failed before the stale read");

    printf("STALE about to read through the old pointer to A (%#lx)\n", (unsigned long)a_addr);
    ANNOUNCE("stale-read-flags"); U32 sflags = probe_read_flags(a);
    ANNOUNCE("stale-read-iv");    IV siv = probe_read_iv(a);
    printf("READ via_A flags=%#x iv=%#lx (C: flags=%#x iv=%#lx)\n", (unsigned)sflags,
           (unsigned long)siv, (unsigned)cflags, (unsigned long)MARK_C2);
    const int reads_c = sflags == cflags && siv == MARK_C2;
    return result(mode, reads_c, reads_c ? "stale_read_completed reads_C=yes"
                                         : "stale_read_completed reads_C=NO");
}

int main(int argc, char **argv, char **env)
{
    static char *args[] = { "", "-e", "0", NULL };
    int rc;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: svhead-controls parent|sibling|successor|stale\n");
        return 2;
    }
#ifdef PERL_CAPSTONE_SUBLET
    printf("BUILD protected (PERL_CAPSTONE_SUBLET)\n");
#else
    printf("BUILD release\n");
#endif
    PERL_SYS_INIT3(&argc, &argv, &env);
    my_perl = perl_alloc();
    perl_construct(my_perl);
    PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
    if (perl_parse(my_perl, NULL, 3, args, NULL) || perl_run(my_perl)) {
        printf("RESULT %s FAIL interpreter did not start\n", argv[1]);
        return 1;
    }
    printf("INTERP up sizeof(SV)=%u arena_heads=%u\n", (unsigned)sizeof(SV),
           (unsigned)SV_ARENA_HEADS);
    if (!strcmp(argv[1], "parent"))
        rc = control_parent();
    else if (!strcmp(argv[1], "sibling"))
        rc = control_sibling();
    else if (!strcmp(argv[1], "successor"))
        rc = control_successor(0);
    else if (!strcmp(argv[1], "stale"))
        rc = control_successor(1);
    else {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        rc = 2;
    }
    printf("DONE %s\n", argv[1]);
    perl_destruct(my_perl);
    perl_free(my_perl);
    PERL_SYS_TERM();
    return rc;
}
