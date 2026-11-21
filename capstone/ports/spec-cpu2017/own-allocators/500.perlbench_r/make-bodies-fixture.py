#!/usr/bin/env python3
"""make-bodies-fixture.py PERLMAIN_C OUT_C: write the SV-bodies fixture (nested lane, 2026-11-22) from a perlbench
build's own perlmain.c, which stays outside the repository: the file with its main renamed, then the fixture's main,
which embeds the interpreter (xs_init is the file's own) and checks the body lifetimes of port-sv-bodies.py through the
C API. Compile OUT_C with perlmain.o's compile line and link it in perlmain.o's place. A body is read through a pointer
the fixture kept in a volatile global, as a client holding a stale XPV* would. Modes:
  0  a live PV body read through a stored and reloaded SvANY (the 32-byte offset below the body's base)
  1  a PV freed (SvREFCNT_dec): its old body must fault
  2  a PV upgraded to PVMG (sv_upgrade frees the old body): the old body must fault, the new one reads
  3  5,000 PV bodies and 500 AV bodies over many arenas; the 11th PV freed; the other 4,999 PVs and every AV must read
     their own values, then the 11th's body must fault
  4  a PV body of one interpreter, then that interpreter destroyed (its segments freed) and a second one built: 3,000
     new PV bodies must read, then the old body must fault
A build with PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE (the control) must read in modes 0-3. In mode 4 it faults as well
(twin, 2026-11-22): freeing a segment ends every lifetime carved from it, revoked bodies or not, so mode 4 shows that
an interpreter's destruction ends its bodies, with no control that separates the per-body revocation."""
import sys
HEADER = """/* perl-bodies-fixture: perlmain.c with its main renamed; the main below embeds the interpreter and checks the SV
 * body lifetimes through the C API (nested lane, 2026-11-22; modes in make-bodies-fixture.py). */
#define main perlbench_real_main
"""
TAIL = r"""
#undef main
static void fixture_boot(void)
{
    static char *args[] = { "perl", "-e", "0", NULL };
    my_perl = perl_alloc();
    perl_construct(my_perl);
    PL_perl_destruct_level = 2;
    PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
    perl_parse(my_perl, xs_init, 3, args, (char **)NULL);
}
static XPV *volatile kept;   /* a client's copy of a body pointer */
#define SAY(...) do { PerlIO_printf(PerlIO_stdout(), __VA_ARGS__); PerlIO_flush(PerlIO_stdout()); } while (0)
int main(int argc, char **argv, char **env)
{
    int mode = argc > 1 ? atoi(argv[1]) : -1;
    volatile STRLEN c;
    if (mode < 0 || mode > 4) return 2;
    PERL_SYS_INIT3(&argc, &argv, &env);
    fixture_boot();
    if (mode == 0) {
        SV *sv = newSVpvs("hello");
        kept = (XPV *)SvANY(sv);
        c = kept->xpv_cur;
        SAY("FIXTURE mode=0 read cur=%lu\n", (unsigned long)c);
    } else if (mode == 1) {
        SV *sv = newSVpvs("hello");
        kept = (XPV *)SvANY(sv);
        SvREFCNT_dec(sv);
        SAY("FIXTURE mode=1 freed\n");
        c = kept->xpv_cur;
        SAY("FIXTURE mode=1 read_after_free cur=%lu\n", (unsigned long)c);
    } else if (mode == 2) {
        SV *sv = newSVpvs("hello");
        kept = (XPV *)SvANY(sv);
        sv_upgrade(sv, SVt_PVMG);
        SAY("FIXTURE mode=2 upgraded new_cur=%lu\n", (unsigned long)SvCUR(sv));
        c = kept->xpv_cur;
        SAY("FIXTURE mode=2 old_body_read cur=%lu\n", (unsigned long)c);
    } else if (mode == 3) {
        static SV *s[5000];
        static AV *a[500];
        static XPV *b[5000];
        int i, bad = 0, abad = 0;
        char buf[16];
        for (i = 0; i < 5000; i++) {
            s[i] = newSVpvn(buf, my_snprintf(buf, sizeof buf, "%d", i));
            b[i] = (XPV *)SvANY(s[i]);
            if (i % 10 == 0) { a[i / 10] = newAV(); av_extend(a[i / 10], i / 10); }
        }
        kept = b[10];
        SvREFCNT_dec(s[10]);
        for (i = 0; i < 5000; i++) if (i != 10 && b[i]->xpv_cur != (STRLEN)my_snprintf(buf, sizeof buf, "%d", i)) bad++;
        for (i = 0; i < 500; i++) if (AvMAX(a[i]) < i) abad++;
        SAY("FIXTURE mode=3 live_read_ok=%d bad=%d av_ok=%d av_bad=%d\n", 4999 - bad, bad, 500 - abad, abad);
        c = kept->xpv_cur;
        SAY("FIXTURE mode=3 read_after_free cur=%lu\n", (unsigned long)c);
    } else {
        static SV *w[3000];
        int i, bad = 0;
        SV *old = newSVpvs("hello");
        kept = (XPV *)SvANY(old);
        perl_destruct(my_perl);
        perl_free(my_perl);
        fixture_boot();
        for (i = 0; i < 3000; i++) w[i] = newSVpvs("abc");
        for (i = 0; i < 3000; i++) if (SvCUR(w[i]) != 3) bad++;
        SAY("FIXTURE mode=4 second_interpreter_read_ok=%d bad=%d\n", 3000 - bad, bad);
        c = kept->xpv_cur;
        SAY("FIXTURE mode=4 old_body_read cur=%lu\n", (unsigned long)c);
    }
    return 0;
}
"""
src = open(sys.argv[1]).read()
assert src.count('xs_init(pTHX)') >= 1 and src.count('main(') >= 1, 'not a perlmain.c'
open(sys.argv[2], 'w').write(HEADER + src + '\n' + TAIL)
print('fixture written', sys.argv[2])
