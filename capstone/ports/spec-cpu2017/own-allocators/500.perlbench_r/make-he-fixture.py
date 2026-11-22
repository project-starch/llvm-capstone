#!/usr/bin/env python3
"""make-he-fixture.py PERLMAIN_C OUT_C: write the hash-entry fixture (nested lane, 2026-11-22) from a perlbench build's
own perlmain.c, which stays outside the repository: the file with its main renamed, then the fixture's main, which
embeds the interpreter (xs_init is the file's own) and checks the HE lifetimes of port-he.py through the C API.
Compile OUT_C with perlmain.o's compile line and link it in perlmain.o's place. An entry is read through an HE* the
fixture kept in a volatile global, as a client holding a stale HE* would; the read is of the entry's value pointer.
Modes:
  0  a live entry (hv_fetch_ent) reads its value
  1  the key deleted (hv_delete): the old entry must fault
  2  an entry kept while its hash grows from 11 to 5,000 keys (the splits move entries between buckets, they free
     none): it must still read its value, every key must fetch its own value, then the key deleted and the entry
     must fault
  3  hv_clear of 100 keys: a kept entry must fault
  4  the hash itself freed (SvREFCNT_dec of 100 keys): a kept entry must fault
A build with PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE (the control) must read in every mode."""
import sys
HEADER = """/* perl-he-fixture: perlmain.c with its main renamed; the main below embeds the interpreter and checks the hash
 * entry lifetimes through the C API (nested lane, 2026-11-22; modes in make-he-fixture.py). */
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
static HE *volatile kept;   /* a client's copy of an entry pointer */
#define SAY(...) do { PerlIO_printf(PerlIO_stdout(), __VA_ARGS__); PerlIO_flush(PerlIO_stdout()); } while (0)
static SV *key(int i) { return sv_2mortal(newSVpvf("k%d", i)); }
static SV *put(HV *hv, int i) { SV *v = newSViv(i); hv_store_ent(hv, key(i), v, 0); return v; }
int main(int argc, char **argv, char **env)
{
    int mode = argc > 1 ? atoi(argv[1]) : -1;
    SV *volatile got;
    if (mode < 0 || mode > 4) return 2;
    PERL_SYS_INIT3(&argc, &argv, &env);
    fixture_boot();
    {
        dTHX;
        HV *hv = newHV();
        if (mode == 0 || mode == 1) {
            SV *v = put(hv, 7);
            kept = hv_fetch_ent(hv, key(7), 0, 0);
            got = kept->he_valu.hent_val;
            SAY("FIXTURE mode=%d live val_ok=%d\n", mode, got == v);
            if (mode == 1) {
                (void)hv_delete_ent(hv, key(7), G_DISCARD, 0);
                SAY("FIXTURE mode=1 deleted\n");
                got = kept->he_valu.hent_val;
                SAY("FIXTURE mode=1 read_after_delete val=%p\n", (void *)got);
            }
        } else if (mode == 2) {
            static SV *vals[5000];
            int i, bad = 0;
            for (i = 0; i < 11; i++) vals[i] = put(hv, i);
            kept = hv_fetch_ent(hv, key(10), 0, 0);
            for (; i < 5000; i++) vals[i] = put(hv, i);
            got = kept->he_valu.hent_val;
            for (i = 0; i < 5000; i++) { HE *he = hv_fetch_ent(hv, key(i), 0, 0); if (!he || HeVAL(he) != vals[i]) bad++; }
            SAY("FIXTURE mode=2 kept_after_splits val_ok=%d fetch_ok=%d bad=%d max=%lu\n", got == vals[10], 5000 - bad,
                bad, (unsigned long)HvMAX(hv));
            (void)hv_delete_ent(hv, key(10), G_DISCARD, 0);
            got = kept->he_valu.hent_val;
            SAY("FIXTURE mode=2 read_after_delete val=%p\n", (void *)got);
        } else {
            int i;
            for (i = 0; i < 100; i++) put(hv, i);
            kept = hv_fetch_ent(hv, key(50), 0, 0);
            if (mode == 3) hv_clear(hv); else SvREFCNT_dec(hv);
            SAY("FIXTURE mode=%d %s\n", mode, mode == 3 ? "cleared" : "hash_freed");
            got = kept->he_valu.hent_val;
            SAY("FIXTURE mode=%d read_after_free val=%p\n", mode, (void *)got);
        }
    }
    return 0;
}
"""
src = open(sys.argv[1]).read()
assert src.count('xs_init(pTHX)') >= 1 and src.count('main(') >= 1, 'not a perlmain.c'
open(sys.argv[2], 'w').write(HEADER + src + '\n' + TAIL)
print('fixture written', sys.argv[2])
