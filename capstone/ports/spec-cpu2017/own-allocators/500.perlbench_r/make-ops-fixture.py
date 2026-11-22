#!/usr/bin/env python3
"""make-ops-fixture.py PERLMAIN_C OUT_C: write the op-slab fixture (nested lane, 2026-11-22) from a perlbench build's own
perlmain.c, which stays outside the repository: the file with its main renamed, then the fixture's main, which embeds
the interpreter (xs_init is the file's own) and checks the op lifetimes of port-op-slabs.py through the C API. Compile
OUT_C with perlmain.o's compile line plus the port's define and link it in perlmain.o's place. The ops are made with
newOP while PL_compcv is a fresh CV with no root and no start, so they come from that CV's slabs, as a compiling sub's
do. An op is read through an OP* the fixture kept in a volatile global, as a client holding a stale OP* would; the read
is of the op's pp function pointer. Modes:
  0  a live op reads its pp function
  1  the op freed (op_free): it must fault
  2  two ops, the first freed and a third made in its slot (the same address): the third and the second must read,
     then the first must fault
  3  3,000 ops over the slabs' growth (64 to 2,048 pointers); the 11th freed; the other 2,999 must read, then the
     11th must fault
  4  the CV freed with its ops still live (SvREFCNT_dec: opslab_force_free frees them through the slabs' own pointers,
     then the slabs): a kept op must fault
A build with PERL_CAPSTONE_SUBLET_OPS_NOREVOKE (the control) must read in modes 0-3. In mode 4 it must fault as well:
freeing a slab ends every op lifetime carved from it, revoked or not, so mode 4 checks the own-pointer path (a fault
in Perl_Slab_Free instead of main would mean an own pointer was taken for a child) and the slab's end, with no control
that separates the per-op revocation."""
import sys
HEADER = """/* perl-ops-fixture: perlmain.c with its main renamed; the main below embeds the interpreter and checks the op
 * lifetimes through the C API (nested lane, 2026-11-22; modes in make-ops-fixture.py). */
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
static OP *volatile kept;   /* a client's copy of an op pointer */
#define SAY(...) do { PerlIO_printf(PerlIO_stdout(), __VA_ARGS__); PerlIO_flush(PerlIO_stdout()); } while (0)
static OP *mk(void) { return newOP(OP_PUSHMARK, 0); }
static int live(OP *o) { return o->op_ppaddr == PL_ppaddr[OP_PUSHMARK]; }
int main(int argc, char **argv, char **env)
{
    int mode = argc > 1 ? atoi(argv[1]) : -1;
    void *volatile got;
    if (mode < 0 || mode > 4) return 2;
    PERL_SYS_INIT3(&argc, &argv, &env);
    fixture_boot();
    {
        dTHX;
        CV *cv = MUTABLE_CV(newSV_type(SVt_PVCV));
        PL_compcv = cv;   /* no root, no start: the first op makes the CV's slab */
        if (mode == 0 || mode == 1) {
            OP *o = mk();
            kept = o;
            got = (void *)kept->op_ppaddr;
            SAY("FIXTURE mode=%d live ok=%d slabbed=%d\n", mode, got == (void *)PL_ppaddr[OP_PUSHMARK], (int)o->op_slabbed);
            if (mode == 1) {
                op_free(o);
                SAY("FIXTURE mode=1 freed\n");
                got = (void *)kept->op_ppaddr;
                SAY("FIXTURE mode=1 read_after_free pp=%p\n", got);
            }
        } else if (mode == 2) {
            OP *o1 = mk(), *o2 = mk(), *o3;
            kept = o1;
            op_free(o1);
            o3 = mk();
            SAY("FIXTURE mode=2 reused_slot same=%d o3_ok=%d o2_ok=%d\n", PTR2UV(o3) == PTR2UV(o1), live(o3), live(o2));
            got = (void *)kept->op_ppaddr;
            SAY("FIXTURE mode=2 read_after_free pp=%p\n", got);
        } else if (mode == 3) {
            static OP *ops[3000];
            int i, bad = 0;
            for (i = 0; i < 3000; i++) ops[i] = mk();
            kept = ops[10];
            op_free(ops[10]);
            for (i = 0; i < 3000; i++) if (i != 10 && !live(ops[i])) bad++;
            SAY("FIXTURE mode=3 live_ok=%d bad=%d\n", 2999 - bad, bad);
            got = (void *)kept->op_ppaddr;
            SAY("FIXTURE mode=3 read_after_free pp=%p\n", got);
        } else {
            int i;
            for (i = 0; i < 100; i++) (void)mk();
            kept = mk();
            PL_compcv = NULL;
            SvREFCNT_dec(cv);
            SAY("FIXTURE mode=4 cv_freed\n");
            got = (void *)kept->op_ppaddr;
            SAY("FIXTURE mode=4 read_after_free pp=%p\n", got);
        }
    }
    return 0;
}
"""
src = open(sys.argv[1]).read()
assert src.count('xs_init(pTHX)') >= 1 and src.count('main(') >= 1, 'not a perlmain.c'
open(sys.argv[2], 'w').write(HEADER + src + '\n' + TAIL)
print('fixture written', sys.argv[2])
