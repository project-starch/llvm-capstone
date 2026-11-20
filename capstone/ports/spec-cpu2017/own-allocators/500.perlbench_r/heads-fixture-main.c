/* perl-heads-fixture: perlmain.c with its main renamed; the main below embeds the interpreter and checks the
 * scalar-head lifetimes through the C API (nested lane, 2026-11-21). MODE:
 *   0  a live head reads
 *   1  a head freed (SvREFCNT_dec) and read through the client's pointer: must fault
 *   2  5,000 heads over many arenas; the 11th freed; all 4,999 others must read, then the 11th must fault
 *   3  a head of one interpreter, then that interpreter destroyed (its arenas freed and the arena table cleared) and a
 *      second one built, whose arenas may take the same addresses and table slots: 3,000 new heads must read, then
 *      the old head must fault  */
#define main perlbench_real_main
/*    miniperlmain.c
 *
 *    Copyright (C) 1994, 1995, 1996, 1997, 1999, 2000, 2001, 2002, 2003,
 *    2004, 2005, 2006, 2007, by Larry Wall and others
 *
 *    You may distribute under the terms of either the GNU General Public
 *    License or the Artistic License, as specified in the README file.
 *
 */

/*
 *      The Road goes ever on and on
 *          Down from the door where it began.
 *
 *     [Bilbo on p.35 of _The Lord of the Rings_, I/i: "A Long-Expected Party"]
 *     [Frodo on p.73 of _The Lord of the Rings_, I/iii: "Three Is Company"]
 */

/* This file contains the main() function for the perl interpreter.
 * Note that miniperlmain.c contains main() for the 'miniperl' binary,
 * while perlmain.c contains main() for the 'perl' binary.
 *
 * Miniperl is like perl except that it does not support dynamic loading,
 * and in fact is used to build the dynamic modules needed for the 'real'
 * perl executable.
 */

#if defined(SPEC_WINDOWS)
# include <signal.h>
#endif

#ifdef OEMVS
#ifdef MYMALLOC
/* sbrk is limited to first heap segment so make it big */
#pragma runopts(HEAP(8M,500K,ANYWHERE,KEEP,8K,4K) STACK(,,ANY,) ALL31(ON))
#else
#pragma runopts(HEAP(2M,500K,ANYWHERE,KEEP,8K,4K) STACK(,,ANY,) ALL31(ON))
#endif
#endif

#define PERL_IN_MINIPERLMAIN_C
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

static void xs_init (pTHX);
static PerlInterpreter *my_perl;

#if defined(PERL_GLOBAL_STRUCT_PRIVATE)
/* The static struct perl_vars* may seem counterproductive since the
 * whole idea PERL_GLOBAL_STRUCT_PRIVATE was to avoid statics, but note
 * that this static is not in the shared perl library, the globals PL_Vars
 * and PL_VarsPtr will stay away. */
static struct perl_vars* my_plvarsp;
struct perl_vars* Perl_GetVarsPrivate(void) { return my_plvarsp; }
#endif

#ifdef NO_ENV_ARRAY_IN_MAIN
extern char **environ;
int
main(int argc, char **argv)
#else
int
main(int argc, char **argv, char **env)
#endif
{
    int exitstatus, i;
#ifdef PERL_GLOBAL_STRUCT
    struct perl_vars *my_vars = init_global_struct();
#  ifdef PERL_GLOBAL_STRUCT_PRIVATE
    int veto;

    my_plvarsp = my_vars;
#  endif
#endif /* PERL_GLOBAL_STRUCT */
#ifndef NO_ENV_ARRAY_IN_MAIN
    PERL_UNUSED_ARG(env);
#endif
#ifndef PERL_USE_SAFE_PUTENV
    PL_use_safe_putenv = FALSE;
#endif /* PERL_USE_SAFE_PUTENV */

    /* if user wants control of gprof profiling off by default */
    /* noop unless Configure is given -Accflags=-DPERL_GPROF_CONTROL */
    PERL_GPROF_MONCONTROL(0);

#ifdef NO_ENV_ARRAY_IN_MAIN
    PERL_SYS_INIT3(&argc,&argv,&environ);
#else
    PERL_SYS_INIT3(&argc,&argv,&env);
#endif

#if defined(USE_ITHREADS)
    /* XXX Ideally, this should really be happening in perl_alloc() or
     * perl_construct() to keep libperl.a transparently fork()-safe.
     * It is currently done here only because Apache/mod_perl have
     * problems due to lack of a call to cancel pthread_atfork()
     * handlers when shared objects that contain the handlers may
     * be dlclose()d.  This forces applications that embed perl to
     * call PTHREAD_ATFORK() explicitly, but if and only if it hasn't
     * been called at least once before in the current process.
     * --GSAR 2001-07-20 */
    PTHREAD_ATFORK(Perl_atfork_lock,
                   Perl_atfork_unlock,
                   Perl_atfork_unlock);
#endif

    PERL_SYS_FPU_INIT;

    if (!PL_do_undump) {
	my_perl = perl_alloc();
	if (!my_perl)
	    exit(1);
	perl_construct(my_perl);
	PL_perl_destruct_level = 0;
    }
    PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
    exitstatus = perl_parse(my_perl, xs_init, argc, argv, (char **)NULL);
    if (!exitstatus)
        perl_run(my_perl);

#ifndef PERL_MICRO
    /* Unregister our signal handler before destroying my_perl */
    for (i = 1; PL_sig_name[i]; i++) {
	if (rsignal_state(PL_sig_num[i]) == (Sighandler_t) PL_csighandlerp) {
	    rsignal(PL_sig_num[i], (Sighandler_t) SIG_DFL);
	}
    }
#endif

    exitstatus = perl_destruct(my_perl);

    perl_free(my_perl);

#if defined(USE_ENVIRON_ARRAY) && defined(PERL_TRACK_MEMPOOL) && !defined(NO_ENV_ARRAY_IN_MAIN)
    /*
     * The old environment may have been freed by perl_free()
     * when PERL_TRACK_MEMPOOL is defined, but without having
     * been restored by perl_destruct() before (this is only
     * done if destruct_level > 0).
     *
     * It is important to have a valid environment for atexit()
     * routines that are eventually called.
     */
    environ = env;
#endif

    PERL_SYS_TERM();

#ifdef PERL_GLOBAL_STRUCT
#  ifdef PERL_GLOBAL_STRUCT_PRIVATE
    veto = my_plvarsp->Gveto_cleanup;
#  endif
    free_global_struct(my_vars);
#  ifdef PERL_GLOBAL_STRUCT_PRIVATE
    if (!veto)
        my_plvarsp = NULL;
    /* Remember, functions registered with atexit() can run after this point,
       and may access "global" variables, and hence end up calling
       Perl_GetVarsPrivate()  */
#endif
#endif /* PERL_GLOBAL_STRUCT */

    exit(exitstatus);
#if defined(SPEC)
    return(exitstatus); /* Quiet warnings about non-void functions not returning a value */
#endif
}

/* Register any extra external extensions */

EXTERN_C void boot_DynaLoader (pTHX_ CV* cv);
EXTERN_C void boot_Cwd (pTHX_ CV* cv);
EXTERN_C void boot_Data__Dumper (pTHX_ CV* cv);
EXTERN_C void boot_Devel__Peek (pTHX_ CV* cv);
EXTERN_C void boot_Digest__MD5 (pTHX_ CV* cv);
EXTERN_C void boot_Digest__SHA (pTHX_ CV* cv);
EXTERN_C void boot_File__Glob (pTHX_ CV* cv);
EXTERN_C void boot_Hash__Util (pTHX_ CV* cv);
EXTERN_C void boot_Hash__Util__FieldHash (pTHX_ CV* cv);
EXTERN_C void boot_HTML__Parser (pTHX_ CV* cv);
EXTERN_C void boot_IO (pTHX_ CV* cv);
EXTERN_C void boot_List__Util (pTHX_ CV* cv);
EXTERN_C void boot_MIME__Base64 (pTHX_ CV* cv);
EXTERN_C void boot_Opcode (pTHX_ CV* cv);
EXTERN_C void boot_PerlIO__scalar (pTHX_ CV* cv);
EXTERN_C void boot_PerlIO__via (pTHX_ CV* cv);
EXTERN_C void boot_Storable (pTHX_ CV* cv);
EXTERN_C void boot_Sys__Hostname (pTHX_ CV* cv);
EXTERN_C void boot_Tie__Hash__NamedCapture (pTHX_ CV* cv);
EXTERN_C void boot_Time__HiRes (pTHX_ CV* cv);
EXTERN_C void boot_arybase (pTHX_ CV* cv);
EXTERN_C void boot_attributes (pTHX_ CV* cv);
EXTERN_C void boot_mro (pTHX_ CV* cv);
EXTERN_C void boot_re (pTHX_ CV* cv);

static void
xs_init(pTHX)
{
    static const char file[] = __FILE__;
    dXSUB_SYS;
    PERL_UNUSED_CONTEXT;

    /* DynaLoader is a special case */
    newXS("DynaLoader::boot_DynaLoader", boot_DynaLoader, file);
    newXS("Cwd::bootstrap", boot_Cwd, file);
    newXS("Data::Dumper::bootstrap", boot_Data__Dumper, file);
    newXS("Devel::Peek::bootstrap", boot_Devel__Peek, file);
    newXS("Digest::MD5::bootstrap", boot_Digest__MD5, file);
    newXS("Digest::SHA::bootstrap", boot_Digest__SHA, file);
    newXS("File::Glob::bootstrap", boot_File__Glob, file);
    newXS("IO::bootstrap", boot_IO, file);
    newXS("Hash::Util::bootstrap", boot_Hash__Util, file);
    newXS("Hash::Util::FieldHash::bootstrap", boot_Hash__Util__FieldHash, file);
    newXS("HTML::Parser::bootstrap", boot_HTML__Parser, file);
    newXS("List::Util::bootstrap", boot_List__Util, file);
    newXS("MIME::Base64::bootstrap", boot_MIME__Base64, file);
    newXS("Opcode::bootstrap", boot_Opcode, file);
    newXS("PerlIO::scalar::bootstrap", boot_PerlIO__scalar, file);
    newXS("PerlIO::via::bootstrap", boot_PerlIO__via, file);
    newXS("Storable::bootstrap", boot_Storable, file);
    newXS("Sys::Hostname::bootstrap", boot_Sys__Hostname, file);
    newXS("Tie::Hash::NamedCapture::bootstrap", boot_Tie__Hash__NamedCapture, file);
    newXS("Time::HiRes::bootstrap", boot_Time__HiRes, file);
    newXS("arybase::bootstrap", boot_arybase, file);
    newXS("attributes::bootstrap", boot_attributes, file);
    newXS("mro::bootstrap", boot_mro, file);
    newXS("re::bootstrap", boot_re, file);
}

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
int main(int argc, char **argv, char **env)
{
    int mode = argc > 1 ? atoi(argv[1]) : -1;
    volatile IV v;
    volatile U32 f;
    if (mode < 0 || mode > 3) return 2;
    PERL_SYS_INIT3(&argc, &argv, &env);
    fixture_boot();
    if (mode == 0) {
        SV *sv = newSViv(7);
        v = SvIV(sv);
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=0 read=%ld\n", (long)v);
    } else if (mode == 1) {
        SV *sv = newSViv(7);
        SvREFCNT_dec(sv);
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=1 freed\n"); PerlIO_flush(PerlIO_stdout());
        f = SvFLAGS(sv);
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=1 read_after_free flags=%lx\n", (unsigned long)f);
    } else if (mode == 2) {
        static SV *s[5000];
        int i, bad = 0;
        for (i = 0; i < 5000; i++) s[i] = newSViv(i);
        SvREFCNT_dec(s[10]);
        for (i = 0; i < 5000; i++) if (i != 10 && SvIV(s[i]) != i) bad++;
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=2 live_read_ok=%d bad=%d\n", 4999 - bad, bad); PerlIO_flush(PerlIO_stdout());
        f = SvFLAGS(s[10]);
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=2 read_after_free flags=%lx\n", (unsigned long)f);
    } else {
        static SV *w[3000];
        int i, bad = 0;
        SV *old = newSViv(7);
        perl_destruct(my_perl);
        perl_free(my_perl);
        fixture_boot();
        for (i = 0; i < 3000; i++) w[i] = newSViv(i);
        for (i = 0; i < 3000; i++) if (SvIV(w[i]) != i) bad++;
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=3 second_interpreter_read_ok=%d bad=%d\n", 3000 - bad, bad); PerlIO_flush(PerlIO_stdout());
        f = SvFLAGS(old);
        PerlIO_printf(PerlIO_stdout(), "FIXTURE mode=3 old_head_read flags=%lx\n", (unsigned long)f);
    }
    return 0;
}
