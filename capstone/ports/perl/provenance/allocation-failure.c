/* Independent host-only exception/allocator-boundary regression fixture.
 * Link with --wrap=malloc against a separately built ASan interpreter.
 * There is no fault hook in the production provenance adapter. */
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

static PerlInterpreter *my_perl;
static int failure_mode;
static unsigned failure_after;
void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size)
{
    if (failure_after && !--failure_after) {
        int mode = failure_mode;
        failure_mode = 0;
        const char *message = mode == 1 ? "ALLOCATION_FAULT catch\n" : "ALLOCATION_FAULT oom\n";
        (void)write(1, message, strlen(message));
        if (mode == 1) Perl_croak_nocontext("test-only allocation boundary exception");
        return NULL; /* Exercise safesysmalloc's real croak_no_mem path. */
    }
    return __real_malloc(size);
}

static XS(fail_replace)
{
    dXSARGS;
    const char *pointer;
    STRLEN length;
    STRLEN offset;
    if (items != 4) croak("fail_replace(target, source, offset, mode)");
    pointer = SvPV(ST(1), length);
    offset = SvUV(ST(2));
    failure_mode = SvIV(ST(3));
    failure_after = 1;
    Perl_cap_prov_add(aTHX_ ST(0), offset, sizeof(pointer), pointer);
    failure_after = 0;
    croak("allocation fault did not trigger");
}

static void xs_init(pTHX)
{
    newXS("main::fail_replace", fail_replace, __FILE__);
}

int main(int argc, char **argv, char **env)
{
    int status;
    PERL_SYS_INIT3(&argc, &argv, &env);
    my_perl = perl_alloc();
    if (!my_perl) return 2;
    perl_construct(my_perl);
    PL_perl_destruct_level = 2;
    PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
    status = perl_parse(my_perl, xs_init, argc, argv, NULL);
    if (!status) status = perl_run(my_perl);
    {
        int cleanup = perl_destruct(my_perl);
        if (!status) status = cleanup;
    }
    perl_free(my_perl);
    PERL_SYS_TERM();
    return status;
}
