#define PERL_CORE
#include "EXTERN.h"
#include "perl.h"
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
unsigned long life_calls, life_fail;
static jmp_buf oom;
void __wrap_Perl_croak_no_mem(void) { longjmp(oom,1); }
static void xs_init(pTHX) { (void)0; }
int main(int argc, char **argv, char **env)
{
  int stale=argc>1 && !strcmp(argv[1],"stale");
  char *args[]={"unit","-e","0",NULL};
  int count=3;
  PERL_SYS_INIT3(&argc,&argv,&env);
  PerlInterpreter *interp=perl_alloc();
  perl_construct(interp);
  if (perl_parse(interp,xs_init,count,args,NULL)) return 10;
  SV *p=newSViv(42);
  if (stale) {
    SvREFCNT_dec(p);
    puts("TRAPTEST: perl stale armed");
    return ((volatile SV *)p)->sv_flags;
  }
  SvREFCNT_dec(p);
  for (unsigned long n=1;n<=3;n++) {
    life_calls=0; life_fail=n;
    for (unsigned long k=1;k<=n;k++) {
      SV *before=PL_sv_root;
      IV used=PL_sv_count;
      if (!setjmp(oom)) {
        p=newSViv(42);
        if (k==n) return 11;
      } else if (k!=n || PL_sv_root!=before || PL_sv_count!=used) return 12;
    }
    life_fail=0;
    p=newSViv(43);
    SvREFCNT_dec(p);
  }
  perl_destruct(interp);
  perl_free(interp);
  PERL_SYS_TERM();
  puts("TRAPTEST: perl injection PASS");
  return 0;
}
