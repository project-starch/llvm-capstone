/* Native oracle for r14d_app.c. Expect 16. */
#include <stdio.h>
#include <string.h>
int main(void)
{
  const char *f[64]; unsigned i; int ok = 0;
  f[0]="ltrim"; f[1]="rtrim"; f[2]="trim"; f[3]="max"; f[4]="min"; f[5]="typeof";
  f[6]="length"; f[7]="instr"; f[8]="substr"; f[9]="upper"; f[10]="lower";
  f[11]="coalesce"; f[12]="hex"; f[13]="unhex"; f[14]="quote"; f[15]="replace";
  for (i=16;i<64;i++) f[i]="filler";
  for (i=0;i<16;i++) if (f[i] && strlen(f[i])>0) ok++;
  printf("%d\n", ok);
  return 0;
}
