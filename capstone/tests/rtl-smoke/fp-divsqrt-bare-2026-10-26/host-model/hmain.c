#include <stdio.h>
int benchmark(void); int initialise_benchmark(void); int verify_benchmark(int);
void rtl_counts(long*,long*,long*,long*);
int main(void){ volatile int r=0; initialise_benchmark();
  for(int i=0;i<2;i++){ initialise_benchmark(); r=benchmark(); }
  int ok=verify_benchmark(r); long a,b,c,d; rtl_counts(&a,&b,&c,&d);
  printf("verify=%d  model calls: fdiv.d %ld fsqrt.d %ld fdiv.s %ld fsqrt.s %ld\n",ok,a,b,c,d); return !ok; }
