#include <stdio.h>
int benchmark(void); int initialise_benchmark(void); int verify_benchmark(int); unsigned long long beebs_state_check(int);
unsigned long long bench_fnv(unsigned long long h, const void *p, unsigned long n){const unsigned char*b=p;for(unsigned long i=0;i<n;i++){h^=b[i];h*=1099511628211ULL;}return h;}
static volatile int result;
static void unit(void){ initialise_benchmark(); for(int i=0;i<9900;i++){ initialise_benchmark(); result=benchmark(); } (void)verify_benchmark(result); }
int main(void){ for(int u=0;u<3;u++){ unit(); printf("unit %d check=%016llx\n",u,beebs_state_check(result)); } return 0; }
