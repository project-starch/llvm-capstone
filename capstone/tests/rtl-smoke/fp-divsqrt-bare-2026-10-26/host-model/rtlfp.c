/* A model of fpnew 0.7.0's fpu_div_sqrt_mvp rounding (RNE): the quotient / root is computed to K extra bits
 * (FP64: 4 on the 1.x path, 3 on the 0.1x path; FP32: 3 and 2) and the sticky bit is the OR of those extra bits
 * only, never the remainder. Normal operands and normal results only; anything else falls back to IEEE. */
#include <math.h>
#include <stdint.h>
#include <string.h>
typedef unsigned __int128 u128;
static uint64_t d2u(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static double u2d(uint64_t u){double x;memcpy(&x,&u,8);return x;}
static uint32_t f2u(float x){uint32_t u;memcpy(&u,&x,4);return u;}
static float u2f(uint32_t u){float x;memcpy(&x,&u,4);return x;}
static u128 isqrt128(u128 n){u128 r=0,b=(u128)1<<126;while(b>n)b>>=2;while(b){if(n>=r+b){n-=r+b;r=(r>>1)+b;}else r>>=1;b>>=2;}return r;}
static uint64_t rnd(u128 Q,int k){uint64_t res=(uint64_t)(Q>>k);int r=(int)((Q>>(k-1))&1);int st=(Q&(((u128)1<<(k-1))-1))!=0;return res+(r&&(st||(res&1)));}
static long nd, ns, nds, nss;   /* call counts, printed by the harness as the rewrite's positive control */
double rtl_fdiv_d(double a,double b){
  nd++;
  double ieee=a/b; uint64_t ua=d2u(a),ub=d2u(b); int ea=(ua>>52)&0x7ff,eb=(ub>>52)&0x7ff;
  if(!ea||ea==0x7ff||!eb||eb==0x7ff||!isnormal(ieee)) return ieee;
  uint64_t ma=(ua&((1ULL<<52)-1))|(1ULL<<52), mb=(ub&((1ULL<<52)-1))|(1ULL<<52);
  u128 Q=((u128)ma<<56)/mb; int hi=(int)((Q>>56)&1); uint64_t m=rnd(Q,hi?4:3);
  int E=(ea-1023)-(eb-1023)-(hi?0:1);
  double r=ldexp((double)m,E-52); if(((ua^ub)>>63)&1) r=-r;
  return isnormal(r)?r:ieee;
}
double rtl_fsqrt_d(double x){
  ns++;
  double ieee=sqrt(x); uint64_t u=d2u(x); int e=(u>>52)&0x7ff;
  if(!e||e==0x7ff||(u>>63)) return ieee;
  uint64_t m=(u&((1ULL<<52)-1))|(1ULL<<52); int E=e-1023; u128 rad=(E&1)?((u128)m<<1):(u128)m; int E2=(E&1)?E-1:E;
  u128 Q=isqrt128(rad<<(52+8)); uint64_t mr=rnd(Q,4);
  return ldexp((double)mr,E2/2-52);
}
float rtl_fdiv_s(float a,float b){
  nds++;
  float ieee=a/b; uint32_t ua=f2u(a),ub=f2u(b); int ea=(ua>>23)&0xff,eb=(ub>>23)&0xff;
  if(!ea||ea==0xff||!eb||eb==0xff||!isnormal(ieee)) return ieee;
  uint64_t ma=(ua&0x7fffff)|(1u<<23), mb=(ub&0x7fffff)|(1u<<23);
  u128 Q=((u128)ma<<26)/mb; int hi=(int)((Q>>26)&1); uint64_t m=rnd(Q,hi?3:2);
  int E=(ea-127)-(eb-127)-(hi?0:1);
  float r=(float)ldexp((double)m,E-23); if(((ua^ub)>>31)&1) r=-r;
  return isnormal(r)?r:ieee;
}
float rtl_fsqrt_s(float x){
  nss++;
  float ieee=sqrtf(x); uint32_t u=f2u(x); int e=(u>>23)&0xff;
  if(!e||e==0xff||(u>>31)) return ieee;
  uint64_t m=(u&0x7fffff)|(1u<<23); int E=e-127; u128 rad=(E&1)?((u128)m<<1):(u128)m; int E2=(E&1)?E-1:E;
  u128 Q=isqrt128(rad<<(23+6)); uint64_t mr=rnd(Q,3);
  return (float)ldexp((double)mr,E2/2-23);
}
void rtl_counts(long *a,long *b,long *c,long *d){*a=nd;*b=ns;*c=nds;*d=nss;}
