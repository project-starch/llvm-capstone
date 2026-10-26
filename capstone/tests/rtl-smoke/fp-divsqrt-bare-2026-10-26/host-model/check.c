#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "vec.h"
double rtl_fdiv_d(double,double); double rtl_fsqrt_d(double); float rtl_fdiv_s(float,float); float rtl_fsqrt_s(float);
static double D(uint64_t u){double x;memcpy(&x,&u,8);return x;} static uint64_t U(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static float F(uint32_t u){float x;memcpy(&x,&u,4);return x;} static uint32_t UF(float x){uint32_t u;memcpy(&u,&x,4);return u;}
int main(void){int bd=0,bs=0,bds=0,bss=0,id=0,is=0,ids=0,iss=0;
 for(int i=0;i<512;i++){double a=D(VD[i][0]),b=D(VD[i][1]);
  bd+=U(rtl_fdiv_d(a,b))!=VD[i][2]; bs+=U(rtl_fsqrt_d(fabs(a)))!=VD[i][3];
  id+=U(a/b)!=VD[i][2]; is+=U(sqrt(fabs(a)))!=VD[i][3];
  float fa=F(VS[i][0]),fb=F(VS[i][1]);
  bds+=UF(rtl_fdiv_s(fa,fb))!=VS[i][2]; bss+=UF(rtl_fsqrt_s(fabsf(fa)))!=VS[i][3];
  ids+=UF(fa/fb)!=VS[i][2]; iss+=UF(sqrtf(fabsf(fa)))!=VS[i][3];}
 printf("model vs RTL trace, mismatches of 512: fdiv.d %d fsqrt.d %d fdiv.s %d fsqrt.s %d\n",bd,bs,bds,bss);
 printf("IEEE  vs RTL trace (control, must equal the sim miss counts 23 19 53 27): %d %d %d %d\n",id,is,ids,iss);
 return (bd|bs|bds|bss)!=0;}
