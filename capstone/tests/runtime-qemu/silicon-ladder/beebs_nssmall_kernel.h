#ifndef NSSMALL_KERNEL_H
#define NSSMALL_KERNEL_H
/* R-9 discriminator variant: 125 entries instead of 500, same 4-level shape.
 * The size test pre-registered in ISSUES under R-9.
 *
 * beebs_ns hangs on silicon and R-1 does not predict it (neither table is ever
 * written). The prologue-scale hypothesis was refuted 2026-07-28 -- the copy path
 * shrank the domain from 3,676 to 2,024 b64 chars and it hung identically. These
 * variants each change exactly ONE property of the kernel so a board run says which
 * property matters. Data is byte-identical to beebs_ns where it is present. */
#define NS_REPS 16
const int nss_keys[1][5][5][5] = {
  {
  {
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  },
  {
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  },
  {
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  },
  {
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  },
  {
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  {1,1,1,1,1},
  },
  },
};
int nss_answer[1][5][5][5] = {
  {
  {
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  },
  {
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  },
  {
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  },
  {
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  },
  {
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  {234,234,234,234,234},
  },
  },
};
static int nss_foo(int x){
  for(int i=0;i<1;i++)for(int j=0;j<5;j++)for(int k=0;k<5;k++)for(int l=0;l<5;l++)
    if(nss_keys[i][j][k][l]==x) return nss_answer[i][j][k][l]+nss_keys[i][j][k][l];
  return -1;
}
static unsigned nssmall_compute(void){
  unsigned h=2166136261u;
  for(int r=0;r<NS_REPS;r++){ h^=(unsigned)nss_foo(((r&7)==7)?401:400); h*=16777619u; }
  return h;
}
#endif
