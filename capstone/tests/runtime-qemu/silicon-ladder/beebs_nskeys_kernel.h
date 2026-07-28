#ifndef NSKEYS_KERNEL_H
#define NSKEYS_KERNEL_H
/* R-9 discriminator variant: reads ONE table, never the second. Isolates whether
 * touching TWO distinct cap-table globals in the same loop is what matters.
 *
 * beebs_ns hangs on silicon and R-1 does not predict it (neither table is ever
 * written). The prologue-scale hypothesis was refuted 2026-07-28 -- the copy path
 * shrank the domain from 3,676 to 2,024 b64 chars and it hung identically. These
 * variants each change exactly ONE property of the kernel so a board run says which
 * property matters. Data is byte-identical to beebs_ns where it is present. */
#define NS_REPS 16
const int nsk_keys[4][5][5][5] = {
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
  {
  {
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  },
  {
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  },
  {
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  },
  {
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  },
  {
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  {2,2,2,2,2},
  },
  },
  {
  {
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  },
  {
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  },
  {
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  },
  {
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  },
  {
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  {3,3,3,3,3},
  },
  },
  {
  {
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  },
  {
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  },
  {
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  },
  {
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  },
  {
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,4},
  {4,4,4,4,401},
  },
  },
};
static int nsk_foo(int x){
  for(int i=0;i<4;i++)for(int j=0;j<5;j++)for(int k=0;k<5;k++)for(int l=0;l<5;l++)
    if(nsk_keys[i][j][k][l]==x) return nsk_keys[i][j][k][l];
  return -1;
}
static unsigned nskeys_compute(void){
  unsigned h=2166136261u;
  for(int r=0;r<NS_REPS;r++){ h^=(unsigned)nsk_foo(((r&7)==7)?401:400); h*=16777619u; }
  return h;
}
#endif
