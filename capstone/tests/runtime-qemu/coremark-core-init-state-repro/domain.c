#include "coremark.h"

void domain_main(unsigned *res, unsigned func) {
  ee_u8 mem[400];
  (void)func;
  core_init_state((ee_u32)sizeof(mem), (ee_s16)8, mem);
  *res = (unsigned)(unsigned char)mem[0];
}

