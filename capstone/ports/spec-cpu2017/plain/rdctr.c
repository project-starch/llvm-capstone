/* rdctr: print the hart's cycle and retired-instruction counters from U mode (rdcycle, rdinstret; the board's OpenSBI
   sets scounteren=7, perf/T0b/STATUS.md section 2). Used before and after each plain SPEC command. */
#include <stdio.h>
int main(void) {
  unsigned long c, i;
  __asm__ volatile("rdcycle %0" : "=r"(c));
  __asm__ volatile("rdinstret %0" : "=r"(i));
  printf("cycles=%lu instret=%lu\n", c, i);
  return 0;
}
