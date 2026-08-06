#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
static inline unsigned long rd(void){ unsigned long v; __asm__ volatile("rdinstret %0":"=r"(v)); return v; }
int main(int argc, char **argv){
  if(argc<2){ fprintf(stderr,"usage: %s <cmd>\n", argv[0]); return 2; }
  unsigned long t0=rd();
  int rc=system(argv[1]);
  unsigned long t1=rd();
  /* Peak RSS of the benchmark child. RUSAGE_CHILDREN reflects terminated children
   * (and their descendants), and this process runs exactly one system() child per
   * invocation, so ru_maxrss is that run's peak resident set (KB on FreeBSD/CheriBSD).
   * The memory analogue of the instruction bracket: temporal maxrss - spatial maxrss
   * is CHERI's temporal-safety memory cost (revocation quarantine + shadow bitmap). */
  struct rusage ru;
  getrusage(RUSAGE_CHILDREN, &ru);
  printf("BENCH instrs=%lu rc=%d maxrss_kb=%ld\n", t1-t0, rc, ru.ru_maxrss);
  fflush(stdout);
  return 0;
}
