#include <stdlib.h>
#include <unistd.h>
#include <string.h>
int main(void) {
  char *a = malloc(17), *b = malloc(1000);
  if (!a || !b) return 1;
  memset(a, 1, 17); memset(b, 2, 1000);
  write(2, "MEMPHASE allocated\n", 19);
  free(a); free(b);
  write(2, "MEMPHASE released\n", 18);
  write(1, "EXP-OK calibration\n", 19);
  return 0;
}
