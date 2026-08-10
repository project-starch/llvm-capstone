/* Native oracle for the s06copy rung. All 32 bytes must survive a block copy, so the oracle
 * is 32; hardware with S-06 returns 16 (each 16-byte chunk keeps its low 8 bytes only). */
#include <stdio.h>
int main(void){printf("32\n");return 0;}
