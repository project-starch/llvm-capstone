#include <stdio.h>
/* correct hardware: SBR_BASE (0xB0000) with every arm bit CLEAR */
int main(void){printf("%u\n", 0xB0000u);return 0;}
