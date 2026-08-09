#include <stdio.h>
/* correct hardware: SBX_MAGIC with every arm bit CLEAR */
int main(void){printf("%u\n", 0xD0000000u);return 0;}
