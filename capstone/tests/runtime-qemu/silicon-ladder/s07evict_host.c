/* Native oracle: on correct hardware every spilled capability survives eviction, mask = 0xFFFF. */
#include <stdio.h>
int main(void) { printf("%u\n", 65535u); return 0; }
