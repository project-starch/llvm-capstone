/* Native oracle: every chunk must copy correctly, so the count equals the chunk count.
   Computed from the same constants the kernel uses, not transcribed. */
#include <stdio.h>
#define S06SCALE_BYTES (32u * 1024u)
int main(void) { printf("%u\n", S06SCALE_BYTES / 16u); return 0; }
