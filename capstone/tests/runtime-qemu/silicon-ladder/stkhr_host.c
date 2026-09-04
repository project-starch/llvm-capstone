/* Oracle for the stkhr probe. This one is not computed, it is MEASURED: the domain
   reports how much stack the entry glue leaves below the first frame, and this file
   pins the number so a change to the glue or the linker script shows up as a red
   rung instead of as a mysterious deep-recursion failure somewhere else.
   Measured 2026-08-16 under QEMU with the generated glue. */
#include <stdio.h>
int main(void) { printf("125120\n"); return 0; }
