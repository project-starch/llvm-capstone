/* Native oracle for the s07chase rung: on correct hardware every loaded link is a capability,
   so the count of untagged loads is 0. */
#include <stdio.h>
int main(void) { printf("%u\n", 0u); return 0; }
