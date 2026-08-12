/* Native oracle. LCC does not exist on the host, so this prints the EXPECTED verdict rather than
   computing it: a correct machine copies the capability and the plain data intact. */
#include <stdio.h>
int main(void) { printf("15\n"); return 0; }
