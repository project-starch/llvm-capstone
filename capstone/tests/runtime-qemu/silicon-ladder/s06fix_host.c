/* Native oracle: both fields must survive an aggregate assignment, so 64. Silicon with S-06
 * returns 66 -- the high half of the plain 16-byte chunk (y) is lost. */
#include <stdio.h>
int main(void){printf("64\n");return 0;}
