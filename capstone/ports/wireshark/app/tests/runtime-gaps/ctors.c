/* ISSUES C-64: constructors and destructors in a domain. Two of each; the output shows which ran,
 * and in what order. Natively: CTOR a 1, CTOR b 2, MAIN 3, DTOR b 4, DTOR a 5. */
#include <stdio.h>
static int order;
__attribute__((constructor)) static void ctor_a(void) { printf("CTOR a %d\n", ++order); }
__attribute__((constructor)) static void ctor_b(void) { printf("CTOR b %d\n", ++order); }
__attribute__((destructor)) static void dtor_a(void) { printf("DTOR a %d\n", ++order); }
__attribute__((destructor)) static void dtor_b(void) { printf("DTOR b %d\n", ++order); }
int main(void) { printf("MAIN %d\n", ++order); return 0; }
