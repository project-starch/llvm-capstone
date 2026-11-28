/* C-56: the address of an undefined weak symbol must be null in a domain. */
#include <stdio.h>
extern int hook(int) __attribute__((weak));
extern int weak_obj __attribute__((weak));
extern int dhook(int) __attribute__((weak));     /* defined in weak2.c */
extern int dweak_obj __attribute__((weak));      /* defined in weak2.c */
int defined_obj = 5;
int *volatile stored;
static int fails;
#define CHECK(c, what) do { if (c) printf("ok   %s\n", what); else { printf("FAIL %s\n", what); fails++; } } while (0)
int main(void) {
	CHECK(defined_obj == 5, "control: defined object reads 5");
	CHECK(&dweak_obj != 0 && dweak_obj == 9, "defined weak object is non-null and reads 9");
	CHECK(dhook != 0 && dhook(1) == 2, "defined weak function is non-null and returns 2");
	CHECK(!hook, "undefined weak function address is null");
	stored = &weak_obj;
	CHECK(!stored, "undefined weak object address, stored and reloaded, is null");
	CHECK(!&weak_obj, "undefined weak object address is null");
	printf("C56-DONE failures=%d\n", fails);
	return fails;
}
