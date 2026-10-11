/* The other translation unit: defines `other` and takes its address here, so the
   test can compare it with the address the first unit computes. */
__thread int other = 1234;
int *other_address(void) { return &other; }
/* The address as a number, from a unit that cannot see which object it is, so
   the caller cannot fold an alignment test on it. */
unsigned long address_of(void *p) { return (unsigned long)p; }
