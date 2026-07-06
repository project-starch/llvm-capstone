#include <stddef.h>
#include <stdint.h>

int g[64];

/* Q3: ptr -> int -> ptr round trip */
int *roundtrip(int *p) {
    uintptr_t x = (uintptr_t)p;   /* demote to integer */
    x += 4;                       /* integer arithmetic */
    return (int *)x;              /* re-promote */
}

/* Q3b: pure forge from an integer */
int *forge(unsigned long x) {
    return (int *)x;
}

/* Q4: pointer difference into an integer */
long pdiff(int *a, int *b) {
    return a - b;
}

/* Q8: the array example */
char a[64];
char deref_in_bounds(void) {
    char *p = &a[10];
    char *q = p + 5;     /* &a[15] */
    return *q;
}
char deref_out_of_object(void) {
    char *p = &a[10];
    char *q = p + 25;    /* &a[35] still within a[64] */
    return *q;
}
char deref_attacker(long i) {
    char *p = &a[10];
    char *q = p + i;     /* attacker-controlled offset */
    return *q;
}

/* capability creation for a global element */
int *take_global(long i) {
    return &g[i];
}
