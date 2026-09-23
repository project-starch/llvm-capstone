struct S { void *p; long x; };
void g(struct S *);
long f(void *q) { struct S s; s.p = q; s.x = 1; g(&s); return s.x; }
