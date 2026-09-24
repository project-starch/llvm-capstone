struct S { long a; long b; };
long f(void) { struct S s; s.b = 2; s.a = 1; return s.a + s.b; }
