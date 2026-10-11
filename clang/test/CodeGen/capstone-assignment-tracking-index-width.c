// C-50: with debug info at -O1 and above, Assignment Tracking accumulated the
// offset of a local's address at the POINTER width (128 for a capability)
// where stripAndAccumulate... requires the index width (64), and asserted on
// any local whose address escapes. Two shapes that asserted, and one control
// (a local SROA removes) that never did.
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -debug-info-kind=limited -O1 -emit-obj -o /dev/null %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -debug-info-kind=limited -O2 -emit-obj -o /dev/null %s

void g(char *);
void escaping_local(void) { char buf[32]; buf[5] = 1; g(buf); }

struct S { void *p; long x; };
void h(struct S *);
long escaping_struct(void *q) { struct S s; s.p = q; s.x = 1; h(&s); return s.x; }

long promoted_local(void) { struct { long a, b; } s; s.b = 2; s.a = 1; return s.a + s.b; }
